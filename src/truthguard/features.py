""" Feature enngineering functions """

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import spacy
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from . import config

ABSOLUTIST_PATTERN = r"\b(always|never|all|none|every|totally|completely)\b"
EMBEDDING_MODEL_NAME = "en_core_web_md"

_NLP_CACHE: dict[str, "spacy.language.Language"] = {}


def _get_spacy_nlp(model_name: str = EMBEDDING_MODEL_NAME):
    # load once, reuse
    if model_name not in _NLP_CACHE:
        _NLP_CACHE[model_name] = spacy.load(
            model_name, disable=["parser", "ner", "tagger", "lemmatizer"]
        )
    return _NLP_CACHE[model_name]


def _embed_texts(texts: pd.Series, model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    nlp = _get_spacy_nlp(model_name)
    cleaned = texts.fillna("").astype(str)
    vectors = [doc.vector for doc in nlp.pipe(cleaned, batch_size=256)]
    return np.vstack(vectors).astype(np.float32)


def split_subjects(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw:
        return []
    return [s.strip() for s in raw.split(config.SUBJECTS_SEP) if s.strip()]


class MultiHotSubjectsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq

    def fit(self, X: pd.Series, y=None):
        all_subjects = X.fillna("").map(split_subjects)
        counts: dict[str, int] = {}
        for subs in all_subjects:
            for s in subs:
                counts[s] = counts.get(s, 0) + 1
        self.vocabulary_ = sorted(s for s, c in counts.items() if c >= self.min_freq)
        self.index_ = {s: i for i, s in enumerate(self.vocabulary_)}
        return self

    def transform(self, X: pd.Series):
        rows, cols = [], []
        for i, raw in enumerate(X.fillna("")):
            for s in split_subjects(raw):
                j = self.index_.get(s)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        data = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(X), len(self.vocabulary_)))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"subject__{s}" for s in self.vocabulary_])


def _engineer_text_stats(statements: pd.Series) -> np.ndarray:
    s = statements.fillna("")
    return np.column_stack([
        s.str.len().values.astype(np.float32),
        s.str.count(r"\d").values.astype(np.float32),
        s.str.count("!").values.astype(np.float32),
        s.str.count(r"\?").values.astype(np.float32),
        s.str.lower().str.count(ABSOLUTIST_PATTERN).values.astype(np.float32),
    ])


def _speaker_stats(names: pd.Series, y: np.ndarray) -> pd.DataFrame:
    tmp = pd.DataFrame({"speaker": names.fillna(config.MISSING_TOKEN).str.lower(), "y": y})
    return tmp.groupby("speaker")["y"].agg(["mean", "count"])


def _oof_speaker_truth_rate(
    names: pd.Series, y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    # out-of-fold so a row never sees its own label through this feature
    names = names.reset_index(drop=True)
    y = np.asarray(y)
    out = np.full(len(names), np.nan, dtype=np.float32)
    global_rate = float(y.mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, ho_idx in kf.split(names):
        stats = _speaker_stats(names.iloc[tr_idx], y[tr_idx])
        spk_ho = names.iloc[ho_idx].fillna(config.MISSING_TOKEN).str.lower()
        out[ho_idx] = spk_ho.map(stats["mean"]).fillna(global_rate).values
    return out


@dataclass
class FeatureBundle:
    text_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    context_vectorizer: TfidfVectorizer
    subjects_encoder: MultiHotSubjectsEncoder
    cat_encoder: OneHotEncoder
    speaker_stats: pd.DataFrame  # lowercased name -> mean, count
    global_truth_rate: float
    embedding_model_name: str = EMBEDDING_MODEL_NAME

    def feature_names(self) -> list[str]:
        names = ["word__" + n for n in self.text_vectorizer.get_feature_names_out()]
        names += ["char__" + n for n in self.char_vectorizer.get_feature_names_out()]
        names += ["ctx__" + n for n in self.context_vectorizer.get_feature_names_out()]
        names += list(self.subjects_encoder.get_feature_names_out())
        names += list(self.cat_encoder.get_feature_names_out(config.CATEGORICAL_COLUMNS))
        names += ["stat__len", "stat__digit_count", "stat__exclaim_count",
                  "stat__question_count", "stat__absolutist_count"]
        names += ["speaker_truth_rate", "speaker_statement_count"]
        names += [f"stmt_emb__{i:03d}" for i in range(300)]
        names += [f"ctx_emb__{i:03d}" for i in range(300)]
        return names


def _prep_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df[config.CATEGORICAL_COLUMNS].copy()
    for col in config.CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna(config.MISSING_TOKEN).astype(str).str.lower().str.strip()
    return out


def _build_text_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )


def _build_char_vectorizer(max_features: int = 3000) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )


def _speaker_feature_columns(names: pd.Series, bundle: "FeatureBundle") -> np.ndarray:
    lower = names.fillna(config.MISSING_TOKEN).str.lower()
    rate = lower.map(bundle.speaker_stats["mean"]).fillna(bundle.global_truth_rate).values
    cnt = lower.map(bundle.speaker_stats["count"]).fillna(0).values
    return np.column_stack([rate, cnt]).astype(np.float32)


def fit_transform(df: pd.DataFrame, y: np.ndarray) -> tuple[sparse.csr_matrix, "FeatureBundle"]:
    # y is needed here (not just in transform) because speaker_truth_rate
    # is target-derived and has to be computed OOF on the training data.
    text_vec = _build_text_vectorizer(max_features=5000)
    X_text = text_vec.fit_transform(df[config.TEXT_COLUMN].fillna(""))

    char_vec = _build_char_vectorizer(max_features=3000)
    X_char = char_vec.fit_transform(df[config.TEXT_COLUMN].fillna(""))

    ctx_vec = _build_text_vectorizer(max_features=1000)
    X_ctx = ctx_vec.fit_transform(df[config.CONTEXT_COLUMN].fillna(""))

    subj_enc = MultiHotSubjectsEncoder(min_freq=5).fit(df[config.SUBJECTS_COLUMN])
    X_subj = subj_enc.transform(df[config.SUBJECTS_COLUMN])

    cats = _prep_categoricals(df)
    cat_enc = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=True)
    X_cat = cat_enc.fit_transform(cats)

    X_eng = _engineer_text_stats(df[config.TEXT_COLUMN])

    speaker_stats = _speaker_stats(df["speaker_name"], y)
    global_truth_rate = float(np.mean(y))
    oof_rate = _oof_speaker_truth_rate(df["speaker_name"], y)
    oof_count = df["speaker_name"].fillna(config.MISSING_TOKEN).str.lower().map(
        speaker_stats["count"]
    ).fillna(0).values
    X_speaker = np.column_stack([oof_rate, oof_count]).astype(np.float32)

    # embeddings on statement + context only, added alongside the TF-IDF
    X_stmt_emb = _embed_texts(df[config.TEXT_COLUMN])
    X_ctx_emb = _embed_texts(df[config.CONTEXT_COLUMN])

    X = sparse.hstack([
        X_text, X_char, X_ctx, X_subj, X_cat,
        sparse.csr_matrix(X_eng), sparse.csr_matrix(X_speaker),
        sparse.csr_matrix(X_stmt_emb), sparse.csr_matrix(X_ctx_emb),
    ]).tocsr()

    bundle = FeatureBundle(
        text_vec, char_vec, ctx_vec, subj_enc, cat_enc, speaker_stats, global_truth_rate
    )
    return X, bundle


def transform(df: pd.DataFrame, bundle: "FeatureBundle") -> sparse.csr_matrix:
    X_text = bundle.text_vectorizer.transform(df[config.TEXT_COLUMN].fillna(""))
    X_char = bundle.char_vectorizer.transform(df[config.TEXT_COLUMN].fillna(""))
    X_ctx = bundle.context_vectorizer.transform(df[config.CONTEXT_COLUMN].fillna(""))
    X_subj = bundle.subjects_encoder.transform(df[config.SUBJECTS_COLUMN])
    cats = _prep_categoricals(df)
    X_cat = bundle.cat_encoder.transform(cats)
    X_eng = _engineer_text_stats(df[config.TEXT_COLUMN])
    X_speaker = _speaker_feature_columns(df["speaker_name"], bundle)
    X_stmt_emb = _embed_texts(df[config.TEXT_COLUMN], bundle.embedding_model_name)
    X_ctx_emb = _embed_texts(df[config.CONTEXT_COLUMN], bundle.embedding_model_name)
    return sparse.hstack([
        X_text, X_char, X_ctx, X_subj, X_cat,
        sparse.csr_matrix(X_eng), sparse.csr_matrix(X_speaker),
        sparse.csr_matrix(X_stmt_emb), sparse.csr_matrix(X_ctx_emb),
    ]).tocsr()


def record_to_df(record: dict) -> pd.DataFrame:
    missing = [f for f in config.REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"Record is missing required field(s): {missing}")
    return pd.DataFrame([{f: record.get(f) for f in config.REQUIRED_FIELDS}])


def transform_single_cached(record: dict, bundle: "FeatureBundle", cache: dict) -> sparse.csr_matrix:
    """ Same as transform() but for one record, reusing cache across calls."""
    def cached(key, compute_fn):
        if key not in cache:
            cache[key] = compute_fn()
        return cache[key]

    stmt = record.get(config.TEXT_COLUMN) or ""
    ctx = record.get(config.CONTEXT_COLUMN) or ""
    subj = record.get(config.SUBJECTS_COLUMN) or ""
    speaker = record.get("speaker_name")
    cat_key = tuple(record.get(c) for c in config.CATEGORICAL_COLUMNS)

    X_text = cached(("word", stmt), lambda: bundle.text_vectorizer.transform([stmt]))
    X_char = cached(("char", stmt), lambda: bundle.char_vectorizer.transform([stmt]))
    X_stmt_emb = cached(("stmt_emb", stmt), lambda: _embed_texts(pd.Series([stmt]), bundle.embedding_model_name))
    X_eng = cached(("eng", stmt), lambda: _engineer_text_stats(pd.Series([stmt])))

    X_ctx = cached(("ctx", ctx), lambda: bundle.context_vectorizer.transform([ctx]))
    X_ctx_emb = cached(("ctx_emb", ctx), lambda: _embed_texts(pd.Series([ctx]), bundle.embedding_model_name))

    X_subj = cached(("subj", subj), lambda: bundle.subjects_encoder.transform(pd.Series([subj])))

    X_cat = cached(
        ("cat", cat_key),
        lambda: bundle.cat_encoder.transform(_prep_categoricals(pd.DataFrame([record]))),
    )

    X_speaker = cached(("spk", speaker), lambda: _speaker_feature_columns(pd.Series([speaker]), bundle))

    return sparse.hstack([
        X_text, X_char, X_ctx, X_subj, X_cat,
        sparse.csr_matrix(X_eng), sparse.csr_matrix(X_speaker),
        sparse.csr_matrix(X_stmt_emb), sparse.csr_matrix(X_ctx_emb),
    ]).tocsr()
