""" TruthfulnessClassifier -- train/predict/modify (Functions 1-3) """

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from . import config, features
from .logging_utils import logger, blank
from .modification import find_minimal_modification


def _normalize_record(record: str | dict) -> dict:
    # spec says "submit a json object" -- accept a JSON string or a dict
    if isinstance(record, str):
        return json.loads(record)
    return record


def _uncertain_label_sample_weight(label_series: pd.Series, weight: float) -> np.ndarray:
    # half-true/barely-true get `weight`, everything else gets 1.0.
    # Training rows only -- validation stays unweighted.
    lower = label_series.str.lower().str.strip()
    sw = np.ones(len(label_series))
    sw[lower.isin(config.UNCERTAIN_LABELS).values] = weight
    return sw


class TruthfulnessClassifier:
    def __init__(self, random_seed: int = config.RANDOM_SEED):
        self.random_seed = random_seed
        self.model: LGBMClassifier | None = None
        self.feature_bundle: features.FeatureBundle | None = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------ #
    # Function 1: Training
    # ------------------------------------------------------------------ #
    def train(
        self,
        data: str | Path | pd.DataFrame,
        val_size: float = 0.2,
        cv_folds: int | None = None,
        uncertain_label_weight: float = config.DEFAULT_UNCERTAIN_LABEL_WEIGHT,
        **lgbm_kwargs,
    ) -> dict:
        """
        Train and validate. 
        `data` is a path to a CSV shaped like data.csv.
        cv_folds: if set, also runs full CV for a mean/std estimate before fitting the model that actually gets stored. Slower, more honest.
        uncertain_label_weight: sample weight for half-true/barely-true rows during training (default 0.5, set to 1.0 to disable).
        """
        df = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data.copy()
        df.columns = [c.strip() for c in df.columns]
        label_col = "Label" if "Label" in df.columns else "label"
        if label_col not in df.columns:
            raise ValueError("Training data must contain a 'label'/'Label' column.")

        logger.info("=== Function 1 [train] called ===")
        logger.info("Purpose: fits the LightGBM classifier on labeled training data.")
        logger.info(f"Loaded {len(df)} rows.")

        y = df[label_col].map(config.label_to_binary).astype(int).values
        groups = df["speaker_name"].fillna(config.MISSING_TOKEN).astype(str).str.lower()

        n_splits = cv_folds if cv_folds else max(2, round(1 / val_size))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        splits = list(sgkf.split(df, y, groups=groups))

        cv_summary = {}
        if cv_folds:
            logger.info(f"cv_folds={cv_folds}: will also run full cross-validation for a statistically honest mean/std before fitting the final model.")
            logger.info(f"Step: running {cv_folds}-fold StratifiedGroupKFold cross-validation...")
            fold_metrics = {"accuracy": [], "f1": [], "roc_auc": []}
            for fold_i, (tr_i, val_i) in enumerate(splits):
                df_tr, df_v = df.iloc[tr_i].reset_index(drop=True), df.iloc[val_i].reset_index(drop=True)
                y_tr, y_v = y[tr_i], y[val_i]
                X_tr, fold_bundle = features.fit_transform(df_tr, y_tr)
                X_v = features.transform(df_v, fold_bundle)
                sw_tr = _uncertain_label_sample_weight(df_tr[label_col], uncertain_label_weight)
                fold_params = dict(
                    config.DEFAULT_LGBM_PARAMS,
                    class_weight="balanced",
                    random_state=self.random_seed,
                    n_jobs=-1,
                )
                fold_params.update(lgbm_kwargs)
                fold_model = LGBMClassifier(**fold_params)
                fold_model.fit(X_tr, y_tr, sample_weight=sw_tr)
                p = fold_model.predict_proba(X_v)[:, 1]
                fold_metrics["accuracy"].append(accuracy_score(y_v, p >= 0.5))
                fold_metrics["f1"].append(f1_score(y_v, p >= 0.5))
                fold_metrics["roc_auc"].append(roc_auc_score(y_v, p))
                logger.debug(f"  fold {fold_i}: accuracy={fold_metrics['accuracy'][-1]:.4f}")
            for k, vals in fold_metrics.items():
                cv_summary[f"cv_{k}_mean"] = float(np.mean(vals))
                cv_summary[f"cv_{k}_std"] = float(np.std(vals))
            cv_summary["cv_n_folds"] = cv_folds
            logger.info(f"CV result: accuracy = {cv_summary['cv_accuracy_mean']:.4f} +/- {cv_summary['cv_accuracy_std']:.4f}")

        # production model trained on the first fold's split
        tr_idx, val_idx = splits[0]
        df_train, df_val = df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
        y_train, y_val = y[tr_idx], y[val_idx]

        logger.info(f"Step: building features for the final production model (features.fit_transform) -- {len(df_train)} train rows, {len(df_val)} validation rows.")
        X_train, bundle = features.fit_transform(df_train, y_train)
        X_val = features.transform(df_val, bundle)

        params = dict(
            config.DEFAULT_LGBM_PARAMS,
            class_weight="balanced",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        params.update(lgbm_kwargs)
        model = LGBMClassifier(**params)
        sw_train = _uncertain_label_sample_weight(df_train[label_col], uncertain_label_weight)
        logger.info(f"Step: fitting LightGBM ({params['n_estimators']} trees) with sample-weighted labels (uncertain_label_weight={uncertain_label_weight})...")
        model.fit(X_train, y_train, sample_weight=sw_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)

        feature_names = bundle.feature_names()
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        top_features = [
            {"feature": feature_names[i], "importance": int(importances[i])}
            for i in top_idx
        ]

        metrics = {
            "n_train": len(df_train),
            "n_val": len(df_val),
            "validation_split": "speaker_grouped + label_stratified (StratifiedGroupKFold): every validation speaker is unseen in training, label balance preserved",
            "uncertain_label_weight": uncertain_label_weight,
            "n_uncertain_train_rows": int((sw_train != 1.0).sum()),
            "accuracy": accuracy_score(y_val, val_pred),
            "precision": precision_score(y_val, val_pred),
            "recall": recall_score(y_val, val_pred),
            "f1": f1_score(y_val, val_pred),
            "roc_auc": roc_auc_score(y_val, val_proba),
            "positive_rate_train": float(np.mean(y_train)),
            "positive_rate_val": float(np.mean(y_val)),
            "feature_importance_top10": top_features,
            **cv_summary,
        }

        self.model = model
        self.feature_bundle = bundle
        self._feature_names = bundle.feature_names()
        logger.info(f"Function 1 [train] result: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}")
        blank()
        return metrics

    # ------------------------------------------------------------------ #
    # Function 2: Prediction
    # ------------------------------------------------------------------ #
    def predict_proba(self, record: str | dict) -> float:
        self._check_fitted()
        record = _normalize_record(record)
        logger.info("--- Function 2 [predict_proba] called ---")
        logger.info("Purpose: returns the model's raw probability that this statement is truthful.")
        df = features.record_to_df(record)
        logger.debug("Step: validated record and converted to a 1-row DataFrame (features.record_to_df).")
        X = features.transform(df, self.feature_bundle)
        logger.debug("Step: built the full feature vector -- TF-IDF, embeddings, categorical encodings, speaker lookup (features.transform).")
        proba = float(self.model.predict_proba(X)[0, 1])
        logger.info(f"Result: probability(True) = {proba:.4f}")
        blank()
        return proba

    def predict(self, record: str | dict) -> bool:
        return self.predict_proba(record) >= 0.5

    def _predict_proba_cached(self, record: dict, cache: dict) -> float:
        # internal fast path for modify()'s search -- see features.transform_single_cached
        self._check_fitted()
        X = features.transform_single_cached(record, self.feature_bundle, cache)
        return float(self.model.predict_proba(X)[0, 1])

    def _predict_cached(self, record: dict, cache: dict) -> bool:
        return self._predict_proba_cached(record, cache) >= 0.5

    # ------------------------------------------------------------------ #
    # Function 3: Modification
    # ------------------------------------------------------------------ #
    def modify(self, record: str | dict, max_word_edits: int = 12) -> dict:
        """
        Finds the smallest change to statement/statement_context/subjectsthat flips the prediction.
        """
        self._check_fitted()
        record = _normalize_record(record)
        logger.info("=== Function 3 [modify] called ===")
        logger.info("Purpose: finds the smallest edit to statement/statement_context/subjects that flips the True/False prediction.")
        result = find_minimal_modification(self, record, max_word_edits=max_word_edits)
        logger.info(f"Function 3 [modify] result: success={result['success']}, flip_type={result['flip_type']}, n_edits={result['n_edits']}")
        blank()
        return result

    # ------------------------------------------------------------------ #
    def _check_fitted(self):
        if self.model is None or self.feature_bundle is None:
            raise RuntimeError("Model is not trained. Call .train(...) first, or .load(...).")

    def save(self, path: str | Path):
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "bundle": self.feature_bundle}, f)

    @classmethod
    def load(cls, path: str | Path) -> "TruthfulnessClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        clf = cls()
        clf.model = obj["model"]
        clf.feature_bundle = obj["bundle"]
        clf._feature_names = obj["bundle"].feature_names()
        return clf
