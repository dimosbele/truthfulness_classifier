""" 
Function 3: find the smallest change to a record that flips the prediction. 
Search is restricted to statement / statement_context /subjects -- speaker_name/affiliation/state are excluded on purpose
Two stages: try every single-edit candidate first (word removal/ substitution, or a subjects tag edit), if none flips it, 
fall back to a greedy multi-edit search.
"""

from __future__ import annotations

import re

import nltk
import spacy
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

from .logging_utils import logger

_STOPWORDS = None

_TOKEN_RE = re.compile(r"\S+")
_NUMERIC_RE = re.compile(r"^\d[\d,]*$")


def _ensure_wordnet():
    # download once if missing -- reading wn.NOUN etc. below triggers the
    # lazy corpus load, so this has to run first
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        wn.ensure_loaded()


_ensure_wordnet()

_WN_POS_MAP = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}

_POS_NLP = None


def _get_pos_nlp():
    global _POS_NLP
    if _POS_NLP is None:
        _POS_NLP = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer"])
    return _POS_NLP


def _stopwords() -> set[str]:
    # spaCy's list (326 words) instead of a hand-picked one -- reuses the
    # already-loaded model, no extra cost
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = _get_pos_nlp().Defaults.stop_words
    return _STOPWORDS


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def _detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def _lesk_synonyms(sentence_tokens: list[str], word: str, spacy_pos: str, max_n: int = 3) -> list[str]:
    wpos = _WN_POS_MAP.get(spacy_pos)
    if wpos is None:
        return []
    try:
        best_synset = lesk(sentence_tokens, word.lower(), pos=wpos)
    except Exception:
        return []
    if best_synset is None:
        return []
    candidates: list[str] = []
    seen = {word.lower()}
    for lemma in best_synset.lemmas():
        name = lemma.name().replace("_", " ")
        if name.lower() not in seen and " " not in name:
            candidates.append(name)
            seen.add(name.lower())
        if len(candidates) >= max_n:
            break
    return candidates


def _numeric_scale_candidates(token: str) -> list[str]:
    bare = token.strip(".,!?'\"")
    trailing = token[len(bare):]
    cleaned = bare.replace(",", "")
    if not _NUMERIC_RE.match(cleaned):
        return []
    n = int(cleaned)
    if n < 2:
        return []
    return [f"{scaled:,}{trailing}" for scaled in {max(1, n // 2), n * 2}]


def _substitution_candidates(sentence_tokens: list[str], idx: int, spacy_pos_tags: list[str]) -> list[str]:
    token = sentence_tokens[idx]
    bare = token.strip(".,!?'\"")
    trailing = token[len(bare):]
    out = []
    for repl in _lesk_synonyms(sentence_tokens, bare, spacy_pos_tags[idx]):
        out.append(repl.capitalize() + trailing if bare[:1].isupper() else repl + trailing)
    out.extend(_numeric_scale_candidates(token))
    return out


def _tag_pos(tokens: list[str]) -> list[str]:
    # spaCy re-tokenizes the joined string; if it doesn't line up with our
    # whitespace tokens 1:1 just bail with "X" (no POS) for that word
    doc = _get_pos_nlp()(_detokenize(tokens))
    if len(doc) == len(tokens):
        return [t.pos_ for t in doc]
    return ["X"] * len(tokens)


def _word_edit_candidates(tokens: list[str], field_name: str, record: dict) -> list[dict]:
    pos_tags = _tag_pos(tokens)
    out = []
    for i, tok in enumerate(tokens):
        bare = tok.strip(".,!?'\"")
        if bare.lower() in _stopwords() or len(bare) <= 2:
            continue
        remaining = tokens[:i] + tokens[i + 1:]
        trial = dict(record)
        trial[field_name] = _detokenize(remaining)
        out.append({"kind": "word_removal", "field": field_name, "word": tok, "index": i, "record": trial})
        for repl in _substitution_candidates(tokens, i, pos_tags):
            new_tokens = tokens[:i] + [repl] + tokens[i + 1:]
            trial2 = dict(record)
            trial2[field_name] = _detokenize(new_tokens)
            out.append({"kind": "word_substitution", "field": field_name, "word": tok,
                        "replacement": repl, "index": i, "record": trial2})
    return out


def _subjects_candidates(clf, record: dict) -> list[dict]:
    from . import features

    current_tags = features.split_subjects(record.get("subjects", "") or "")
    vocab = list(getattr(clf.feature_bundle.subjects_encoder, "vocabulary_", []))
    out = []

    for i, tag in enumerate(current_tags):
        if len(current_tags) >= 2:  # keep at least one tag
            remaining = current_tags[:i] + current_tags[i + 1:]
            trial = dict(record)
            trial["subjects"] = features.config.SUBJECTS_SEP.join(remaining)
            out.append({"kind": "subjects_edit", "field": "subjects",
                        "description": f"removed tag '{tag}'", "record": trial})
        for new_tag in vocab:
            if new_tag == tag:
                continue
            swapped = current_tags[:i] + [new_tag] + current_tags[i + 1:]
            trial = dict(record)
            trial["subjects"] = features.config.SUBJECTS_SEP.join(swapped)
            out.append({"kind": "subjects_edit", "field": "subjects",
                        "description": f"tag '{tag}' -> '{new_tag}'", "record": trial})

    for new_tag in vocab:
        if new_tag not in current_tags:
            added = current_tags + [new_tag]
            trial = dict(record)
            trial["subjects"] = features.config.SUBJECTS_SEP.join(added)
            out.append({"kind": "subjects_edit", "field": "subjects",
                        "description": f"added tag '{new_tag}'", "record": trial})

    return out


_KIND_PREFERENCE = {"word_substitution": 0, "word_removal": 1, "subjects_edit": 2}


def _search_single_edit(clf, record: dict, target: bool, cache: dict) -> dict | None:
    logger.info("Stage A [_search_single_edit] started.")
    logger.info("Purpose: generate every possible single-edit candidate (word removal/substitution in statement and statement_context, tag edits in subjects), then check if any one of them alone flips the prediction.")
    stmt_tokens = _tokenize(record.get("statement", "") or "")
    ctx_tokens = _tokenize(record.get("statement_context", "") or "")

    stmt_candidates = _word_edit_candidates(stmt_tokens, "statement", record)
    ctx_candidates = _word_edit_candidates(ctx_tokens, "statement_context", record)
    subj_candidates = _subjects_candidates(clf, record)
    candidates = stmt_candidates + ctx_candidates + subj_candidates
    logger.info(f"  Generated {len(stmt_candidates)} statement candidates, {len(ctx_candidates)} statement_context candidates, {len(subj_candidates)} subjects candidates ({len(candidates)} total).")

    logger.debug("  Evaluating all candidates via the cached prediction path (_predict_cached)...")
    flips = [c for c in candidates if clf._predict_cached(c["record"], cache) == target]
    if not flips:
        logger.info("Stage A result: no single edit flips the prediction. Proceeding to Stage B (greedy multi-edit).")
        return None

    flips.sort(key=lambda c: _KIND_PREFERENCE.get(c["kind"], 9))
    best = flips[0]

    def describe(c):
        if c["kind"] == "subjects_edit":
            return f"subjects: {c['description']}"
        if c["kind"] == "word_substitution":
            return f"{c['field']}: '{c['word']}' -> '{c['replacement']}'"
        return f"{c['field']}: removed '{c['word']}'"

    logger.info(f"Stage A result: {len(flips)} single-edit candidate(s) flip the prediction. Chosen (most natural, by preference order): {describe(best)}")
    if len(flips) > 1:
        logger.debug(f"  {len(flips)-1} other candidate(s) also flip at the same cost: {[describe(c) for c in flips[1:6]]}")

    return {
        "success": True,
        "flip_type": best["kind"],
        "changed_fields": [best["field"]],
        "n_edits": 1,
        "modified_record": best["record"],
        "note": f"Minimal single edit found: {describe(best)}.",
        "alternative_flips_found": len(flips) - 1,
        "alternative_edits": [describe(c) for c in flips[1:6]],
    }


def _greedy_multi_edit_search(clf, record: dict, target: bool, max_edits: int, cache: dict) -> dict:
    logger.info("Stage B [_greedy_multi_edit_search] started.")
    logger.info("Purpose: no single edit worked, so greedily apply one edit at a time -- at each step, re-evaluate every remaining word in statement + statement_context and pick whichever single change moves the probability most toward the target, then repeat.")
    stmt_tokens = _tokenize(record.get("statement", "") or "")
    ctx_tokens = _tokenize(record.get("statement_context", "") or "")
    if not stmt_tokens and not ctx_tokens:
        logger.info("Stage B result: both statement and statement_context are empty, nothing to search.")
        return {
            "success": False, "flip_type": None, "changed_fields": [], "n_edits": 0,
            "modified_record": dict(record),
            "note": "Both statement and statement_context are empty; no text-based modification possible.",
        }

    working = {"statement": list(stmt_tokens), "statement_context": list(ctx_tokens)}
    # cap edits per field so we don't grind a statement into nonsense
    # chasing a flip -- readability matters more than success rate here
    caps = {
        "statement": max(3, round(len(stmt_tokens) * 0.4)) if stmt_tokens else 0,
        "statement_context": max(2, round(len(ctx_tokens) * 0.4)) if ctx_tokens else 0,
    }
    logger.debug(f"  Readability caps: statement<={caps['statement']} edits, statement_context<={caps['statement_context']} edits.")
    used = {"statement": 0, "statement_context": 0}
    applied_edits: list[str] = []
    direction = 1 if target else -1

    def current_record():
        return {
            **record,
            "statement": _detokenize(working["statement"]),
            "statement_context": _detokenize(working["statement_context"]),
        }

    total_budget = max(3, max_edits)
    for step in range(total_budget):
        baseline = clf._predict_proba_cached(current_record(), cache)
        best_move = None  # (movement, field, desc, new_tokens)

        for field in ("statement", "statement_context"):
            if used[field] >= caps[field]:
                continue
            tokens = working[field]
            pos_tags = _tag_pos(tokens)
            for i, tok in enumerate(tokens):
                bare = tok.strip(".,!?'\"")
                if bare.lower() in _stopwords() or len(bare) <= 2:
                    continue
                variants = [("removed '" + tok + "'", tokens[:i] + tokens[i + 1:])]
                for repl in _substitution_candidates(tokens, i, pos_tags):
                    variants.append((f"'{tok}' -> '{repl}'", tokens[:i] + [repl] + tokens[i + 1:]))
                for desc, variant_tokens in variants:
                    trial = dict(working)
                    trial[field] = variant_tokens
                    p = clf._predict_proba_cached({**record, "statement": _detokenize(trial["statement"]),
                                                    "statement_context": _detokenize(trial["statement_context"])}, cache)
                    movement = direction * (p - baseline)
                    if best_move is None or movement > best_move[0]:
                        best_move = (movement, field, desc, variant_tokens)

        if best_move is None:
            logger.debug(f"  Step {step}: no further candidate edits available (readability caps reached).")
            break
        _, field, desc, new_tokens = best_move
        working[field] = new_tokens
        used[field] += 1
        applied_edits.append(f"{field}: {desc}")
        logger.debug(f"  Step {step}: applied {field}: {desc} (probability moved from {baseline:.4f})")

        trial_record = current_record()
        if clf._predict_cached(trial_record, cache) == target:
            logger.info(f"Stage B result: flipped after {len(applied_edits)} greedy edit(s): {'; '.join(applied_edits)}")
            return {
                "success": True,
                "flip_type": "greedy_multi_edit",
                "changed_fields": sorted({e.split(":")[0] for e in applied_edits}),
                "n_edits": len(applied_edits),
                "modified_record": trial_record,
                "note": f"Prediction flipped after {len(applied_edits)} greedy edit(s): "
                        + "; ".join(applied_edits) + ".",
            }

    logger.info(f"Stage B result: could not flip the prediction within the edit budget ({len(applied_edits)} edits tried).")
    return {
        "success": False,
        "flip_type": None,
        "changed_fields": [],
        "n_edits": len(applied_edits),
        "modified_record": current_record(),
        "note": (
            f"Could not flip the prediction within the edit budget (statement capped at "
            f"{caps['statement']}, statement_context at {caps['statement_context']} edits, "
            f"to keep results readable). The classification appears robust for this record."
        ),
    }


def find_minimal_modification(clf, record: dict, max_word_edits: int = 12) -> dict:
    from . import features

    logger.info("Step: validating the input record (features.record_to_df) before any search work.")
    features.record_to_df(record)

    # fresh cache per call, shared across both stages -- see
    # features.transform_single_cached
    cache: dict = {}

    logger.info("Step: calling Function 2 [predict] to get the original prediction (as the challenge spec requires) ...")
    original_pred = clf.predict(record)
    target = not original_pred
    logger.info(f"Original prediction: {original_pred}. Target for the search: {target}.")

    result = _search_single_edit(clf, record, target, cache)
    if result is None:
        result = _greedy_multi_edit_search(clf, record, target, max_edits=max_word_edits, cache=cache)

    result["original_prediction"] = original_pred
    result["target_prediction"] = target
    return result
