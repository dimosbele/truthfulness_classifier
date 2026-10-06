"""
Hyperparameter search for the LightGBM model. 
Needs data.csv in the working directory. Results in tuning_results.md.
"""

import random
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from truthguard import config, features
from truthguard.model import _uncertain_label_sample_weight

BASELINE_PARAMS = dict(
    n_estimators=700, learning_rate=0.03, num_leaves=63, min_child_samples=10,
    reg_lambda=1.0, colsample_bytree=0.6, subsample=0.8,
)

SEARCH_SPACE = {
    "n_estimators": [300, 500, 700],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "num_leaves": [31, 63, 95, 127],
    "min_child_samples": [5, 10, 20, 30],
    "reg_lambda": [0.1, 0.5, 1.0, 3.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "subsample": [0.7, 0.8, 0.9, 1.0],
}

# winner from the screening step -- also what model.py uses
WINNING_PARAMS = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=30,
    reg_lambda=0.1, colsample_bytree=0.5, subsample=0.8,
)


def _get_folds(df, y):
    groups = df["speaker_name"].fillna("unknown").astype(str).str.lower()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    return list(sgkf.split(df, y, groups=groups))


def _fit_eval(X_train, y_train, sw, X_val, y_val, params):
    m = LGBMClassifier(class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1, **params)
    m.fit(X_train, y_train, sample_weight=sw)
    p = m.predict_proba(X_val)[:, 1]
    return {
        "accuracy": accuracy_score(y_val, p >= 0.5),
        "f1": f1_score(y_val, p >= 0.5),
        "roc_auc": roc_auc_score(y_val, p),
    }


def screen(n_trials: int = 10, seed: int = 7):
    """Step 1: scoped random search on a single fold."""
    df = pd.read_csv("data.csv")
    y_all = df["Label"].map(config.label_to_binary).astype(int).values
    tr_idx, val_idx = _get_folds(df, y_all)[0]
    df_train, df_val = df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
    y_train, y_val = y_all[tr_idx], y_all[val_idx]

    X_train, bundle = features.fit_transform(df_train, y_train)
    X_val = features.transform(df_val, bundle)
    sw = _uncertain_label_sample_weight(df_train["Label"], config.DEFAULT_UNCERTAIN_LABEL_WEIGHT)

    random.seed(seed)
    trials = [("BASELINE", BASELINE_PARAMS)] + [
        (f"trial{i+1}", {k: random.choice(v) for k, v in SEARCH_SPACE.items()})
        for i in range(n_trials)
    ]

    results = []
    for tag, params in trials:
        t0 = time.time()
        metrics = _fit_eval(X_train, y_train, sw, X_val, y_val, params)
        elapsed = time.time() - t0
        results.append({"tag": tag, **params, **metrics, "seconds": round(elapsed)})
        print(f"{tag} ({elapsed:.0f}s): acc={metrics['accuracy']:.4f} "
              f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f} | {params}")

    results.sort(key=lambda r: r["accuracy"], reverse=True)
    print("\n=== ranked by accuracy (fold 0 only -- confirm winner across all folds before trusting) ===")
    for r in results:
        print(f"{r['tag']}: acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['roc_auc']:.4f} ({r['seconds']}s)")
    return results


def confirm(candidate_params: dict = WINNING_PARAMS, baseline_params: dict = BASELINE_PARAMS):
    """Step 2: full 5-fold comparison of the screening winner vs. baseline."""
    df = pd.read_csv("data.csv")
    y_all = df["Label"].map(config.label_to_binary).astype(int).values
    folds = _get_folds(df, y_all)

    rows = []
    for fold_i, (tr_idx, val_idx) in enumerate(folds):
        df_train, df_val = df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
        y_train, y_val = y_all[tr_idx], y_all[val_idx]
        X_train, bundle = features.fit_transform(df_train, y_train)
        X_val = features.transform(df_val, bundle)
        sw = _uncertain_label_sample_weight(df_train["Label"], config.DEFAULT_UNCERTAIN_LABEL_WEIGHT)

        for tag, params in [("baseline", baseline_params), ("candidate", candidate_params)]:
            m = _fit_eval(X_train, y_train, sw, X_val, y_val, params)
            rows.append({"fold": fold_i, "tag": tag, **m})
            print(f"fold {fold_i} {tag}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

    df_r = pd.DataFrame(rows)
    summary = df_r.groupby("tag")[["accuracy", "f1", "roc_auc"]].agg(["mean", "std"])
    print("\n=== 5-fold summary ===")
    print(summary)
    return df_r


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "screen":
        screen()
    elif mode == "confirm":
        confirm()
    else:
        raise SystemExit("usage: tune_hyperparameters.py [screen|confirm]")
