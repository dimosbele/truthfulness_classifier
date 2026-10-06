# Hyperparameter tuning results

Produced by `scripts/tune_hyperparameters.py`. Recorded here so the numbers
persist even though the search itself takes ~20 minutes to reproduce.

## Step 1: screening (fold 0 only, baseline + 10 random trials)

| Trial | n_estimators | learning_rate | num_leaves | min_child_samples | reg_lambda | colsample_bytree | subsample | Accuracy | F1 | ROC-AUC | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trial8 | 300 | 0.03 | 63 | 5 | 1.0 | 0.8 | 0.9 | 0.6167 | 0.6522 | 0.6468 | 53 |
| trial3 | 300 | 0.03 | 31 | 30 | 0.1 | 0.5 | 0.8 | 0.6157 | 0.6523 | 0.6494 | 24 |
| trial1 | 500 | 0.03 | 127 | 5 | 0.1 | 0.5 | 0.9 | 0.6115 | 0.6637 | 0.6418 | 146 |
| trial4 | 700 | 0.02 | 127 | 5 | 0.5 | 0.5 | 0.8 | 0.6094 | 0.6600 | 0.6453 | 199 |
| BASELINE | 700 | 0.03 | 63 | 10 | 1.0 | 0.6 | 0.8 | 0.6063 | 0.6491 | 0.6377 | 99 |
| trial5 | 500 | 0.08 | 63 | 5 | 1.0 | 0.6 | 0.7 | 0.6048 | 0.6511 | 0.6365 | 69 |
| trial2 | 700 | 0.02 | 63 | 5 | 0.1 | 0.8 | 1.0 | 0.6027 | 0.6460 | 0.6350 | 116 |
| trial7 | 500 | 0.08 | 95 | 30 | 3.0 | 0.7 | 0.9 | 0.6017 | 0.6503 | 0.6345 | 70 |
| trial6 | 700 | 0.03 | 95 | 5 | 0.1 | 0.5 | 0.8 | 0.5897 | 0.6429 | 0.6306 | 157 |
| trial9 | 700 | 0.08 | 95 | 5 | 0.1 | 0.8 | 0.8 | 0.5861 | 0.6383 | 0.6231 | 161 |

Top 2 candidates (trial8, trial3) both use fewer trees (300 vs. the
baseline's 700) and are both faster AND slightly more accurate on this
fold. `trial3` was carried forward to step 2 for being nearly identical in
accuracy to `trial8` (0.6157 vs 0.6167) while running over 2x faster
(24s vs 53s) and using simpler trees (`num_leaves=31` vs 63).

## Step 2: confirmation (all 5 folds, baseline vs. trial3)

| Fold | Baseline accuracy | Candidate (trial3) accuracy | Δ Accuracy | Δ F1 | Δ ROC-AUC |
|---|---|---|---|---|---|
| 0 | 0.6063 | 0.6157 | +0.94 | +0.32 | +1.17 |
| 1 | 0.6014 | 0.6133 | +1.19 | −0.34 | +1.65 |
| 2 | 0.6255 | 0.6145 | −1.10 | −2.52 | +0.46 |
| 3 | 0.6085 | 0.6070 | −0.15 | −2.49 | +0.93 |
| 4 | 0.6164 | 0.6123 | −0.41 | −1.30 | +0.41 |
| **Mean** | **0.6116** | **0.6126** | **+0.09** | **−1.27** | **+0.92** |

## Conclusion

Accuracy is statistically indistinguishable between baseline and the
candidate (+0.09 points average, well within fold-to-fold noise already
established elsewhere in this project). F1 is slightly worse for the
candidate on average (−1.27 points); ROC-AUC is consistently better
(+0.92 points, positive in all 5 folds).

**The candidate (`trial3` params) was adopted as the new default anyway --
not for accuracy, but for efficiency**: ~4x faster training (24s vs.
99-100s per fold) using 300 trees instead of 700, for performance that is
not meaningfully worse on the metric that matters most (accuracy) and is
consistently better on ROC-AUC. This is a deliberate engineering tradeoff,
documented as such.
