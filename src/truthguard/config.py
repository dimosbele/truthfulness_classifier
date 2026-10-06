# random_seed for TruthfulnessClassifier
RANDOM_SEED = 42  

# Binary label policy. Source data is a 6-way ordinal scale and we split at the midpoint for True/False
TRUE_LABELS = {"true", "mostly-true", "half-true"}
FALSE_LABELS = {"barely-true", "false", "extremely-false"}
ALL_LABELS = TRUE_LABELS | FALSE_LABELS

# For half-true/barely-true (noisiest labels) we use different weight compared to the rest.
UNCERTAIN_LABELS = {"half-true", "barely-true"}
DEFAULT_UNCERTAIN_LABEL_WEIGHT = 0.5

# Parameters are derived from scripts/tune_hyperparameters.py
DEFAULT_LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=30,
    reg_lambda=0.1,
    colsample_bytree=0.5,
    subsample=0.8,
)

TEXT_COLUMN = "statement"
CONTEXT_COLUMN = "statement_context"
SUBJECTS_COLUMN = "subjects"
SUBJECTS_SEP = "$"

CATEGORICAL_COLUMNS = ["speaker_affiliation", "speaker_state", "speaker_job"]

# every column in data.csv except the label
REQUIRED_FIELDS = [
    "statement",
    "subjects",
    "speaker_name",
    "speaker_job",
    "speaker_state",
    "speaker_affiliation",
    "statement_context",
]

MISSING_TOKEN = "unknown"


def label_to_binary(label: str) -> bool:
    """ Converts the 6 original labels to True or False. """
    key = str(label).strip().lower()
    if key in TRUE_LABELS:
        return True
    if key in FALSE_LABELS:
        return False
    raise ValueError(f"Unrecognised label '{label}'. Expected one of {sorted(ALL_LABELS)}.")
