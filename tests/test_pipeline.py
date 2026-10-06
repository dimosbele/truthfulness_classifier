"""
Functions created and used only for testing purposes. 
This file is not used from the main Python package.
"""
import pandas as pd
import pytest

from truthguard import TruthfulnessClassifier
from truthguard.config import label_to_binary

FIELDS = [
    "statement", "subjects", "speaker_name", "speaker_job",
    "speaker_state", "speaker_affiliation", "statement_context",
]


@pytest.fixture(scope="module")
def trained_clf():
    df = pd.read_csv("data.csv")
    clf = TruthfulnessClassifier()
    clf.train(df.sample(1500, random_state=0))  # small subset: fast tests
    return clf


def test_label_mapping():
    assert label_to_binary("true") is True
    assert label_to_binary("mostly-true") is True
    assert label_to_binary("half-true") is True
    assert label_to_binary("barely-true") is False
    assert label_to_binary("false") is False
    assert label_to_binary("extremely-false") is False
    with pytest.raises(ValueError):
        label_to_binary("not-a-real-label")


def test_train_returns_metrics(trained_clf):
    # trained_clf fixture already asserts train() doesn't raise;
    # spot-check the model/bundle are populated.
    assert trained_clf.model is not None
    assert trained_clf.feature_bundle is not None


def test_predict_returns_bool(trained_clf):
    df = pd.read_csv("data.csv")
    record = {f: df.iloc[0][f] for f in FIELDS}
    pred = trained_clf.predict(record)
    assert isinstance(pred, (bool,))
    proba = trained_clf.predict_proba(record)
    assert 0.0 <= proba <= 1.0


def test_predict_missing_field_raises(trained_clf):
    with pytest.raises(ValueError):
        trained_clf.predict({"statement": "hello world"})


def test_modify_flips_or_reports_failure(trained_clf):
    df = pd.read_csv("data.csv")
    record = {f: df.iloc[5][f] for f in FIELDS}
    result = trained_clf.modify(record, max_word_edits=15)
    assert "success" in result and "modified_record" in result
    if result["success"]:
        new_pred = trained_clf.predict(result["modified_record"])
        assert new_pred == result["target_prediction"]


def test_predict_accepts_json_string(trained_clf):
    # spec says "submit a json object" -- check a literal JSON string works too
    import json
    df = pd.read_csv("data.csv")
    record = {f: df.iloc[0][f] for f in FIELDS}
    json_record = json.dumps(record)
    assert trained_clf.predict(json_record) == trained_clf.predict(record)


def test_save_and_load_roundtrip(trained_clf, tmp_path):
    path = tmp_path / "model.pkl"
    trained_clf.save(path)
    reloaded = TruthfulnessClassifier.load(path)
    df = pd.read_csv("data.csv")
    record = {f: df.iloc[0][f] for f in FIELDS}
    assert reloaded.predict(record) == trained_clf.predict(record)
