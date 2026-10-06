import os

from truthguard import TruthfulnessClassifier, enable_logging, logger

# Turn on detailed logging
enable_logging("truthguard.log", also_console=True)
logger.info(f"Log will be written to: {os.path.abspath('truthguard.log')}")

# 1. Train (Function 1) -- only if we don't already have a saved model.
#    Delete model.pkl any time you want to force a fresh training run
#    (e.g. after changing data.csv, or after editing the package code).
if os.path.exists("model.pkl"):
    logger.info("Found existing model.pkl -- loading it instead of retraining.\n")
    clf = TruthfulnessClassifier.load("model.pkl")
else:
    logger.info("No model.pkl found -- training a new model (this takes ~1 minute with cross validation deactivated).\n")
    clf = TruthfulnessClassifier()
    metrics = clf.train("data.csv")
    logger.info(f"Validation metrics: {metrics}")
    clf.save("model.pkl")

# 2. Predict (Function 2) -- reading the record from an actual JSON file
with open("record.json", "r") as f:
    record_json = f.read()

prediction = clf.predict(record_json)  # passed as a raw JSON string, read straight from the file
logger.info(f"Prediction: {prediction}\n")

# 3. Modify (Function 3) -- same JSON file, same pattern
result = clf.modify(record_json)
logger.info(f"Modified record: {result['modified_record']}\n")
logger.info(f"Edit summary: {result['note']}\n")
