# truthguard

## Description
Binary truthfulness classifier for political/public statements. 
LightGBM (gradient-boosted trees) over TF-IDF text (word
+ character n-grams) plus pretrained embeddings, multi-label subjects,
speaker/context categoricals, and a speaker truthfulness history feature.
Plus a minimal-edit "modification" search (word substitution +
removal, restricted to `statement`/`statement_context`/`subjects`) that
finds the smallest change to a statement that flips its predicted label.

## Project structure
```
truthguard/
  pyproject.toml
  README.md
  LICENSE
  src/
    truthguard/
      __init__.py
      config.py          # label policy + constants
      features.py         # feature engineering pipeline
      model.py             # TruthfulnessClassifier (train/predict/modify/diagnostics)
      modification.py      # minimal-edit search
      logging_utils.py      # optional enable_logging() flow tracer
  scripts/
    tune_hyperparameters.py  # reproducible hyperparameter search
    tuning_results.md         # full results table from that search
  tests/
    test_pipeline.py          # test functions that were used during development and testing
```

## Instructions to run the package
1) Open a terminal/command prompt

2) Run: "cd .\truthguard" -- According to your directory.

3) Run: "python -m venv venv" -- To create a virtual environment.

4) Run: "venv\Scripts\activate" -- To activate the virtual environment.

5) Run: "pip install -e ." -- Downloads spaCy model + NLTK WordNet, ~35MB

6) Secure "data.csv" exists in ".\truthguard" folder. This is already included in the folder.

7) Secure the "record.json" exists in ".\truthguard" folder. This is already included in the folder.
 
8) You can alter the values in "record.json" fields, to test different statements.

9) Run "python demo.py"
This will run the truthguard Python package (Train, Test, Modify).
In the first run of "demo.py", the Train process will run and a "model.pkl" file will be saved in ".\truthguard" folder.
If "model.pkl" file exists, the Train process is skipped for time efficiency reasons.
To run again the Train process, the "model.pkl" needs to be deleted before running the "demo.py".

10) Check ".\truthguard\truthguard.log" for output. This file will be created only after you run the demo.py (step 9).
"truthguard.log" includes the textual output of every run. 
The same output can be seen in the terminal console, after you run "python demo.py".