"""Random Forest - bagging ensemble family.

Why: averaging many decorrelated deep trees lowers variance, which makes it robust to the noisy
reference labels in this dataset; it is the natural contrast to boosting (parallel vs sequential
trees). Limits: vote-averaged probabilities are less calibrated (matters for HMM emissions), and
the model is large and slower to predict.

Run on a 30 % sample of GPS points (all candidates of a sampled point kept) with max_samples=0.5
and min_samples_leaf=20: full-data RF x 5 CV folds does not fit in the hackathon time box and the
fully grown model would exceed the 50 MB repo limit. Recorded in config.json.
"""
from sklearn.ensemble import RandomForestClassifier

from pathsnap.experiment import run_experiment

params = dict(n_estimators=200, min_samples_leaf=20, max_features="sqrt", max_samples=0.5,
              class_weight="balanced_subsample", n_jobs=-1, random_state=42)
run_experiment("rf", "Bagging ensemble", RandomForestClassifier(**params),
               {k: v for k, v in params.items() if k != "n_jobs"} | {"criterion": "gini"},
               subsample_points=0.3)
