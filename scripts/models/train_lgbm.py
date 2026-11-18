"""LightGBM - gradient boosting family.

Why: features mix very different scales (metres, degrees, counts, 0/1 flags); trees need no scaling.
Boosting learns interactions such as "close AND heading matches AND neighbours agree" that a linear
model cannot, trains fast on ~1.5M candidate rows, optimises log loss (binary cross-entropy) so its
probabilities can be used directly as HMM emissions, and predicts fast enough for the API.
"""
from lightgbm import LGBMClassifier

from pathsnap.experiment import run_experiment

params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
run_experiment("lgbm", "Boosting", LGBMClassifier(**params),
               {k: v for k, v in params.items() if k not in ("n_jobs", "verbose")} | {"objective": "binary (log loss)"})
