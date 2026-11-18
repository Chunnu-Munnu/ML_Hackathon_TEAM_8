"""B0 baseline: snap every GPS point to its nearest candidate arc. No learning.

Why: the lower bound every model must beat. If a classifier can't beat "closest road wins",
its features or training are not adding anything. Scores are hard 0/1 (1 for dist_rank == 0),
so log loss is reported as N/A.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from pathsnap.experiment import run_experiment


class NearestArc(ClassifierMixin, BaseEstimator):
    no_log_loss = True

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (X["dist_rank"].to_numpy() == 0).astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


if __name__ == "__main__":
    run_experiment("b0_nearest", "Baseline (no ML)", NearestArc(),
                   {"rule": "proba=1 for nearest candidate (dist_rank==0), else 0"}, save_model=False)
