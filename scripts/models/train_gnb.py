"""Gaussian Naive Bayes - probabilistic family.

Why: a generative baseline that models each feature as class-conditionally Gaussian and independent.
Our features are strongly correlated (dist / ctx_dist / dist_rank), so this tests how much the
independence assumption hurts ranking and calibration compared with discriminative models.
"""
from sklearn.naive_bayes import GaussianNB

from pathsnap.experiment import run_experiment

for vs in [1e-9, 1e-6]:
    run_experiment(f"gnb_vs{vs:g}", "Probabilistic", GaussianNB(var_smoothing=vs),
                   {"var_smoothing": vs, "priors": "empirical"})
