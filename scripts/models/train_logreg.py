"""Model 1 (M1): Logistic Regression - linear family.

Why: a linear decision boundary over well-engineered geometric/heading features is a strong, fast,
well-calibrated baseline; its probabilities feed the HMM emission directly.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.experiment import run_experiment

for C in [0.1, 1.0]:
    est = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=42))
    run_experiment(f"logreg_C{C}", "Linear", est,
                   {"C": C, "class_weight": "balanced", "max_iter": 2000, "scaler": "StandardScaler"})
