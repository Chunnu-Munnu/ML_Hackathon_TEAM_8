"""MLP - neural-network family.

Why: a small multilayer perceptron can learn non-linear interactions between the geometric, heading and
temporal-context features (e.g. 'close AND heading-consistent AND connected to the previous candidate') that a
linear model cannot, and it outputs probabilities for the HMM emission. sklearn's MLPClassifier trains with log-loss
(cross-entropy) using Adam, so it produces a real epoch-wise loss_curve_ (the harness plots it).

Trained on a 20% sample of GPS points (all candidates of a sampled point kept) to fit the time box,
like KNN / RF / SVM. Two architectures are compared; every run is kept and must be reported.
Note: early_stopping holds out a random 10% of *rows* inside each training fold (sklearn cannot group by trip),
so its validation score is slightly optimistic; the harness' GroupKFold-by-trip CV is the honest estimate.
"""
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.experiment import run_experiment

for name, hidden in [("mlp_64_32", (64, 32)), ("mlp_128_64_32", (128, 64, 32))]:
    est = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=1e-4,
                      learning_rate_init=1e-3, batch_size=256, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=5, max_iter=60, random_state=42),
    )
    run_experiment(name, "Neural network", est,
                   {"hidden_layer_sizes": list(hidden), "activation": "relu", "solver": "adam", "alpha": 1e-4,
                    "learning_rate_init": 1e-3, "batch_size": 256, "early_stopping": True,
                    "max_iter": 60, "n_iter_no_change": 5, "scaler": "StandardScaler"},
                   subsample_points=0.2)
