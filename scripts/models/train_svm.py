"""linear SVM with probability calibration - kernel/margin family.

Why: a max-margin linear classifier (hinge-type loss) is a strong, fast margin-based baseline on scaled features.
A raw SVM outputs decision scores, not probabilities, so it is wrapped in CalibratedClassifierCV (sigmoid / Platt
scaling) to produce the probabilities the HMM emission needs. An RBF kernel SVM is O(n^2)-O(n^3) and infeasible on
millions of (point, candidate) rows, which is why LinearSVC is used and training is subsampled by point groups.

Note: the inner cv=3 of CalibratedClassifierCV splits rows, not trips (sklearn cannot pass groups through a
Pipeline); the outer GroupKFold-by-trip CV in the harness remains leakage-free for model selection.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pathsnap.experiment import run_experiment

for C in [0.1, 1.0]:
    est = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(LinearSVC(C=C, dual=False, max_iter=5000, random_state=42),
                               method="sigmoid", cv=3),
    )
    run_experiment(f"svm_linear_C{C}", "Kernel / margin", est,
                   {"C": C, "kernel": "linear (LinearSVC, squared hinge)", "calibration": "sigmoid, cv=3",
                    "dual": False, "scaler": "StandardScaler"},
                   subsample_points=0.3)
