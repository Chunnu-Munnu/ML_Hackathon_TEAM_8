"""Train and score the distance-weighted KNN candidate classifier."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.config import SEED


def main():
    from pathsnap.experiment import run_experiment

    params = {"n_neighbors": 25, "weights": "distance", "n_jobs": -1}
    estimator = make_pipeline(StandardScaler(), KNeighborsClassifier(**params))
    return run_experiment(
        "knn", "Instance-based", estimator,
        {**params, "scaler": "StandardScaler"},
        subsample_points=0.2, seed=SEED,
    )


if __name__ == "__main__":
    main()
