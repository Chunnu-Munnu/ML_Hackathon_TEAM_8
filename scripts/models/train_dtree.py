"""Train and score the class-balanced decision tree candidate classifier."""

from sklearn.tree import DecisionTreeClassifier

from pathsnap.config import SEED


def main():
    from pathsnap.experiment import run_experiment

    params = {"max_depth": 12, "min_samples_leaf": 50, "class_weight": "balanced"}
    estimator = DecisionTreeClassifier(**params, random_state=SEED)
    return run_experiment("dtree", "Tree", estimator, params, seed=SEED)


if __name__ == "__main__":
    main()
