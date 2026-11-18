"""Bayesian HPO for the selected model (LightGBM) with Optuna TPE.

Objective: mean GroupKFold(5) point Top-1 on the filtered TRAIN trips only (Top-1 is the
map-matching quantity we care about; folds are split by trip so no trip leaks across folds).
Student-test is used only after tuning, for the tuned-vs-untuned comparison. Evaluator data: never.

usage: python scripts/tune_optuna.py [--trials 50] [--timeout 1800] [--subsample 0.3]
"""
import argparse
import json
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.experiment import MODELS_DIR, load_filtered_train, run_experiment
from pathsnap.features import FEATURES
from pathsnap.metrics import point_top1

OUT = config.RESULTS_DIR / "optuna"
FIXED = dict(subsample_freq=1, random_state=42, n_jobs=-1, verbose=-1)


def space(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--subsample", type=float, default=None,
                    help="fraction of train GPS points used inside the search (speed); final refit uses all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, points, excl = load_filtered_train(a.subsample, config.SEED)
    X = rows[FEATURES].fillna(0).astype(np.float32)
    y = rows["label"].to_numpy().astype(int)
    g = rows["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X, y, g))
    print(f"search data: {len(X):,} rows, subsample={a.subsample}, excluded={excl}", flush=True)

    def objective(trial):
        params = space(trial) | FIXED
        scores = []
        for k, (i_tr, i_va) in enumerate(folds):
            m = LGBMClassifier(**params).fit(X.iloc[i_tr], y[i_tr])
            p = m.predict_proba(X.iloc[i_va])[:, 1]
            vp = points[points["trip_id"].isin(np.unique(g[i_va]))]
            scores.append(point_top1(rows.iloc[i_va], p, vp))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_top1", scores)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=f"sqlite:///{OUT / 'study.db'}", study_name="pathsnap_lgbm", load_if_exists=True)
    study.enqueue_trial({"learning_rate": 0.05, "num_leaves": 63, "max_depth": 12, "min_child_samples": 20,
                         "reg_lambda": 1e-3, "colsample_bytree": 0.8, "subsample": 0.8, "n_estimators": 400})
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout)
    search_s = time.time() - t0

    study.trials_dataframe().to_csv(OUT / "trials.csv", index=False)
    n_done = sum(t.state.name == "COMPLETE" for t in study.trials)
    n_pruned = sum(t.state.name == "PRUNED" for t in study.trials)
    (OUT / "best_params.json").write_text(json.dumps(
        {"best_params": study.best_params, "best_cv_top1": study.best_value, "objective":
         "mean GroupKFold(5) point Top-1 on filtered train", "sampler": "TPE(seed=42)", "pruner": "Median",
         "n_trials": len(study.trials), "n_complete": n_done, "n_pruned": n_pruned,
         "search_subsample_points": a.subsample, "search_seconds": round(search_s, 1)}, indent=2))

    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    for fn, f in [(plot_optimization_history, "history.png"), (plot_param_importances, "param_importance.png")]:
        try:
            ax = fn(study); ax.figure.set_size_inches(8, 4.5); ax.figure.tight_layout()
            ax.figure.savefig(OUT / f, dpi=120); plt.close(ax.figure)
        except Exception as e:  # importance needs >1 complete trial
            print(f"skip {f}: {e}")

    best = study.best_params | FIXED
    run_experiment("lgbm_tuned", "Boosting", LGBMClassifier(**best),
                   {k: (round(v, 5) if isinstance(v, float) else v) for k, v in study.best_params.items()}
                   | {"tuned": "Optuna TPE"})

    rows_out = []
    for name in ["lgbm", "lgbm_tuned"]:
        m = json.loads((MODELS_DIR / name / "metrics.json").read_text())
        rows_out.append({"model": name, "cv_macro_f1": m["cv_macro_f1_mean"], "cv_top1": m["cv_top1_mean"],
                         "st_top1": m["st_top1"], "st_macro_f1": m["st_macro_f1"],
                         "st_log_loss": m["st_log_loss"], "infer_s_per_trip": m["infer_s_per_trip"],
                         "fit_s": m["fit_s"]})
    pd.DataFrame(rows_out).to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print(pd.DataFrame(rows_out).round(4).to_string(index=False))

    src = MODELS_DIR / "lgbm_tuned" / "model.joblib"
    if src.exists():
        config.ARTIFACTS_DIR.mkdir(exist_ok=True)
        shutil.copy(src, config.ARTIFACTS_DIR / "model.joblib")
        print("copied ->", config.ARTIFACTS_DIR / "model.joblib")


if __name__ == "__main__":
    main()
