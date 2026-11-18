"""Experiment harness: every model goes through run_experiment so all results are comparable.

Protocol (identical for all models):
  1. train rows minus excluded trips (unreliable reference labels); student-test kept whole
  2. optional subsample of whole GPS points (all candidate rows of a sampled point kept)
  3. GroupKFold(5) by trip_id on train -> CV row metrics + point Top-1 + fit time
  4. refit on all filtered train rows -> student-test metrics, confusion matrix, inference time,
     and train-set scores for the over/underfitting check
  5. write results/models/<name>/ and one row of results/models/model_comparison.csv
Student-test data is never used for fitting; evaluator data is never touched here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold

from . import config
from .features import FEATURES
from .metrics import (candidate_miss_rate, confusion, point_top1, point_top1_covered,
                      row_metrics)

MODELS_DIR = config.RESULTS_DIR / "models"
COMPARISON_CSV = MODELS_DIR / "model_comparison.csv"
COMPARISON_COLS = ["name", "family", "key_config", "cv_macro_f1_mean", "cv_macro_f1_std",
                   "cv_top1_mean", "st_top1", "st_macro_f1", "st_log_loss", "st_miss_rate",
                   "fit_s", "infer_s_per_trip", "route_continuity"]
MAX_MODEL_MB = 50
N_FOLDS = 5


# ----------------------------------------------------------------------------- data
def excluded_trips() -> list[str]:
    p = config.FEATURES_DIR / "excluded_train_trips.json"
    return json.loads(p.read_text())["trips"] if p.exists() else []


def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """("train" | "student_test") -> (candidate rows, points)."""
    if split not in ("train", "student_test"):
        raise ValueError("load_split only serves 'train' and 'student_test'")
    rows = pd.read_parquet(config.FEATURES_DIR / f"{split}.parquet")
    points = pd.read_parquet(config.FEATURES_DIR / f"points_{split}.parquet")
    return rows, points


def load_filtered_train(subsample_points: float | None = None, seed: int = config.SEED):
    """Train rows/points with excluded trips dropped and optional point-level subsampling."""
    rows, points = load_split("train")
    excl = excluded_trips()
    rows = rows[~rows["trip_id"].isin(excl)]
    points = points[~points["trip_id"].isin(excl)]
    if subsample_points is not None and subsample_points < 1.0:
        keys = points[["trip_id", "point_idx"]].sample(frac=subsample_points, random_state=seed)
        points = points.merge(keys, on=["trip_id", "point_idx"])
        rows = rows.merge(keys, on=["trip_id", "point_idx"])
    return rows.reset_index(drop=True), points.reset_index(drop=True), excl


def _X(rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return rows[cols].fillna(0).astype(np.float32)


# ----------------------------------------------------------------------------- helpers
def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def _final_step(est):
    return est.steps[-1][1] if hasattr(est, "steps") else est


def _proba(est, X) -> np.ndarray:
    return est.predict_proba(X)[:, 1]


def _save_confusion(cm: np.ndarray, name: str, out: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["not ref arc (0)", "ref arc (1)"]).plot(
        ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"{name}: student-test confusion (thr 0.5)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_loss_curve(est, name: str, out: Path) -> bool:
    last = _final_step(est)
    if not hasattr(last, "loss_curve_"):
        return False
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(last.loss_curve_, label="training loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    if getattr(last, "validation_scores_", None) is not None:
        ax2 = ax1.twinx()
        ax2.plot(last.validation_scores_, color="tab:orange", label="validation score")
        ax2.set_ylabel("validation score (accuracy)")
        ax2.legend(loc="center right")
    ax1.legend(loc="upper right")
    ax1.set_title(f"{name}: training history")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def update_comparison(row: dict):
    """Append or replace (keyed by name) one row of model_comparison.csv; keeps route_continuity."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if COMPARISON_CSV.exists():
        df = pd.read_csv(COMPARISON_CSV, dtype={"name": str})
        old = df[df["name"] == row["name"]]
        if len(old) and pd.notna(old.iloc[0].get("route_continuity")) and row.get("route_continuity") is None:
            row["route_continuity"] = old.iloc[0]["route_continuity"]
        df = df[df["name"] != row["name"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.reindex(columns=COMPARISON_COLS)
    df.to_csv(COMPARISON_CSV, index=False)


# ----------------------------------------------------------------------------- main entry
def run_experiment(name: str, family: str, estimator, config: dict,
                   feature_cols: list[str] = FEATURES, subsample_points: float | None = None,
                   seed: int = 42, save_model: bool = True) -> dict:
    out = MODELS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    no_logloss = bool(getattr(estimator, "no_log_loss", False))   # e.g. the B0 0/1 baseline

    # 1-2. data
    tr_rows, tr_points, excl = load_filtered_train(subsample_points, seed)
    st_rows, st_points = load_split("student_test")
    X_tr, y_tr = _X(tr_rows, feature_cols), tr_rows["label"].to_numpy().astype(int)
    X_st, y_st = _X(st_rows, feature_cols), st_rows["label"].to_numpy().astype(int)
    groups = tr_rows["trip_id"].to_numpy()
    print(f"[{name}] train rows={len(X_tr):,} (pos={y_tr.mean():.3f}), "
          f"student-test rows={len(X_st):,}, excluded={excl}", flush=True)

    # 3. group-aware CV
    folds = []
    for k, (i_tr, i_va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X_tr, y_tr, groups)):
        est = clone(estimator)
        t0 = time.perf_counter()
        est.fit(X_tr.iloc[i_tr], y_tr[i_tr])
        fit_s = time.perf_counter() - t0
        p = _proba(est, X_tr.iloc[i_va])
        va_rows = tr_rows.iloc[i_va]
        va_points = tr_points[tr_points["trip_id"].isin(np.unique(groups[i_va]))]
        m = row_metrics(y_tr[i_va], p)
        folds.append({"fold": k, "n_train_rows": len(i_tr), "n_val_rows": len(i_va), **m,
                      "top1": point_top1(va_rows, p, va_points), "fit_s": fit_s})
        print(f"[{name}] fold {k}: macro_f1={m['macro_f1']:.4f} top1={folds[-1]['top1']:.4f} "
              f"fit={fit_s:.1f}s", flush=True)
    cv = pd.DataFrame(folds)[["fold", "n_train_rows", "n_val_rows", "macro_f1", "f1", "precision",
                              "recall", "accuracy", "log_loss", "top1", "fit_s"]]
    if no_logloss:
        cv["log_loss"] = np.nan
    cv.to_csv(out / "cv_folds.csv", index=False)

    # 4. refit on all filtered train, score student-test
    est = clone(estimator)
    t0 = time.perf_counter()
    est.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    p_train = _proba(est, X_tr)
    train_m = row_metrics(y_tr, p_train)
    train_top1 = point_top1(tr_rows, p_train, tr_points)

    n_st_trips = st_rows["trip_id"].nunique()
    t0 = time.perf_counter()
    p_st = _proba(est, X_st)
    infer_s = (time.perf_counter() - t0) / max(n_st_trips, 1)

    st_m = row_metrics(y_st, p_st)
    cm = confusion(y_st, p_st)
    _save_confusion(cm, name, out / "confusion_matrix.png")
    has_curve = _save_loss_curve(est, name, out / "loss_curve.png")

    metrics = {
        "cv_macro_f1_mean": float(cv["macro_f1"].mean()), "cv_macro_f1_std": float(cv["macro_f1"].std()),
        "cv_top1_mean": float(cv["top1"].mean()), "cv_top1_std": float(cv["top1"].std()),
        "cv_log_loss_mean": None if no_logloss else float(cv["log_loss"].mean()),
        "train_macro_f1": train_m["macro_f1"], "train_top1": train_top1,
        "st_accuracy": st_m["accuracy"], "st_precision": st_m["precision"], "st_recall": st_m["recall"],
        "st_f1": st_m["f1"], "st_macro_f1": st_m["macro_f1"],
        "st_log_loss": None if no_logloss else st_m["log_loss"],
        "st_top1": point_top1(st_rows, p_st, st_points),
        "st_top1_covered": point_top1_covered(st_rows, p_st),
        "st_miss_rate": candidate_miss_rate(st_points),
        "fit_s": fit_s, "infer_s_per_trip": infer_s,
        "st_confusion": cm.tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pd.DataFrame({"trip_id": st_rows["trip_id"].to_numpy(), "point_idx": st_rows["point_idx"].to_numpy(),
                  "arc_id": st_rows["arc_id"].to_numpy(), "proba": p_st}
                 ).to_parquet(out / "student_test_proba.parquet", index=False)

    # 5. model file + config
    model_note = "not saved (save_model=False)"
    if save_model:
        mp = out / "model.joblib"
        joblib.dump(est, mp, compress=3)
        mb = mp.stat().st_size / 1e6
        if mb > MAX_MODEL_MB:
            mp.unlink()
            model_note = f"not committed: {mb:.0f} MB > {MAX_MODEL_MB} MB; rerun this script to regenerate"
        else:
            model_note = f"model.joblib ({mb:.1f} MB)"
    cfg = {"name": name, "family": family, "config": _jsonable(config),
           "estimator_params": _jsonable(estimator.get_params()),
           "feature_cols": list(feature_cols), "subsample_points": subsample_points, "seed": seed,
           "excluded_train_trips": excl, "cv": f"GroupKFold(n_splits={N_FOLDS}) by trip_id",
           "model_file": model_note, "loss_curve": has_curve,
           "log_loss_note": "N/A: hard 0/1 scores, log loss undefined/meaningless" if no_logloss else None}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    update_comparison({
        "name": name, "family": family,
        "key_config": "; ".join(f"{k}={v}" for k, v in config.items()),
        "cv_macro_f1_mean": metrics["cv_macro_f1_mean"], "cv_macro_f1_std": metrics["cv_macro_f1_std"],
        "cv_top1_mean": metrics["cv_top1_mean"], "st_top1": metrics["st_top1"],
        "st_macro_f1": metrics["st_macro_f1"], "st_log_loss": metrics["st_log_loss"],
        "st_miss_rate": metrics["st_miss_rate"], "fit_s": fit_s, "infer_s_per_trip": infer_s,
        "route_continuity": None,
    })

    print(f"[{name}] CV macroF1={metrics['cv_macro_f1_mean']:.4f}±{metrics['cv_macro_f1_std']:.4f} "
          f"CV top1={metrics['cv_top1_mean']:.4f} | train top1={train_top1:.4f} | "
          f"ST top1={metrics['st_top1']:.4f} ST macroF1={metrics['st_macro_f1']:.4f} "
          f"ST logloss={metrics['st_log_loss']} | fit={fit_s:.1f}s infer={infer_s:.3f}s/trip", flush=True)
    return metrics
