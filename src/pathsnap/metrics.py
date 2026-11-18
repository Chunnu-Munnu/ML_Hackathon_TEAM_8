"""Candidate-level and point-level metrics shared by every model.

Row level  = one (GPS point, candidate arc) pair, binary label "is this the reference arc?".
Point level = one GPS point; the model's choice is the candidate with the highest probability.
Top-1 is the metric that matters for map-matching: each point has at most one correct candidate,
so threshold-based F1 mostly measures calibration, while Top-1 measures the actual arc choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

EPS = 1e-7


def _top_rows(rows: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """The highest-probability candidate row of every (trip_id, point_idx). Ties -> first row."""
    df = rows[["trip_id", "point_idx", "label"]].copy()
    df["proba"] = np.asarray(proba, dtype=float)
    df["_ord"] = np.arange(len(df))
    df = df.sort_values(["trip_id", "point_idx", "proba", "_ord"],
                        ascending=[True, True, False, True], kind="mergesort")
    return df.drop_duplicates(["trip_id", "point_idx"], keep="first")


def point_top1(rows: pd.DataFrame, proba: np.ndarray, points: pd.DataFrame) -> float:
    """Share of ALL GPS points in `points` whose top-scored candidate is the reference arc.
    Points without candidates count as wrong."""
    if len(points) == 0:
        return float("nan")
    top = _top_rows(rows, proba)
    return float((top["label"] == 1).sum() / len(points))


def point_top1_covered(rows: pd.DataFrame, proba: np.ndarray) -> float:
    """Same as point_top1, but only over points that have at least one candidate."""
    top = _top_rows(rows, proba)
    return float((top["label"] == 1).mean()) if len(top) else float("nan")


def candidate_miss_rate(points: pd.DataFrame) -> float:
    """Share of points whose reference arc is not among their candidates."""
    return float(np.mean(~points["label_in_cands"].astype(bool)))


def row_metrics(y_true, proba, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(proba, dtype=float), EPS, 1 - EPS)
    y_pred = (p >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
    }


def confusion(y_true, proba, thr: float = 0.5) -> np.ndarray:
    y_pred = (np.asarray(proba, dtype=float) >= thr).astype(int)
    return confusion_matrix(np.asarray(y_true).astype(int), y_pred, labels=[0, 1])
