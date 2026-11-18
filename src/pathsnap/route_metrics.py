"""Route-level metrics for decoded trips."""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork


def continuity(arcs: np.ndarray, net: RoadNetwork) -> float:
    """Share of consecutive point pairs (both with a predicted arc) whose arcs are equal or directly connected."""
    arcs = np.asarray(arcs, dtype=np.int64)
    a, b = arcs[:-1], arcs[1:]
    valid = (a >= 0) & (b >= 0)
    if not valid.any():
        return float("nan")
    a, b = a[valid], b[valid]
    ok = (a == b) | (net.v[a] == net.u[b])
    return float(ok.mean())


def point_accuracy(pred_arcs: np.ndarray, label_arcs: np.ndarray) -> float:
    """Mean(pred == label). Points without a label (-1) or without a prediction count as wrong."""
    pred = np.asarray(pred_arcs)
    lab = np.asarray(label_arcs)
    return float(((pred == lab) & (lab >= 0)).mean()) if len(lab) else float("nan")


def route_prf(pred_arcs: np.ndarray, route_canon: np.ndarray) -> dict:
    """Precision / recall / F1 of the SET of predicted arcs against the set of reference-route arcs.

    ``route_canon`` must already be canonical ids, i.e. ``net.canon[trip.route]``.
    """
    pred = set(int(a) for a in np.asarray(pred_arcs) if a >= 0)
    ref = set(int(a) for a in np.asarray(route_canon))
    tp = len(pred & ref)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(ref) if ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}


def compress(arcs: np.ndarray) -> np.ndarray:
    """Remove consecutive duplicates (and unassigned -1 points): the predicted route as an arc sequence."""
    arcs = np.asarray(arcs, dtype=np.int64)
    arcs = arcs[arcs >= 0]
    if len(arcs) == 0:
        return arcs
    keep = np.ones(len(arcs), dtype=bool)
    keep[1:] = arcs[1:] != arcs[:-1]
    return arcs[keep]
