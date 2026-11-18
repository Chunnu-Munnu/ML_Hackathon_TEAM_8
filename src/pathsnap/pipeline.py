"""End-to-end prediction for one trip: GPS track + road network -> one arc per GPS point + route polyline.

This is the only code path the API uses for predictions. It NEVER reads ``trip.route``:
callers must load trips with ``load_trip(dir, with_route=False)``.
"""
from __future__ import annotations

import time

import numpy as np

from .features import FEATURES, build_trip_features
from .io import Trip
from .network import RoadNetwork


def _compress(arcs: np.ndarray) -> np.ndarray:
    arcs = arcs[arcs >= 0]
    if len(arcs) == 0:
        return arcs
    keep = np.ones(len(arcs), dtype=bool)
    keep[1:] = arcs[1:] != arcs[:-1]
    return arcs[keep]


def predict_trip(trip: Trip, model, hmm_cfg: dict | None = None, use_hmm: bool = True) -> dict:
    if trip.route is not None:
        # Defensive: drop the reference route so nothing downstream can use it.
        trip = Trip(trip.trip_id, trip.track, trip.nodes, trip.arcs, None, trip.split)
    tm = {}
    t0 = time.perf_counter()
    net = RoadNetwork(trip)
    tm["network_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    rows, points = build_trip_features(trip, net)
    tm["features_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    if len(rows):
        rows["proba"] = model.predict_proba(rows[FEATURES].fillna(0))[:, 1]
    else:
        rows["proba"] = []
    tm["model_s"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    n = len(net.x)
    info: dict = {}
    if use_hmm and len(rows):
        from .hmm import DEFAULT_HMM_CFG, viterbi
        cfg = {**DEFAULT_HMM_CFG, **(hmm_cfg or {})}
        cands = rows[["point_idx", "arc_id", "proba"]].merge(
            build_cand_geom(rows, net), on=["point_idx", "arc_id"], how="left")
        point_arcs, info = viterbi(cands, net, cfg, emission="proba")
    else:
        point_arcs = np.full(n, -1, dtype=np.int64)
        if len(rows):
            best = rows.loc[rows.groupby("point_idx")["proba"].idxmax()]
            point_arcs[best["point_idx"].to_numpy()] = best["arc_id"].to_numpy()
    tm["hmm_s"] = time.perf_counter() - t3

    route_arcs = _compress(np.asarray(point_arcs))
    tm["total_s"] = time.perf_counter() - t0
    return {
        "trip_id": trip.trip_id,
        "n_points": int(n),
        "point_arcs": np.asarray(point_arcs, dtype=np.int64),
        "route_arcs": route_arcs,
        "route_coords": net.arc_lonlat(route_arcs) if len(route_arcs) else [],
        "gps": list(zip(trip.track["lat"].tolist(), trip.track["lon"].tolist())),
        "info": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()},
        "timings": {k: round(v, 4) for k, v in tm.items()},
        "net": net,
    }


def build_cand_geom(rows, net: RoadNetwork):
    """dist and t_proj per candidate row (needed by the HMM), recomputed from geometry."""
    from .geo import point_segment
    p = rows["point_idx"].to_numpy()
    a = rows["arc_id"].to_numpy()
    d, t, *_ = point_segment(net.x[p], net.y[p], net.ux[a], net.uy[a], net.vx[a], net.vy[a])
    import pandas as pd
    return pd.DataFrame({"point_idx": p, "arc_id": a, "dist": d, "t_proj": t})
