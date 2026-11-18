"""Feature engineering: one row per (GPS point, candidate arc).

Design notes
* Only relative / local quantities are used: no absolute lon/lat or trip id, so models cannot memorise places.
* Direction features are essential because ~91 % of arcs have a reverse twin with identical geometry.
* Temporal-context features use neighbouring GPS points (offsets up to +-7) - they are computed from the
  GPS track only, never from labels, so they are equally available at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .candidates import get_candidates
from .geo import angle_diff, point_segment, smoothed_heading
from .io import Trip
from .network import RoadNetwork

FEATURES: list[str] = [
    # geometry
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min", "dist_z", "t_proj", "off_end_m",
    "dist_to_endpoint", "arc_len", "log_arc_len",
    # direction
    "head_diff_s", "head_diff_l", "head_cos", "head_rank", "has_reverse", "beats_reverse", "turn_rate",
    # kinematics
    "speed", "speed_smooth", "accel", "dt_prev", "dt_next", "is_gap", "is_jump", "is_stationary",
    # temporal context
    "ctx_dist_3", "ctx_dist_7", "ctx_nearest_frac_3", "prev_dist", "next_dist",
    "in_prev_cands", "in_next_cands", "conn_prev", "conn_prev_nearest",
    # network / candidate set
    "out_deg_v", "in_deg_u", "n_cands", "n_cands_10m", "score_rank",
    # heading reliability and motion along the arc
    "head_diff_vl", "head_consistency", "straightness", "along_v1", "along_v3", "along_frac3",
]

HEADING_FILL = 90.0   # neutral value when heading is undefined (stationary / duplicate points)


def point_kinematics(net: RoadNetwork, trip: Trip) -> pd.DataFrame:
    """Per-GPS-point quantities in metres / seconds."""
    x, y, t = net.x, net.y, net.t
    n = len(x)
    dt_prev = np.diff(t, prepend=np.nan)
    dt_next = np.append(np.diff(t), np.nan)
    step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = step / dt_prev
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0)
    speed_smooth = pd.Series(speed).rolling(5, center=True, min_periods=1).median().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        accel = np.diff(speed_smooth, prepend=speed_smooth[0]) / np.where(dt_prev > 0, dt_prev, 1.0)
    head_s = smoothed_heading(x, y, 2)
    head_l = smoothed_heading(x, y, 5)
    head_vl = smoothed_heading(x, y, 10)
    # straightness over i-5..i+5: net displacement / path length (1 = straight line, ~0 = GPS wandering)
    idx = np.arange(n); lo, hi = np.clip(idx - 5, 0, n - 1), np.clip(idx + 5, 0, n - 1)
    cum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(step[1:]))])
    path = cum[hi] - cum[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        straight = np.where(path > 0, np.hypot(x[hi] - x[lo], y[hi] - y[lo]) / path, 0.0)
    with np.errstate(invalid="ignore"):
        turn = angle_diff(head_s, np.roll(head_s, 1)) / np.where(dt_prev > 0, dt_prev, 1.0)
    turn[0] = 0.0
    stationary = speed_smooth < config.STATIONARY_MS
    consistency = angle_diff(np.nan_to_num(head_s), np.nan_to_num(head_l))
    consistency = np.where(np.isnan(head_s) | np.isnan(head_l), 180.0, consistency)
    head_s = np.where(stationary, np.nan, head_s)
    head_l = np.where(stationary, np.nan, head_l)
    head_vl = np.where(stationary, np.nan, head_vl)
    return pd.DataFrame({
        "point_idx": np.arange(n),
        "speed": speed, "speed_smooth": speed_smooth, "accel": np.nan_to_num(accel),
        "dt_prev": np.nan_to_num(dt_prev, nan=1.0), "dt_next": np.nan_to_num(dt_next, nan=1.0),
        "is_gap": (np.nan_to_num(dt_prev, nan=1.0) > config.GAP_S).astype(np.int8),
        "is_jump": trip.track["is_jump"].to_numpy().astype(np.int8),
        "is_stationary": stationary.astype(np.int8),
        "turn_rate": np.nan_to_num(turn),
        "head_s": head_s, "head_l": head_l, "head_vl": head_vl,
        "head_consistency": consistency, "straightness": np.clip(straight, 0, 1),
    })


def _dist_to_arc(net: RoadNetwork, pts: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    d, *_ = point_segment(net.x[pts], net.y[pts], net.ux[arcs], net.uy[arcs], net.vx[arcs], net.vy[arcs])
    return d


def build_trip_features(trip: Trip, net: RoadNetwork | None = None, r: float = config.CAND_RADIUS_M,
                        k: int = config.CAND_K, cands: pd.DataFrame | None = None,
                        labels: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rows_df, points_df).

    rows_df  : trip_id, point_idx, arc_id, *FEATURES (+ label, align_dist when the trip has a route)
    points_df: trip_id, point_idx, t, lon, lat, n_cands (+ label_arc, align_dist, label_in_cands)
    ``labels`` may pass a precomputed (label_arc, align_dist) to avoid re-aligning.
    """
    net = net or RoadNetwork(trip)
    n = len(net.x)
    c = cands if cands is not None else get_candidates(net, r=r, k=k)
    c = c.copy()
    p = c["point_idx"].to_numpy()
    a = c["arc_id"].to_numpy()
    kin = point_kinematics(net, trip)

    g = c.groupby("point_idx")["dist"]
    c["dist_rank"] = g.cumcount().to_numpy()
    dmin = g.transform("min").to_numpy()
    dmean = g.transform("mean").to_numpy()
    dstd = g.transform("std").fillna(0).to_numpy()
    d = c["dist"].to_numpy()
    c["dist_minus_min"] = d - dmin
    c["dist_ratio_min"] = d / (dmin + 1.0)
    c["dist_z"] = (d - dmean) / (dstd + 1.0)
    L = net.length[a]
    tr_ = c["t_raw"].to_numpy()
    c["off_end_m"] = np.maximum(0.0, np.maximum(-tr_, tr_ - 1.0)) * L
    tp = c["t_proj"].to_numpy()
    c["dist_to_endpoint"] = np.minimum(tp, 1.0 - tp) * L
    c["arc_len"] = L
    c["log_arc_len"] = np.log1p(L)

    # direction
    hs = kin["head_s"].to_numpy()[p]
    hl = kin["head_l"].to_numpy()[p]
    hd_s = np.where(np.isnan(hs), HEADING_FILL, angle_diff(np.nan_to_num(hs), net.bearing[a]))
    hd_l = np.where(np.isnan(hl), HEADING_FILL, angle_diff(np.nan_to_num(hl), net.bearing[a]))
    c["head_diff_s"] = hd_s
    c["head_diff_l"] = hd_l
    c["head_cos"] = np.cos(np.radians(hd_s))
    c["head_rank"] = c.groupby("point_idx")["head_diff_s"].rank(method="min").to_numpy() - 1
    c["has_reverse"] = (net.rev_of[a] >= 0).astype(np.int8)
    c["beats_reverse"] = (hd_s < 90.0).astype(np.int8)

    # kinematics (broadcast from point)
    for col in ["turn_rate", "speed", "speed_smooth", "accel", "dt_prev", "dt_next", "is_gap", "is_jump", "is_stationary"]:
        c[col] = kin[col].to_numpy()[p]

    # temporal context: distances of neighbouring points to the same arc
    offs = {}
    for o in range(-7, 8):
        q = np.clip(p + o, 0, n - 1)
        offs[o] = _dist_to_arc(net, q, a)
    c["ctx_dist_3"] = np.mean([offs[o] for o in range(-3, 4)], axis=0)
    c["ctx_dist_7"] = np.mean([offs[o] for o in range(-7, 8)], axis=0)
    c["prev_dist"] = offs[-1]
    c["next_dist"] = offs[1]

    nearest = np.full(n, -1, dtype=np.int64)
    first = c["dist_rank"].to_numpy() == 0
    nearest[p[first]] = a[first]
    frac = np.zeros(len(c))
    for o in range(-3, 4):
        q = np.clip(p + o, 0, n - 1)
        frac += (nearest[q] == a)
    c["ctx_nearest_frac_3"] = frac / 7.0

    M = np.int64(net.n_arcs + 1)
    keys = p.astype(np.int64) * M + a
    c["in_prev_cands"] = np.isin((p - 1).astype(np.int64) * M + a, keys).astype(np.int8)
    c["in_next_cands"] = np.isin((p + 1).astype(np.int64) * M + a, keys).astype(np.int8)
    NN = np.int64(net.n_nodes + 1)
    prev_end_keys = (p.astype(np.int64) + 1) * NN + net.v[a]          # "point p+1 can be reached from arc a"
    reach = np.isin(p.astype(np.int64) * NN + net.u[a], prev_end_keys)
    c["conn_prev"] = (c["in_prev_cands"].to_numpy().astype(bool) | reach).astype(np.int8)
    pn = nearest[np.clip(p - 1, 0, n - 1)]
    ok = (p > 0) & (pn >= 0)
    c["conn_prev_nearest"] = (ok & ((pn == a) | (net.v[np.maximum(pn, 0)] == net.u[a]))).astype(np.int8)

    # heading reliability + signed motion along the arc direction
    hv = kin["head_vl"].to_numpy()[p]
    c["head_diff_vl"] = np.where(np.isnan(hv), HEADING_FILL, angle_diff(np.nan_to_num(hv), net.bearing[a]))
    c["head_consistency"] = kin["head_consistency"].to_numpy()[p]
    c["straightness"] = kin["straightness"].to_numpy()[p]
    ux, uy = net.vx[a] - net.ux[a], net.vy[a] - net.uy[a]
    L_ = np.maximum(net.length[a], 1e-6); ux, uy = ux / L_, uy / L_
    for w, col in [(1, "along_v1"), (3, "along_v3")]:
        i0, i1 = np.clip(p - w, 0, n - 1), np.clip(p + w, 0, n - 1)
        dtw = np.maximum(net.t[i1] - net.t[i0], 1e-6)
        c[col] = ((net.x[i1] - net.x[i0]) * ux + (net.y[i1] - net.y[i0]) * uy) / dtw
    i0, i1 = np.clip(p - 3, 0, n - 1), np.clip(p + 3, 0, n - 1)
    disp = np.hypot(net.x[i1] - net.x[i0], net.y[i1] - net.y[i0])
    along = (net.x[i1] - net.x[i0]) * ux + (net.y[i1] - net.y[i0]) * uy
    c["along_frac3"] = np.where(disp > 0.5, along / np.maximum(disp, 1e-6), 0.0)

    # network / candidate set
    c["out_deg_v"] = net.out_deg[net.v[a]]
    c["in_deg_u"] = net.in_deg[net.u[a]]
    c["n_cands"] = c.groupby("point_idx")["arc_id"].transform("size").to_numpy()
    c["n_cands_10m"] = (c["dist"] < 10).groupby(c["point_idx"]).transform("sum").to_numpy()
    c["score"] = c["dist"] + 0.5 * c["head_diff_s"]
    c["score_rank"] = c.groupby("point_idx")["score"].rank(method="min").to_numpy() - 1

    c.insert(0, "trip_id", trip.trip_id)
    rows = c[["trip_id", "point_idx", "arc_id"] + FEATURES].copy()

    n_c = np.bincount(p, minlength=n)
    points = pd.DataFrame({"trip_id": trip.trip_id, "point_idx": np.arange(n), "t": net.t,
                           "lon": trip.track["lon"].to_numpy(), "lat": trip.track["lat"].to_numpy(),
                           "n_cands": n_c})

    if labels is None and trip.route is not None:
        from .labeling import align_route
        lab, ad, _ = align_route(net, trip.route)
        labels = (lab, ad)
    if labels is not None:
        lab, ad = labels
        rows["label"] = (a == lab[p]).astype(np.int8)
        rows["align_dist"] = ad[p]
        points["label_arc"] = lab
        points["align_dist"] = ad
        hit = np.zeros(n, dtype=bool)
        hit[p[rows["label"].to_numpy() == 1]] = True
        points["label_in_cands"] = hit

    float_cols = rows.select_dtypes("float64").columns
    rows[float_cols] = rows[float_cols].astype(np.float32)
    return rows.reset_index(drop=True), points
