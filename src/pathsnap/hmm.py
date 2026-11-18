"""HMM / Viterbi route reconstruction.

Hidden state at step i  = one candidate road arc of GPS point i (canonical arc id).
Emission                = how well the candidate explains the GPS point:
                            "proba": lam * log(clip(classifier probability, 1e-6, 1))
                            "dist" : -0.5 * (perpendicular distance / sigma)^2     (baseline B1)
Transition (i -> j)     = -|d_route - d_gps| / beta, where d_route is the driving distance along the road
                          network between the two projected positions and d_gps is the straight-line
                          distance between the two GPS points. Unreachable or physically impossible
                          (d_route > v_max * dt + 50 m) transitions get log-prob = -inf.

Decoding is done in log space. The work is split in two stages so the decoder can be tuned cheaply:
  * ``prepare``         : network-dependent, parameter-independent work (route distances). Done once per trip.
  * ``decode_prepared`` : the Viterbi recurrence, which depends on lam / beta / sigma / v_max / gap_s.

``viterbi`` = prepare + decode_prepared. It never reads ``trip.route``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geo import point_segment
from .network import RoadNetwork

DEFAULT_HMM_CFG = {"lam": 0.3, "beta": 5.0, "sigma": 8.0, "v_max": 60.0, "gap_s": 30.0,
                   "backward_tol_m": 10.0, "route_cutoff_m": 3000.0, "min_logp": -50.0,
                   # signed_backward=1: a backward move along the same (or the preceding) arc is scored as a
                   # negative driving distance instead of a detour, so 1 Hz GPS jitter never breaks the chain
                   # while a run of points on the wrong-direction twin is penalised every step.
                   # speed_hard=0: implausible speeds are penalised through |d_route - d_gps| instead of -inf.
                   "signed_backward": 1, "speed_hard": 0}

_NEG_INF = -np.inf


def cand_geometry(rows: pd.DataFrame, net: RoadNetwork) -> pd.DataFrame:
    """Return rows with ``dist`` and ``t_proj`` recomputed from geometry (rows need point_idx, arc_id)."""
    p = rows["point_idx"].to_numpy()
    a = rows["arc_id"].to_numpy()
    d, t, *_ = point_segment(net.x[p], net.y[p], net.ux[a], net.uy[a], net.vx[a], net.vy[a])
    out = rows.copy()
    out["dist"] = d
    out["t_proj"] = t
    return out


@dataclass
class Prepared:
    """Parameter-independent decoding inputs for one trip."""
    n_points: int
    pts: np.ndarray            # point_idx of every point that has candidates (sorted)
    off: np.ndarray            # candidate offsets, len(pts) + 1
    arc: np.ndarray            # flat candidate arc ids
    dist: np.ndarray           # flat candidate distances (m)
    proba: np.ndarray | None   # flat classifier probabilities (or None)
    dt: np.ndarray             # dt[k] = t[pts[k]] - t[pts[k-1]]  (dt[0] = 0)
    d_gps: np.ndarray          # straight-line distance between consecutive decoded points
    d_route: list              # d_route[k]: (K_{k-1}, K_k) matrix of network distances, None if not prepared
    t: np.ndarray              # timestamps of all GPS points
    max_gap_s: float           # transitions with dt above this were not prepared


def prepare(cands: pd.DataFrame, net: RoadNetwork, cfg: dict | None = None,
            max_gap_s: float | None = None) -> Prepared:
    """Compute network distances between candidates of consecutive points (the slow part)."""
    cfg = {**DEFAULT_HMM_CFG, **(cfg or {})}
    max_gap_s = cfg["gap_s"] if max_gap_s is None else max_gap_s
    n = len(net.x)
    empty = Prepared(n, np.array([], dtype=np.int64), np.array([0]), np.array([], dtype=np.int64),
                     np.array([]), None, np.array([]), np.array([]), [], net.t, max_gap_s)
    if cands is None or len(cands) == 0:
        return empty

    c = cands.sort_values(["point_idx"], kind="mergesort").reset_index(drop=True)
    pidx = c["point_idx"].to_numpy(dtype=np.int64)
    arc = c["arc_id"].to_numpy(dtype=np.int64)
    dist = c["dist"].to_numpy(dtype=float)
    tproj = c["t_proj"].to_numpy(dtype=float)
    proba = c["proba"].to_numpy(dtype=float) if "proba" in c.columns else None

    pts, start = np.unique(pidx, return_index=True)
    off = np.append(start, len(c))
    m = len(pts)

    t = net.t
    dt = np.zeros(m)
    d_gps = np.zeros(m)
    dt[1:] = t[pts[1:]] - t[pts[:-1]]
    d_gps[1:] = np.hypot(net.x[pts[1:]] - net.x[pts[:-1]], net.y[pts[1:]] - net.y[pts[:-1]])

    cutoff = cfg["route_cutoff_m"]
    tol = cfg["backward_tol_m"]
    signed = bool(cfg.get("signed_backward", 1))
    length, u, v = net.length, net.u, net.v
    d_route: list = [None] * m
    for k in range(1, m):
        if dt[k] > max_gap_s:
            continue  # will be a gap split; no transition needed
        i0, i1 = off[k - 1], off[k]
        j0, j1 = off[k], off[k + 1]
        A, ta = arc[i0:i1], tproj[i0:i1]
        B, tb = arc[j0:j1], tproj[j0:j1]
        ub = u[B]
        lb = length[B]
        M = np.full((len(A), len(B)), np.inf)
        for ai in range(len(A)):
            a = A[ai]
            nodes, d = net.node_dists(int(v[a]), cutoff)
            idx = np.searchsorted(nodes, ub)
            ok = idx < len(nodes)
            idx_c = np.where(ok, idx, 0)
            ok &= nodes[idx_c] == ub
            mid = np.where(ok, d[idx_c], np.inf)
            row = (1.0 - ta[ai]) * length[a] + mid + tb * lb
            same = B == a
            if signed:
                # GPS jitter at 1 Hz often projects a point slightly *behind* the previous one. Instead of
                # forcing a detour around the network (which breaks the chain), score the backward move as
                # a negative driving distance, so |d_route - d_gps| penalises it smoothly: cheap for a
                # stationary wobble, expensive for a run of points on the wrong-direction twin.
                back_same = same & (tb < ta[ai])
                back = -(ta[ai] - tb) * length[a]
                row = np.where(back_same, back, row)
                prev_arc = v[B] == u[a]                      # b directly precedes a: stepped back over a node
                back_node = -(ta[ai] * length[a] + (1.0 - tb) * lb)
                use = prev_arc & ~same & (np.abs(back_node) < np.abs(row))
                row = np.where(use, back_node, row)
            if same.any():
                fwd = same & (tb >= ta[ai])
                row = np.where(fwd, (tb - ta[ai]) * length[a], row)
                if not signed:
                    jitter = same & (tb < ta[ai]) & ((ta[ai] - tb) * length[a] <= tol)
                    row = np.where(jitter, 0.0, row)
            M[ai] = row
        d_route[k] = M

    return Prepared(n, pts, off, arc, dist, proba, dt, d_gps, d_route, t, max_gap_s)


def _log_emission(prep: Prepared, cfg: dict, emission: str) -> np.ndarray:
    if emission == "proba":
        if prep.proba is None:
            raise ValueError("emission='proba' needs a 'proba' column in cands")
        return cfg["lam"] * np.log(np.clip(prep.proba, 1e-6, 1.0))
    if emission == "dist":
        return -0.5 * (prep.dist / cfg["sigma"]) ** 2
    raise ValueError(f"unknown emission {emission!r}")


def decode_prepared(prep: Prepared, cfg: dict | None = None, emission: str = "proba") -> tuple[np.ndarray, dict]:
    """Viterbi over a Prepared trip. Returns (arcs per GPS point, info)."""
    cfg = {**DEFAULT_HMM_CFG, **(cfg or {})}
    if cfg["gap_s"] > prep.max_gap_s + 1e-9:
        raise ValueError("cfg['gap_s'] is larger than the max_gap_s used in prepare()")
    n = prep.n_points
    arcs = np.full(n, -1, dtype=np.int64)
    info = {"n_breaks": 0, "n_no_cand": 0, "n_gap_splits": 0, "no_cand_mask": np.ones(n, dtype=bool)}
    m = len(prep.pts)
    if m == 0:
        info["n_no_cand"] = n
        return arcs, info

    logE = _log_emission(prep, cfg, emission)
    off, arc = prep.off, prep.arc
    beta, v_max, min_logp, gap_s = cfg["beta"], cfg["v_max"], cfg["min_logp"], cfg["gap_s"]

    chosen = np.empty(m, dtype=np.int64)       # chosen flat candidate index per step
    bps: list = [None] * m                     # backpointers (local index into previous step), None = chain start
    chain_start = 0
    delta = logE[off[0]:off[1]].copy()

    def backtrack(start: int, end: int, last_delta: np.ndarray) -> None:
        s = int(np.argmax(last_delta))
        for k in range(end, start - 1, -1):
            chosen[k] = off[k] + s
            if k > start:
                s = int(bps[k][s])

    for k in range(1, m):
        e = logE[off[k]:off[k + 1]]
        if prep.dt[k] > gap_s:
            backtrack(chain_start, k - 1, delta)
            info["n_gap_splits"] += 1
            chain_start, delta = k, e.copy()
            continue
        dR = prep.d_route[k]
        with np.errstate(invalid="ignore"):
            logT = -np.abs(dR - prep.d_gps[k]) / beta
            logT = np.maximum(logT, min_logp)
            feasible = np.isfinite(dR)
            if bool(cfg.get("speed_hard", 0)):
                feasible &= dR <= v_max * prep.dt[k] + 50.0
            logT = np.where(feasible, logT, _NEG_INF)
            scores = delta[:, None] + logT
        best = scores.max(axis=0)
        if not np.isfinite(best).any():
            backtrack(chain_start, k - 1, delta)
            info["n_breaks"] += 1
            chain_start, delta = k, e.copy()
            continue
        bps[k] = scores.argmax(axis=0)
        delta = best + e
    backtrack(chain_start, m - 1, delta)

    decoded_pts = prep.pts
    arcs[decoded_pts] = arc[chosen]
    mask = np.ones(n, dtype=bool)
    mask[decoded_pts] = False
    info["no_cand_mask"] = mask
    info["n_no_cand"] = int(mask.sum())

    # Points without candidates copy the nearest decoded point in time (ties -> previous).
    if mask.any():
        miss = np.flatnonzero(mask)
        tt = prep.t
        pos = np.searchsorted(tt[decoded_pts], tt[miss])           # first decoded with t >= t_miss
        prv = np.clip(pos - 1, 0, m - 1)
        nxt = np.clip(pos, 0, m - 1)
        use_prev = np.abs(tt[miss] - tt[decoded_pts[prv]]) <= np.abs(tt[decoded_pts[nxt]] - tt[miss])
        arcs[miss] = arcs[decoded_pts[np.where(use_prev, prv, nxt)]]
    return arcs, info


def viterbi(cands: pd.DataFrame, net: RoadNetwork, cfg: dict = DEFAULT_HMM_CFG,
            emission: str = "proba") -> tuple[np.ndarray, dict]:
    """Decode one arc per GPS point.

    cands columns: point_idx, arc_id, dist, t_proj, proba (proba only needed for emission='proba').
    Returns (arcs, info) with info = {n_breaks, n_no_cand, n_gap_splits, no_cand_mask}.
    """
    cfg = {**DEFAULT_HMM_CFG, **(cfg or {})}
    prep = prepare(cands, net, cfg)
    return decode_prepared(prep, cfg, emission)
