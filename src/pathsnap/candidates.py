"""Candidate road segments for every GPS point: all candidate arcs within r metres, keep the k nearest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .geo import point_segment
from .network import RoadNetwork


def get_candidates(net: RoadNetwork, r: float = config.CAND_RADIUS_M, k: int = config.CAND_K,
                   x: np.ndarray | None = None, y: np.ndarray | None = None) -> pd.DataFrame:
    """Return DataFrame[point_idx, arc_id, dist, t_proj, t_raw, qx, qy] sorted by point then distance.

    Points with no arc within r simply have no rows (the HMM handles them).
    """
    x = net.x if x is None else x
    y = net.y if y is None else y
    pi, ai = net.query_radius(x, y, r)
    if len(pi) == 0:
        return pd.DataFrame(columns=["point_idx", "arc_id", "dist", "t_proj", "t_raw", "qx", "qy"])
    d, t, t_raw, qx, qy = point_segment(x[pi], y[pi], net.ux[ai], net.uy[ai], net.vx[ai], net.vy[ai])
    df = pd.DataFrame({"point_idx": pi, "arc_id": ai, "dist": d, "t_proj": t, "t_raw": t_raw, "qx": qx, "qy": qy})
    df = df[df["dist"] <= r]
    df = df.sort_values(["point_idx", "dist"], kind="mergesort")
    df = df[df.groupby("point_idx").cumcount() < k]
    return df.reset_index(drop=True)
