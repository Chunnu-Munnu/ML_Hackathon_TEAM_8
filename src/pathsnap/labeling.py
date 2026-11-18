"""Point-level labels from the reference route.

The .route file is an ordered list of arcs with no timestamps, so a GPS row does not map to a route row.
We align them with a monotone dynamic programme: point i gets route position k_i with
k_0 <= k_1 <= ... <= k_{N-1}, minimising the total point-to-arc distance

    C[i, k] = D[i, k] + min_{k' <= k} C[i-1, k']

Monotonicity follows the driving order, so loops / roads driven twice are labelled correctly
(nearest-arc labelling would mix the two passes).
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment
from .network import RoadNetwork


def align_route(net: RoadNetwork, route: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (label_arc per point [canonical id], align_dist per point [m], route position per point)."""
    route = np.asarray(route, dtype=np.int64)
    R, N = len(route), len(net.x)
    ax, ay, bx, by = net.ux[route], net.uy[route], net.vx[route], net.vy[route]

    idx_dtype = np.int16 if R < np.iinfo(np.int16).max else np.int32
    back = np.empty((N, R), dtype=idx_dtype)      # argmin of prefix of previous cost row
    prev = None
    ar = np.arange(R)
    dists_rows = np.empty((N, R), dtype=np.float32) if N * R <= 60_000_000 else None

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        D, *_ = point_segment(net.x[s:e, None], net.y[s:e, None], ax, ay, bx, by)
        if dists_rows is not None:
            dists_rows[s:e] = D
        for j in range(e - s):
            i = s + j
            if prev is None:
                cur = D[j].astype(np.float64)
                back[i] = ar
            else:
                # running minimum of prev and where it occurred
                pm = np.minimum.accumulate(prev)
                is_new = np.empty(R, dtype=bool)
                is_new[0] = True
                is_new[1:] = prev[1:] < pm[:-1]
                arg = np.maximum.accumulate(np.where(is_new, ar, 0))
                back[i] = arg
                cur = D[j] + pm
            prev = cur

    # backtrack
    pos = np.empty(N, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(N - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    if dists_rows is not None:
        align_d = dists_rows[np.arange(N), pos].astype(float)
    else:
        align_d, *_ = point_segment(net.x, net.y, ax[pos], ay[pos], bx[pos], by[pos])
    labels = net.canon[route[pos]]
    return labels, align_d, pos
