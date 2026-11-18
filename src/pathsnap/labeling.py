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

DIRECTION_WEIGHT_M = 0.0


def align_route(net: RoadNetwork, route: np.ndarray, chunk: int = 512,
                direction_weight: float = DIRECTION_WEIGHT_M) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (label_arc per point [canonical id], align_dist per point [m], route position per point).

    ``direction_weight`` (metres) adds w * (1 - cos(heading - arc bearing)) / 2 to the alignment cost when the
    GPS heading is reliable (moving > 3 m/s on a straight stretch). It only matters where geometry ties, i.e.
    when the route drives the same road in both directions (out-and-back): the point is then assigned to the
    pass whose direction matches the vehicle's motion. Reported align_dist stays the pure distance.
    """
    route = np.asarray(route, dtype=np.int64)
    R, N = len(route), len(net.x)
    ax, ay, bx, by = net.ux[route], net.uy[route], net.vx[route], net.vy[route]
    pen_head = None
    if direction_weight > 0:
        i = np.arange(N); lo, hi = np.clip(i - 3, 0, N - 1), np.clip(i + 3, 0, N - 1)
        dx, dy = net.x[hi] - net.x[lo], net.y[hi] - net.y[lo]
        dtw = np.maximum(net.t[hi] - net.t[lo], 1e-6)
        step = np.hypot(np.diff(net.x, prepend=net.x[0]), np.diff(net.y, prepend=net.y[0]))
        cum = np.cumsum(step); path = cum[hi] - cum[lo]
        disp = np.hypot(dx, dy)
        reliable = (disp / dtw > 3.0) & (path > 0) & (disp / np.maximum(path, 1e-9) > 0.8)
        hx, hy = np.where(reliable, dx / np.maximum(disp, 1e-9), 0.0), np.where(reliable, dy / np.maximum(disp, 1e-9), 0.0)
        L = np.maximum(np.hypot(bx - ax, by - ay), 1e-9)
        ex, ey = (bx - ax) / L, (by - ay) / L
        pen_head = (hx, hy, ex, ey, reliable)

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
        if pen_head is not None:
            hx, hy, ex, ey, rel = pen_head
            cosd = hx[s:e, None] * ex[None, :] + hy[s:e, None] * ey[None, :]
            D = D + np.where(rel[s:e, None], direction_weight * (1.0 - cosd) / 2.0, 0.0)
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
