"""Geometry helpers.

Trips are spread over the whole globe, so every trip is projected to its own UTM zone
(metres, conformal, <0.1 % scale error inside a zone). Euclidean distance in degrees is not used anywhere.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


class Projector:
    def __init__(self, lon0: float, lat0: float):
        zone = int((lon0 + 180.0) // 6.0) + 1
        zone = min(max(zone, 1), 60)
        self.epsg = (32600 if lat0 >= 0 else 32700) + zone
        self._fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
        self._inv = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)

    @classmethod
    def for_track(cls, lon: np.ndarray, lat: np.ndarray) -> "Projector":
        return cls(float(np.mean(lon)), float(np.mean(lat)))

    def fwd(self, lon, lat):
        return self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))

    def inv(self, x, y):
        return self._inv.transform(np.asarray(x, float), np.asarray(y, float))


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments (element-wise, broadcastable).

    Returns dist, t_clamped in [0,1], t_raw (unclamped projection parameter), qx, qy (closest point).
    """
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    safe = np.where(L2 > 0, L2, 1.0)
    t_raw = np.where(L2 > 0, ((px - ax) * vx + (py - ay) * vy) / safe, 0.0)
    t = np.clip(t_raw, 0.0, 1.0)
    qx, qy = ax + t * vx, ay + t * vy
    return np.hypot(px - qx, py - qy), t, t_raw, qx, qy


def bearing(dx, dy):
    """Compass bearing in degrees [0, 360) of a displacement (x east, y north)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d


def smoothed_heading(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """Heading at each point from the displacement between points i-w and i+w (clipped at the ends)."""
    n = len(x)
    i = np.arange(n)
    lo, hi = np.clip(i - w, 0, n - 1), np.clip(i + w, 0, n - 1)
    dx, dy = x[hi] - x[lo], y[hi] - y[lo]
    h = bearing(dx, dy)
    h[(dx == 0) & (dy == 0)] = np.nan
    return h
