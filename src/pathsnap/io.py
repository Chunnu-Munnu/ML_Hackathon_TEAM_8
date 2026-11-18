"""Reading raw PathSnap trip files.

Raw format per trip folder ``<id>/``:
  <id>.track  lon lat t        (whitespace separated, CRLF line endings, t in seconds)
  <id>.nodes  lon lat          (row index = node id)
  <id>.arcs   from_node to_node (row index = arc id)
  <id>.route  arc_id           (chronological ground-truth arcs; no timestamps)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame          # lon, lat, t (+ cleaning flags)
    nodes: pd.DataFrame          # lon, lat
    arcs: pd.DataFrame           # u, v
    route: np.ndarray | None     # arc ids, or None when loaded for prediction
    split: str = ""

    @property
    def n_points(self) -> int:
        return len(self.track)


def _read(path: Path, names: list[str], dtype=float) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, engine="c")
    return df.astype(dtype)


def add_track_flags(track: pd.DataFrame) -> pd.DataFrame:
    """Flag (never drop) suspicious observations: every GPS point still needs a predicted arc."""
    tr = track.copy()
    lon, lat, t = tr["lon"].to_numpy(), tr["lat"].to_numpy(), tr["t"].to_numpy()
    dt = np.diff(t, prepend=np.nan)
    # Approximate metres for flagging only (precise projection happens in geo.py).
    k = 111_320.0
    dx = np.diff(lon, prepend=np.nan) * k * np.cos(np.radians(lat))
    dy = np.diff(lat, prepend=np.nan) * k
    step = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = step / dt
    tr["dt_prev"] = dt
    tr["is_dup"] = (step == 0)
    tr["is_jump"] = speed > config.JUMP_SPEED_MS
    tr["is_gap"] = dt > config.GAP_S
    return tr


def load_trip(trip_dir: str | Path, with_route: bool = True) -> Trip:
    """Load one trip folder. Use ``with_route=False`` for anything on the prediction path."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", ["lon", "lat", "t"])
    track = add_track_flags(track)
    nodes = _read(trip_dir / f"{tid}.nodes", ["lon", "lat"])
    arcs = _read(trip_dir / f"{tid}.arcs", ["u", "v"], dtype=np.int64)
    route = None
    if with_route and (trip_dir / f"{tid}.route").exists():
        route = _read(trip_dir / f"{tid}.route", ["arc"], dtype=np.int64)["arc"].to_numpy()
    return Trip(tid, track, nodes, arcs, route, split=trip_dir.parent.name)


def load_route(trip_dir: str | Path) -> np.ndarray:
    """Reference route only (used server-side for scoring, never by the predictor)."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    return _read(trip_dir / f"{tid}.route", ["arc"], dtype=np.int64)["arc"].to_numpy()


def list_trips(split_dir: str | Path) -> list[Path]:
    split_dir = Path(split_dir)
    return sorted(p for p in split_dir.iterdir() if p.is_dir())
