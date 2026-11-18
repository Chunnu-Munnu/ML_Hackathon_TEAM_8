"""Road network for one trip: projected geometry, spatial index, directed graph, shortest paths."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .geo import Projector, bearing
from .io import Trip


class RoadNetwork:
    """Everything the candidate generator, feature builder and HMM need about one trip's roads.

    Arc handling decisions (from EDA):
      * duplicate arcs (same u->v) are collapsed to a canonical id = smallest arc id with that (u, v);
      * self-loops and zero-length arcs are never candidates (no direction, no geometry);
      * ``rev_of[a]`` is the canonical id of the opposite-direction twin (v->u) or -1.
    """

    def __init__(self, trip: Trip, proj: Projector | None = None):
        self.trip_id = trip.trip_id
        tr = trip.track
        self.proj = proj or Projector.for_track(tr["lon"].to_numpy(), tr["lat"].to_numpy())

        # Track in metres
        self.x, self.y = self.proj.fwd(tr["lon"].to_numpy(), tr["lat"].to_numpy())
        self.t = tr["t"].to_numpy(dtype=float)

        # Nodes / arcs in metres
        self.nx, self.ny = self.proj.fwd(trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy())
        self.u = trip.arcs["u"].to_numpy()
        self.v = trip.arcs["v"].to_numpy()
        self.n_arcs = len(self.u)
        self.n_nodes = len(self.nx)
        self.ux, self.uy = self.nx[self.u], self.ny[self.u]
        self.vx, self.vy = self.nx[self.v], self.ny[self.v]
        self.length = np.hypot(self.vx - self.ux, self.vy - self.uy)
        self.bearing = bearing(self.vx - self.ux, self.vy - self.uy)

        # Canonical ids for duplicate arcs
        uv = pd.DataFrame({"u": self.u, "v": self.v, "a": np.arange(self.n_arcs)})
        first = uv.groupby(["u", "v"])["a"].transform("min").to_numpy()
        self.canon = first
        self.is_canon = first == np.arange(self.n_arcs)

        # Reverse twin
        key = {(int(a), int(b)): int(i) for a, b, i in zip(self.u[self.is_canon], self.v[self.is_canon],
                                                            np.flatnonzero(self.is_canon))}
        self.rev_of = np.array([key.get((int(b), int(a)), -1) for a, b in zip(self.u, self.v)], dtype=np.int64)

        # Degrees (on canonical arcs)
        cu, cv = self.u[self.is_canon], self.v[self.is_canon]
        self.out_deg = np.bincount(cu, minlength=self.n_nodes)
        self.in_deg = np.bincount(cv, minlength=self.n_nodes)

        # Candidate arcs: canonical, not a self-loop, non-zero length
        self.cand_mask = self.is_canon & (self.u != self.v) & (self.length > 0.01)
        self.cand_ids = np.flatnonzero(self.cand_mask)
        coords = np.stack([np.stack([self.ux[self.cand_ids], self.uy[self.cand_ids]], 1),
                           np.stack([self.vx[self.cand_ids], self.vy[self.cand_ids]], 1)], 1)
        self.tree = shapely.STRtree(shapely.linestrings(coords))

        # Directed graph u -> v weighted by length (zero-length arcs kept for connectivity)
        g = self.is_canon & (self.u != self.v)
        w = np.maximum(self.length[g], 1e-3)
        self.graph = csr_matrix((w, (self.u[g], self.v[g])), shape=(self.n_nodes, self.n_nodes))
        self._sp_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}

    # ---------------------------------------------------------------- spatial queries
    def query_radius(self, x: np.ndarray, y: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
        """(point_idx, arc_id) pairs whose bounding geometry lies within r metres."""
        pts = shapely.points(np.asarray(x), np.asarray(y))
        pi, ti = self.tree.query(pts, predicate="dwithin", distance=r)
        return pi, self.cand_ids[ti]

    # ---------------------------------------------------------------- shortest paths
    def node_dists(self, src: int, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
        """Nodes reachable from ``src`` within ``cutoff`` metres: (sorted node ids, distances)."""
        key = (int(src), float(cutoff))
        hit = self._sp_cache.get(key)
        if hit is None:
            d = dijkstra(self.graph, directed=True, indices=int(src), limit=cutoff)
            nodes = np.flatnonzero(np.isfinite(d))
            hit = (nodes, d[nodes])
            self._sp_cache[key] = hit
        return hit

    def node_to_node(self, src: int, dst: int, cutoff: float) -> float:
        if src == dst:
            return 0.0
        nodes, d = self.node_dists(src, cutoff)
        j = np.searchsorted(nodes, dst)
        return float(d[j]) if j < len(nodes) and nodes[j] == dst else np.inf

    def route_dist(self, a: int, ta: float, b: int, tb: float, cutoff: float = 2000.0) -> float:
        """Driving distance (m) from position ``ta`` (fraction) on arc a to ``tb`` on arc b; inf if unreachable."""
        if a == b and tb >= ta:
            return (tb - ta) * self.length[a]
        rem = (1.0 - ta) * self.length[a]
        mid = self.node_to_node(int(self.v[a]), int(self.u[b]), cutoff)
        return rem + mid + tb * self.length[b]

    def connected(self, a: int, b: int) -> bool:
        """True when b equals a or directly follows a (v(a) == u(b))."""
        return a == b or self.v[a] == self.u[b]

    def arc_lonlat(self, arc_ids: np.ndarray) -> list[list[tuple[float, float]]]:
        """[(lat, lon), (lat, lon)] per arc, for maps."""
        lon_u, lat_u = self.proj.inv(self.ux[arc_ids], self.uy[arc_ids])
        lon_v, lat_v = self.proj.inv(self.vx[arc_ids], self.vy[arc_ids])
        return [[(a, b), (c, d)] for a, b, c, d in zip(lat_u, lon_u, lat_v, lon_v)]
