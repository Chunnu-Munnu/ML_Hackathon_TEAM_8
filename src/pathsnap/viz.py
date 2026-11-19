"""Folium maps for PathSnap GPS tracks and road geometry."""

from __future__ import annotations

import folium
import numpy as np

from .geo import point_segment
from .io import Trip
from .network import RoadNetwork

GPS_COLOR = "#176bc0"
NETWORK_COLOR = "#7c8793"
REFERENCE_COLOR = "#20794e"
PREDICTED_COLOR = "#c42d39"
MAX_NETWORK_ARCS = 5000
START_COLOR = "#185e8a"
END_COLOR = "#302b63"
MAX_GPS_POINTS = 3000


def view_for(gps) -> tuple[list[float], int]:
    """Centre and zoom that frame the whole track (used explicitly so the map never opens zoomed out)."""
    pts = np.asarray(gps, dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span_deg = max(hi[0] - lo[0], (hi[1] - lo[1]) * np.cos(np.radians((hi[0] + lo[0]) / 2)), 1e-4)
    zoom = int(np.clip(np.floor(np.log2(360.0 / span_deg)) - 1, 3, 17))
    return [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)], zoom


def _gps_map(gps) -> folium.Map:
    if len(gps) == 0:
        raise ValueError("A map requires at least one GPS observation")
    points = np.asarray(gps, dtype=float)
    center, zoom = view_for(points)
    fmap = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=zoom)
    fmap.fit_bounds([points.min(axis=0).tolist(), points.max(axis=0).tolist()], max_zoom=17)

    observations = folium.FeatureGroup(name="GPS observations").add_to(fmap)
    indices = np.linspace(0, len(points) - 1, min(len(points), MAX_GPS_POINTS), dtype=int)
    for idx in indices:
        folium.CircleMarker(
            points[idx].tolist(), radius=2, color=GPS_COLOR, weight=0,
            fill=True, fill_opacity=0.45, tooltip=f"GPS observation {idx + 1}",
        ).add_to(observations)

    endpoints = folium.FeatureGroup(name="Start and end").add_to(fmap)
    for point, label, color in [
        (points[0], "Start", START_COLOR), (points[-1], "End", END_COLOR),
    ]:
        folium.CircleMarker(
            point.tolist(), radius=6, color="white", weight=2, fill=True,
            fill_color=color, fill_opacity=1, tooltip=label,
        ).add_to(endpoints)
    fmap.keep_in_front(observations, endpoints)
    return fmap


def _finish_map(fmap: folium.Map, layers: list[tuple[str, str]]) -> folium.Map:
    items = "".join(
        f'<div><span style="color:{color};font-size:16px">&#9679;</span> {label}</div>'
        for label, color in [("GPS observations", GPS_COLOR), ("Start", START_COLOR),
                             ("End", END_COLOR), *layers]
    )
    fmap.get_root().html.add_child(folium.Element(
        '<aside aria-label="Map legend" style="position:fixed;bottom:26px;left:12px;'
        'z-index:1000;background:#fff;color:#182637;border:1px solid #bbc9d4;'
        'border-radius:6px;padding:8px 12px;font:13px system-ui">'
        f'<strong>Map legend</strong>{items}</aside>'
    ))
    folium.LayerControl().add_to(fmap)
    return fmap


def _nearby_arcs(net: RoadNetwork, radius_m: float) -> np.ndarray:
    if not np.isfinite(radius_m) or radius_m < 0:
        raise ValueError("network_buffer_m must be finite and nonnegative")
    closest = np.full(net.n_arcs, np.inf)
    for start in range(0, len(net.x), 1024):
        x, y = net.x[start:start + 1024], net.y[start:start + 1024]
        points, arcs = net.query_radius(x, y, radius_m)
        dist, *_ = point_segment(
            x[points], y[points], net.ux[arcs], net.uy[arcs], net.vx[arcs], net.vy[arcs],
        )
        np.minimum.at(closest, arcs, dist)
    ids = np.flatnonzero(np.isfinite(closest))
    order = np.lexsort((ids, closest[ids]))
    return ids[order[:MAX_NETWORK_ARCS]]


def _line_layer(fmap, name, segments, *, color, weight, opacity=1):
    layer = folium.FeatureGroup(name=name).add_to(fmap)
    if len(segments):
        # A Leaflet multi-polyline keeps individual arcs separate.
        folium.PolyLine(segments, color=color, weight=weight, opacity=opacity).add_to(layer)
    return layer


def trip_map(
    trip: Trip,
    net: RoadNetwork | None = None,
    pred_arcs: np.ndarray | None = None,
    show_route: bool = True,
    network_buffer_m: float = 150.0,
) -> folium.Map:
    """Map GPS, nearby roads, and optional reference and predicted directed arcs."""
    gps = trip.track[["lat", "lon"]].to_numpy()
    fmap = _gps_map(gps)
    net = net if net is not None else RoadNetwork(trip)
    nearby = _nearby_arcs(net, network_buffer_m)
    name = "Road network (nearby)"
    _line_layer(fmap, name, net.arc_lonlat(nearby), color=NETWORK_COLOR, weight=1, opacity=0.6)
    layers = [(name, NETWORK_COLOR)]

    if show_route and trip.route is not None:
        name = "Reference route"
        ref_layer = _line_layer(fmap, name, net.arc_lonlat(net.canon[trip.route]), color=REFERENCE_COLOR, weight=6, opacity=0.8)
        layers.append((name, REFERENCE_COLOR))
        fmap.keep_in_front(ref_layer)
    if pred_arcs is not None:
        predicted = np.asarray(pred_arcs, dtype=np.int64)
        if predicted.ndim != 1 or ((predicted < -1) | (predicted >= net.n_arcs)).any():
            raise ValueError("pred_arcs must be a sequence of valid arc IDs or -1")
        predicted = predicted[predicted >= 0]
        if len(predicted):
            predicted = predicted[np.r_[True, predicted[1:] != predicted[:-1]]]
        name = "Predicted route"
        pred_layer = _line_layer(fmap, name, net.arc_lonlat(predicted), color=PREDICTED_COLOR, weight=3)
        layers.append((name, PREDICTED_COLOR))
        fmap.keep_in_front(pred_layer)

    return _finish_map(fmap, layers)


def reconstruction_map(gps, predicted_segments, network_segments=None) -> folium.Map:
    """Map the GPS observations, nearby road network and predicted route returned by the API."""
    center, zoom = view_for(gps)
    fmap = _gps_map(gps)
    fmap.location, fmap.options["zoom"] = center, zoom
    layers = []
    if network_segments:
        name = "Road network (nearby)"
        _line_layer(fmap, name, network_segments, color=NETWORK_COLOR, weight=1.5, opacity=0.55)
        layers.append((name, NETWORK_COLOR))
    name = "Predicted route"
    route = _line_layer(fmap, name, predicted_segments, color=PREDICTED_COLOR, weight=5, opacity=0.95)
    layers.append((name, PREDICTED_COLOR))
    fmap.keep_in_front(route)
    return _finish_map(fmap, layers)
