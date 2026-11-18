"""Folium maps for PathSnap GPS tracks and road geometry."""

from __future__ import annotations

import folium
import numpy as np

from .io import Trip
from .network import RoadNetwork

GPS_COLOR = "#176bc0"
START_COLOR = "#185e8a"
END_COLOR = "#302b63"
MAX_GPS_POINTS = 3000


def _gps_map(gps) -> folium.Map:
    if len(gps) == 0:
        raise ValueError("A map requires at least one GPS observation")
    points = np.asarray(gps, dtype=float)
    fmap = folium.Map(location=points[0].tolist(), tiles="OpenStreetMap", zoom_start=14)
    fmap.fit_bounds([points.min(axis=0).tolist(), points.max(axis=0).tolist()], max_zoom=17)

    observations = folium.FeatureGroup(name="GPS observations").add_to(fmap)
    indices = np.linspace(0, len(points) - 1, min(len(points), MAX_GPS_POINTS), dtype=int)
    for idx in indices:
        folium.CircleMarker(
            points[idx].tolist(), radius=2, color=GPS_COLOR, weight=1,
            fill=True, fill_opacity=0.72, tooltip=f"GPS observation {idx + 1}",
        ).add_to(observations)

    endpoints = folium.FeatureGroup(name="Start and end").add_to(fmap)
    for point, label, color in [
        (points[0], "Start", START_COLOR), (points[-1], "End", END_COLOR),
    ]:
        folium.CircleMarker(
            point.tolist(), radius=6, color="white", weight=2, fill=True,
            fill_color=color, fill_opacity=1, tooltip=label,
        ).add_to(endpoints)
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


def trip_map(
    trip: Trip,
    net: RoadNetwork | None = None,
    pred_arcs: np.ndarray | None = None,
    show_route: bool = True,
    network_buffer_m: float = 150.0,
) -> folium.Map:
    """Map the observations of one trip."""
    gps = trip.track[["lat", "lon"]].to_numpy()
    return _finish_map(_gps_map(gps), [])
