"""Streamlit client for the PathSnap reconstruction API."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import requests
import streamlit as st


class ApiError(RuntimeError):
    """An API error that can be shown directly in the dashboard."""


def api_json(api_url: str, endpoint: str, *, body: dict | None = None) -> dict:
    method = "POST" if body is not None else "GET"
    try:
        response = requests.request(
            method, f"{api_url}{endpoint}", json=body,
            timeout=(4, 180 if method == "POST" else 8),
        )
    except requests.Timeout as exc:
        raise ApiError("API request timed out.") from exc
    except requests.RequestException as exc:
        raise ApiError("Could not connect to the PathSnap API.") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiError(f"The API returned an unreadable response (HTTP {response.status_code}).") from exc
    if not isinstance(payload, dict):
        raise ApiError("The API response has an unexpected format.")
    if not response.ok:
        detail = payload.get("detail")
        raise ApiError(detail if isinstance(detail, str) else f"API request failed (HTTP {response.status_code}).")
    return payload


def show_health(api_url: str) -> bool:
    try:
        url = urlsplit(api_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.query or url.fragment:
            raise ValueError
        health = api_json(api_url, "/health")
    except ValueError:
        st.sidebar.error("Enter a valid HTTP or HTTPS API URL.")
        st.stop()
    except ApiError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    if health.get("status") == "ok":
        model = health.get("model")
        st.sidebar.success(f"API ready · {model}" if model else "API ready")
        return True
    st.sidebar.warning("API connected · reconstruction unavailable")
    for issue in health.get("issues", []):
        st.sidebar.caption(str(issue))
    return False


def main() -> None:
    st.set_page_config(page_title="PathSnap | Route reconstruction", layout="wide")
    st.title("PathSnap")
    st.caption("Map matching and route reconstruction")
    st.sidebar.header("API connection")
    api_url = st.sidebar.text_input(
        "API URL", value=os.environ.get("PATHSNAP_API", "http://localhost:8000"),
    ).strip().rstrip("/")
    ready = show_health(api_url)
    try:
        trips = api_json(api_url, "/trips").get("trips", [])
    except ApiError as exc:
        st.error(str(exc))
        return
    if not trips:
        st.info("The API has no available trips.")
        return

    trip_id = st.selectbox("Trip", trips, key=f"trip:{api_url}")
    if st.button("Reconstruct route", type="primary", disabled=not ready):
        st.session_state.pop("prediction", None)
        with st.spinner(f"Reconstructing trip {trip_id}..."):
            try:
                prediction = api_json(api_url, "/predict", body={"trip_id": trip_id})
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.session_state["prediction"] = (api_url, trip_id, prediction)

    saved = st.session_state.get("prediction")
    if saved and saved[:2] == (api_url, trip_id):
        st.success(f"Trip {trip_id} reconstructed")
    elif not ready:
        st.info("Reconstruction will be available when the API model and decoder are ready.")


if __name__ == "__main__":
    main()
