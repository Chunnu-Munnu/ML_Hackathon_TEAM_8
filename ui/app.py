"""Streamlit evaluator UI for the PathSnap reconstruction API (Laptop 2).

The UI never reads data files: the trip list, reconstruction, server-side scores and the results summary all
come from the API over HTTP.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

from pathsnap.viz import reconstruction_map, view_for

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}
h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0;}
.ps-sub {color: #6b7685; margin: 0.15rem 0 1.1rem 0; font-size: 0.98rem;}
div[data-testid="stMetric"] {background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.18);
  border-radius: 10px; padding: 0.7rem 0.9rem;}
div[data-testid="stMetricLabel"] p {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em;}
.ps-pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem;
  background: rgba(31,95,191,0.12); color: #1f5fbf; margin-right: 6px;}
</style>
"""


class ApiError(RuntimeError):
    """An API error that can be shown directly in the dashboard."""


def api_json(api_url: str, endpoint: str, *, body: dict | None = None) -> dict:
    method = "POST" if body is not None else "GET"
    try:
        response = requests.request(
            method, f"{api_url}{endpoint}", json=body,
            timeout=(4, 180 if method == "POST" else 15),
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
        folder = str(health.get("eval_dir", "")).replace("\\", "/").rstrip("/").split("/")[-1]
        st.sidebar.caption(f"Trip set on the server: **{folder}**")
        return True
    st.sidebar.warning("API connected · reconstruction unavailable")
    for issue in health.get("issues", []):
        st.sidebar.caption(str(issue))
    return False


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


def show_prediction(result: dict) -> None:
    metrics, timings = result["metrics"], result["timings"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Top-1 accuracy", _percent(metrics["top1_accuracy"]),
              help="Share of GPS observations assigned to the aligned reference arc (computed on the server).")
    c2.metric("Route continuity", _percent(metrics["continuity"]),
              help="Share of consecutive observations on the same or directly connected road segments.")
    c3.metric("Route F1", _percent(metrics.get("route_f1")),
              help="Overlap between the set of predicted road segments and the reference route's segments.")
    c4.metric("GPS observations", f"{result['n_points']:,}")
    c5.metric("Inference time", f"{1000 * timings['inference_s']:,.0f} ms",
              help="Road network, features, classifier and HMM decoding (server-side scoring excluded).")

    if result["gps"]:
        st.markdown(f"**Trip {result['trip_id']}** · model `{result.get('model', 'deployed')}` · "
                    f"{len(result['pred_route']):,} predicted road segments · blue = GPS, red = predicted route, grey = road network")
        fmap = reconstruction_map(result["gps"], result["pred_route"], result.get("network"))
        center, zoom = view_for(result["gps"])
        st_folium(fmap, height=620, use_container_width=True, returned_objects=[], center=center, zoom=zoom,
                  key=f"route-map:{result['trip_id']}:{result.get('model')}")
        if len(result["gps"]) < result["n_points"]:
            st.caption(f"Map shows {len(result['gps']):,} of {result['n_points']:,} GPS observations.")
        if not result["pred_route"]:
            st.info("The decoder returned no road segments for this trip.")
    else:
        st.info("The API returned no GPS coordinates for this trip.")

    stage_labels = {
        "load_s": "Read trip files", "network_s": "Build the road network",
        "features_s": "Generate candidate features", "model_s": "Score candidates (classifier)",
        "hmm_s": "Decode the route (HMM / Viterbi)", "evaluation_s": "Score against the reference (server)",
        "inference_s": "Inference subtotal", "total_s": "Server processing total",
    }
    with st.expander("Stage timings"):
        st.dataframe(pd.DataFrame([
            {"Stage": stage_labels.get(key, key), "Duration (ms)": round(1000 * timings[key], 1)}
            for key in stage_labels if key in timings
        ]), hide_index=True, use_container_width=True)


def model_picker(api_url: str) -> str | None:
    """Left-panel model selector with the chosen model's recorded scores."""
    try:
        cat = api_json(api_url, "/models")
    except ApiError:
        return None
    models = cat.get("models", [])
    if not models:
        return None
    deployed = cat.get("deployed")
    names = ["deployed"] + [m["name"] for m in models]
    labels = {"deployed": f"Deployed model ({deployed})" if deployed else "Deployed model"}
    st.sidebar.header("Model")
    choice = st.sidebar.selectbox("Classifier used for reconstruction", names,
                                  format_func=lambda n: labels.get(n, n), key="model_choice")
    info = next((m for m in models if m["name"] == (deployed if choice == "deployed" else choice)), None)
    if info:
        st.sidebar.caption(f"Family: **{info.get('family', '—')}**")
        ll, inf = info.get("st_log_loss"), info.get("infer_s_per_trip")
        rows = [("CV Top-1 (train)", _percent(info.get("cv_top1_mean"))),
                ("CV macro F1", f"{info['cv_macro_f1_mean']:.3f}" if info.get("cv_macro_f1_mean") is not None else "—"),
                ("Student-test Top-1", _percent(info.get("st_top1"))),
                ("Student-test + HMM", _percent(info.get("st_top1_hmm"))),
                ("Log loss", f"{ll:.3f}" if ll is not None else "N/A"),
                ("Inference / trip", f"{1000 * inf:.0f} ms" if inf is not None else "—")]
        table = ["| Metric | Value |", "|---|---|"] + [f"| {k} | **{v}** |" for k, v in rows]
        st.sidebar.markdown(chr(10).join(table))
        if info.get("subsample_points"):
            st.sidebar.caption(f"Trained on a {int(100 * info['subsample_points'])}% sample of GPS points.")
    return choice


def reconstruct_tab(api_url: str, ready: bool, model: str | None = None) -> None:
    try:
        trips = api_json(api_url, "/trips").get("trips", [])
    except ApiError as exc:
        st.error(str(exc))
        return
    if not trips:
        st.info("The API has no available trips.")
        return

    left, right = st.columns([3, 1])
    trip_id = left.selectbox(f"Evaluation trip ({len(trips)} available)", trips, key=f"trip:{api_url}")
    right.write("")
    right.write("")
    go = right.button("Reconstruct route", type="primary", disabled=not ready, use_container_width=True)
    if go:
        st.session_state.pop("prediction", None)
        with st.spinner(f"Reconstructing trip {trip_id} on the server..."):
            try:
                body = {"trip_id": trip_id}
                if model:
                    body["model"] = model
                prediction = api_json(api_url, "/predict", body=body)
            except ApiError as exc:
                st.error(str(exc))
            else:
                st.session_state["prediction"] = (api_url, trip_id, prediction)

    saved = st.session_state.get("prediction")
    if saved and saved[:2] == (api_url, trip_id) and saved[2].get("model", "deployed") == (model or "deployed"):
        show_prediction(saved[2])
    elif not ready:
        st.info("Reconstruction will be available when the API model and decoder are ready.")
    else:
        st.caption("Select a trip and press **Reconstruct route**. The reference route never leaves the server.")


def results_tab(api_url: str) -> None:
    try:
        s = api_json(api_url, "/metrics")
    except ApiError as exc:
        st.info(f"Results summary unavailable: {exc}")
        return
    ev, stf = s.get("evaluator"), s.get("student_test_final") or {}
    models = s.get("models", [])
    best = models[0] if models else {}
    c = st.columns(4)
    c[0].metric("Evaluator Top-1 (pooled)", _percent(ev and ev.get("pooled_top1")))
    c[1].metric("Evaluator continuity", _percent(ev and ev.get("pooled_continuity")))
    c[2].metric("Student-test Top-1 (pooled)", _percent(stf.get("pooled_top1")))
    c[3].metric("Best CV Top-1 (train)", _percent(best.get("cv_top1_mean")))

    if ev:
        st.subheader("Final evaluation · 10 evaluator trips")
        df = pd.DataFrame(ev["trips"])[["trip_id", "n_points", "top1_accuracy", "continuity", "route_f1", "n_breaks", "total_s"]]
        st.dataframe(df.rename(columns={"trip_id": "Trip", "n_points": "GPS points", "top1_accuracy": "Top-1",
                                        "continuity": "Continuity", "route_f1": "Route F1", "n_breaks": "HMM breaks",
                                        "total_s": "Latency (s)"}),
                     hide_index=True, use_container_width=True,
                     column_config={k: st.column_config.ProgressColumn(k, format="%.3f", min_value=0, max_value=1)
                                    for k in ["Top-1", "Continuity", "Route F1"]})

    st.subheader("Model comparison")
    rows = [{"Model": m["label"], "Family": m.get("family"), "CV Top-1": m.get("cv_top1_mean"),
             "CV macro F1": m.get("cv_macro_f1_mean"), "CV log loss": m.get("cv_log_loss_mean"),
             "ST Top-1 (classifier)": m.get("st_top1"), "ST Top-1 (+HMM)": m.get("st_top1_with_hmm"),
             "Fit (s)": m.get("fit_s"), "Infer (s/trip)": m.get("infer_s_per_trip")} for m in models]
    mdf = pd.DataFrame(rows)
    st.dataframe(mdf, hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.NumberColumn(k, format="%.4f") for k in
                                ["CV Top-1", "CV macro F1", "CV log loss", "ST Top-1 (classifier)", "ST Top-1 (+HMM)"]})
    chart = mdf.dropna(subset=["CV Top-1"]).set_index("Model")[["CV Top-1"]].sort_values("CV Top-1")
    st.bar_chart(chart, horizontal=True, height=360)

    if s.get("tuned_vs_untuned"):
        st.subheader("Bayesian optimisation (Optuna)")
        st.dataframe(pd.DataFrame(s["tuned_vs_untuned"]), hide_index=True, use_container_width=True)
    if s.get("hmm_variants_student_test"):
        st.subheader("Classifier only vs classifier + HMM (student-test)")
        h = pd.DataFrame(s["hmm_variants_student_test"])[["model", "variant", "top1", "continuity", "route_f1", "n_breaks"]]
        st.dataframe(h, hide_index=True, use_container_width=True)
    st.caption(f"Summary generated {s.get('generated_utc')} from the experiment result files.")


def main() -> None:
    st.set_page_config(page_title="PathSnap | Route reconstruction", page_icon="🧭", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("PathSnap")
    st.markdown("<p class='ps-sub'>Map-matching noisy GPS to the road network · "
                "<span class='ps-pill'>Candidate classifier</span><span class='ps-pill'>HMM / Viterbi</span>"
                "<span class='ps-pill'>FastAPI</span></p>", unsafe_allow_html=True)
    st.sidebar.header("API connection")
    api_url = st.sidebar.text_input(
        "API URL", value=os.environ.get("PATHSNAP_API", "http://localhost:8000"),
        help="Address of the FastAPI server on Laptop 1, e.g. http://192.168.1.20:8000",
    ).strip().rstrip("/")
    ready = show_health(api_url)
    model = model_picker(api_url)

    tab_run, tab_res = st.tabs(["Reconstruct route", "Results"])
    with tab_run:
        reconstruct_tab(api_url, ready, model)
    with tab_res:
        results_tab(api_url)


if __name__ == "__main__":
    main()
