"""Exercise running PathSnap services using the ten supplied student-test trips."""

import argparse
import csv
import os
from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest


STUDENT_TRIPS = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091",
]
RESPONSE_FIELDS = {"trip_id", "n_points", "gps", "pred_route", "point_arcs", "metrics", "timings"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--ui", default="http://127.0.0.1:8501")
    parser.add_argument("--output", type=Path, help="Write per-trip timings and aggregate metrics to CSV")
    args = parser.parse_args()
    api_url = args.api.rstrip("/")

    with requests.Session() as http:
        health = http.get(f"{api_url}/health", timeout=10)
        health.raise_for_status()
        details = health.json()
        eval_dir = details["eval_dir"].replace("\\", "/").rstrip("/")
        assert Path(eval_dir).name == "student_test", "Point the API at the supplied student-test partition"
        assert details["status"] == "ok", details
        trips = http.get(f"{api_url}/trips", timeout=10)
        trips.raise_for_status()
        assert trips.json()["trips"] == STUDENT_TRIPS
        ui_health = http.get(f"{args.ui.rstrip('/')}/_stcore/health", timeout=10)
        ui_health.raise_for_status()

        observations = []
        for trip_id in STUDENT_TRIPS:
            response = http.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=180)
            response.raise_for_status()
            payload = response.json()
            assert set(payload) == RESPONSE_FIELDS, "The response schema must contain only public fields"
            assert payload["trip_id"] == trip_id and payload["pred_route"]
            assert len(payload["point_arcs"]) == payload["n_points"] > 0
            assert len(payload["gps"]) == min(payload["n_points"], 3000)
            assert payload["timings"]["hmm_s"] > 0
            assert all(0 <= payload["metrics"][key] <= 1 for key in ("top1_accuracy", "continuity", "route_f1"))
            observed = {
                "split": "student_test", "trip_id": trip_id, "n_points": payload["n_points"],
                "gps_sent": len(payload["gps"]), "predicted_segments": len(payload["pred_route"]),
                **payload["metrics"], **payload["timings"],
            }
            observations.append(observed)
            print(f"{trip_id}  Top-1 {observed['top1_accuracy']:.3%}  "
                  f"continuity {observed['continuity']:.3%}  HMM {observed['hmm_s']:.3f}s", flush=True)

        for trip_id, status in [("unknown", 404), ("../train", 422)]:
            response = http.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=10)
            assert response.status_code == status

    app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"
    with patch.dict(os.environ, {"PATHSNAP_API": api_url}):
        app = AppTest.from_file(app_path, default_timeout=180).run()
        assert not app.exception and not app.error
        assert app.selectbox[0].options == STUDENT_TRIPS
        app.selectbox[0].select(STUDENT_TRIPS[1]).run()
        app.button[0].click().run()
        assert not app.exception and not app.error
        assert [metric.label for metric in app.metric] == [
            "Top-1 accuracy", "Route continuity", "GPS observations", "Inference latency",
        ]
        assert app.metric[0].value == f"{observations[1]['top1_accuracy']:.1%}"
        assert app.metric[1].value == f"{observations[1]['continuity']:.1%}"
        assert app.expander[0].label == "Detailed timings"
        assert app.get("component_instance"), "Expected the Folium map component"

    if args.output is not None:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=observations[0].keys())
            writer.writeheader()
            writer.writerows(observations)
    print("FastAPI, all 10 HMM predictions, and the Streamlit dashboard passed.", flush=True)


if __name__ == "__main__":
    main()
