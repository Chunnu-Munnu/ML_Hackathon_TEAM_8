"""Collect everything the API needs at inference time into artifacts/.

Run after model selection / tuning (artifacts/model.joblib from tune_optuna.py) and decoder tuning
(artifacts/hmm_config.json from tune_decoder.py). Writes:
  artifacts/feature_list.json   ordered feature columns the model expects
  artifacts/metadata.json       versions, seed, candidate settings, projection rule, model scores
and checks that the model loads and predicts on one student-test trip end to end.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata as md
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import config  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.pipeline import predict_trip  # noqa: E402

PKGS = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "optuna",
        "shapely", "pyproj", "fastapi", "streamlit", "folium"]


def _ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None


def main():
    art = config.ARTIFACTS_DIR
    art.mkdir(exist_ok=True)
    (art / "feature_list.json").write_text(json.dumps(FEATURES, indent=2))

    model_path = art / "model.joblib"
    hmm_path = art / "hmm_config.json"
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.SEED,
        "python": platform.python_version(),
        "packages": {p: _ver(p) for p in PKGS},
        "candidates": {"radius_m": config.CAND_RADIUS_M, "k": config.CAND_K},
        "projection": "per-trip UTM zone from mean track longitude (EPSG 326xx north / 327xx south)",
        "n_features": len(FEATURES),
        "model_file": model_path.name if model_path.exists() else None,
        "hmm_config_file": hmm_path.name if hmm_path.exists() else None,
    }
    tuned = config.RESULTS_DIR / "optuna" / "best_params.json"
    if tuned.exists():
        meta["best_params"] = json.loads(tuned.read_text())
    excl = config.FEATURES_DIR / "excluded_train_trips.json"
    if excl.exists():
        meta["excluded_train_trips"] = json.loads(excl.read_text())

    if model_path.exists():
        model = joblib.load(model_path)
        meta["model_class"] = type(model).__name__
        hmm_cfg = json.loads(hmm_path.read_text()) if hmm_path.exists() else None
        trip_dir = list_trips(config.STUDENT_TEST_DIR)[0]
        try:
            out = predict_trip(load_trip(trip_dir, with_route=False), model, hmm_cfg, use_hmm=hmm_cfg is not None)
            meta["smoke_test"] = {"trip_id": trip_dir.name, "n_points": out["n_points"], "timings": out["timings"]}
            print("smoke test ok:", meta["smoke_test"])
        except ImportError as e:   # hmm.py not merged yet
            meta["smoke_test"] = f"skipped: {e}"
            print("smoke test skipped:", e)
    else:
        print("artifacts/model.joblib not found yet - run tune_optuna.py first; writing metadata anyway")

    (art / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print("wrote", art / "feature_list.json", art / "metadata.json")


if __name__ == "__main__":
    main()
