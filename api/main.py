"""FastAPI service for PathSnap route reconstruction."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from time import perf_counter
from types import ModuleType

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pathsnap import config
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import predict_trip

TRIP_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=TRIP_ID_PATTERN, min_length=1, max_length=64)


@dataclass
class Runtime:
    model: object | None = None
    hmm_cfg: dict | None = None
    route_metrics: ModuleType | None = None
    issues: list[str] = field(default_factory=list)


def _configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.Logger("pathsnap.api", level=logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.exception("Could not open the API log file")
    return logger


def _read_model(path: Path):
    model = joblib.load(path)
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Model artifact must implement predict_proba")
    return model


def _read_hmm_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in cfg.values()
    ):
        raise ValueError("HMM configuration must be an object with finite numeric parameters")
    for key in ("beta", "sigma", "v_max", "gap_s", "route_cutoff_m"):
        if key in cfg and cfg[key] <= 0:
            raise ValueError(f"HMM parameter {key} must be positive")
    return cfg


def _load_runtime(artifact_dir: Path, logger: logging.Logger) -> Runtime:
    runtime = Runtime()
    for attribute, filename, load in [
        ("model", "model.joblib", _read_model),
        ("hmm_cfg", "hmm_config.json", _read_hmm_config),
    ]:
        try:
            setattr(runtime, attribute, load(artifact_dir / filename))
        except FileNotFoundError:
            runtime.issues.append(f"Missing {filename}")
        except Exception:
            logger.exception("Could not load %s", filename)
            runtime.issues.append(f"Invalid {filename}")

    try:
        decoder = import_module("pathsnap.hmm")
        if not callable(getattr(decoder, "viterbi", None)) or not isinstance(
            getattr(decoder, "DEFAULT_HMM_CFG", None), dict
        ):
            raise ImportError("Incompatible pathsnap.hmm module")
        runtime.route_metrics = import_module("pathsnap.route_metrics")
    except ImportError as exc:
        logger.warning("Decoder dependencies unavailable: %s", exc)
        runtime.issues.append("PathSnap HMM or route metrics module unavailable")
    return runtime


def _available_trips(eval_dir: Path) -> dict[str, Path]:
    root = eval_dir.resolve()
    return {
        folder.name: folder
        for folder in list_trips(root)
        if re.fullmatch(TRIP_ID_PATTERN, folder.name)
        and folder.resolve().parent == root
        and all((folder / f"{folder.name}.{ext}").is_file() for ext in ("track", "nodes", "arcs"))
    }


def create_app(
    eval_dir: Path | None = None,
    artifact_dir: Path | None = None,
    log_path: Path | None = None,
) -> FastAPI:
    eval_dir = Path(eval_dir if eval_dir is not None else config.EVAL_DIR)
    artifact_dir = Path(artifact_dir if artifact_dir is not None else os.environ.get(
        "PATHSNAP_ARTIFACTS", config.ARTIFACTS_DIR,
    ))
    log_path = Path(log_path if log_path is not None else config.ROOT / "logs" / "api.log")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger = _configure_logger(log_path)
        try:
            app.state.logger = logger
            app.state.runtime = _load_runtime(artifact_dir, logger)
            for issue in app.state.runtime.issues:
                logger.warning("%s", issue)
            yield
        finally:
            for handler in logger.handlers:
                handler.close()

    app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])
    @app.get("/health")
    def health():
        runtime = app.state.runtime
        issues = list(runtime.issues)
        if not eval_dir.is_dir():
            issues.append("Trip directory unavailable")
        return {
            "status": "degraded" if issues else "ok",
            "model": type(runtime.model).__name__ if runtime.model is not None else None,
            "eval_dir": str(eval_dir),
            "issues": issues,
        }

    def get_trips() -> dict[str, Path]:
        try:
            return _available_trips(eval_dir)
        except OSError as exc:
            app.state.logger.warning("Trip directory could not be listed: %s", exc)
            raise HTTPException(status_code=503, detail="Trip directory unavailable") from exc

    @app.get("/trips")
    def trips():
        return {"trips": list(get_trips())}

    @app.post("/predict")
    def predict(request: PredictRequest):
        trip_dir = get_trips().get(request.trip_id)
        if trip_dir is None:
            raise HTTPException(status_code=404, detail="Trip not found")
        runtime = app.state.runtime
        if runtime.issues:
            raise HTTPException(status_code=503, detail="; ".join(runtime.issues))

        start = perf_counter()
        try:
            trip = load_trip(trip_dir, with_route=False)
            load_s = perf_counter() - start
            out = predict_trip(trip, runtime.model, runtime.hmm_cfg)
            point_arcs = np.asarray(out["point_arcs"], dtype=np.int64)
            if point_arcs.shape != (trip.n_points,):
                raise ValueError("Pipeline must return one predicted arc per observation")
        except Exception as exc:
            app.state.logger.exception("Inference failed for trip=%s", request.trip_id)
            raise HTTPException(status_code=500, detail="Route reconstruction failed") from exc

        timings = {key: float(value) for key, value in out["timings"].items()}
        timings["inference_s"] = timings["total_s"]
        timings["load_s"] = load_s
        timings["total_s"] = perf_counter() - start
        return {
            "trip_id": trip.trip_id,
            "n_points": trip.n_points,
            "gps": np.asarray(out["gps"], dtype=float).tolist(),
            "pred_route": np.asarray(out["route_coords"], dtype=float).tolist(),
            "point_arcs": point_arcs.tolist(),
            "timings": timings,
        }

    return app


app = create_app()
