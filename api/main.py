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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.labeling import align_route
from pathsnap.network import RoadNetwork
from pathsnap.pipeline import predict_trip

TRIP_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
MAX_GPS_POINTS = 3000
Coordinate = tuple[float, float]


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=TRIP_ID_PATTERN, min_length=1, max_length=64)
    model: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")


class PredictionMetrics(BaseModel):
    top1_accuracy: float | None
    continuity: float | None
    n_breaks: int
    route_f1: float | None


class Prediction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    trip_id: str
    n_points: int
    gps: list[Coordinate]
    pred_route: list[tuple[Coordinate, Coordinate]]
    point_arcs: list[int]
    network: list[tuple[Coordinate, Coordinate]] = []
    model: str = "deployed"
    metrics: PredictionMetrics
    timings: dict[str, float]


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
    if not np.array_equal(getattr(model, "classes_", None), [0, 1]):
        raise ValueError("Model artifact must score candidate classes 0 and 1 in that order")
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and list(feature_names) != FEATURES:
        raise ValueError("Model artifact feature columns do not match the prediction pipeline")
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
        route_metrics = import_module("pathsnap.route_metrics")
        if not all(callable(getattr(route_metrics, name, None)) for name in (
            "continuity", "point_accuracy", "route_prf",
        )):
            raise ImportError("Incompatible pathsnap.route_metrics module")
        runtime.route_metrics = route_metrics
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


def _score_route(
    trip_dir: Path,
    point_arcs: np.ndarray,
    net: RoadNetwork,
    route_metrics: ModuleType,
    n_breaks: int,
    logger: logging.Logger,
) -> PredictionMetrics:
    """Read the optional reference route only after reconstruction has finished."""
    continuity = route_metrics.continuity(point_arcs, net)
    metrics = PredictionMetrics(
        top1_accuracy=None, continuity=float(continuity) if np.isfinite(continuity) else None,
        n_breaks=n_breaks, route_f1=None,
    )
    try:
        route = load_route(trip_dir)
        if not len(route) or ((route < 0) | (route >= net.n_arcs)).any():
            raise ValueError("Reference route is empty or contains invalid arc IDs")
    except (OSError, ValueError) as exc:
        logger.warning("Reference scoring unavailable for trip=%s (%s)", trip_dir.name, type(exc).__name__)
        return metrics

    labels, _, _ = align_route(net, route)
    metrics.top1_accuracy = float(route_metrics.point_accuracy(point_arcs, labels))
    metrics.route_f1 = float(route_metrics.route_prf(point_arcs, net.canon[route])["f1"])
    return metrics


MODELS_DIR = config.RESULTS_DIR / "models"
MAX_NETWORK_SEGMENTS = 3000


def _model_catalog() -> dict[str, dict]:
    """Every trained model with a saved file (results/models/<name>/model.joblib) and its recorded scores."""
    out = {}
    for f in sorted(MODELS_DIR.glob("*/model.joblib")):
        name = f.parent.name
        info = {"name": name}
        for fn in ("metrics.json", "config.json"):
            p = f.parent / fn
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                info.update({k: v for k, v in d.items() if k in (
                    "family", "cv_top1_mean", "cv_top1_std", "cv_macro_f1_mean", "cv_log_loss_mean", "train_top1",
                    "st_top1", "st_macro_f1", "st_log_loss", "fit_s", "infer_s_per_trip", "subsample_points")})
        out[name] = info
    return out


def _nearby_network(net: RoadNetwork, radius_m: float = 60.0) -> list:
    """Road segments near the track (map context only; this is the trip's road network, not the reference)."""
    from pathsnap.viz import _nearby_arcs
    ids = _nearby_arcs(net, radius_m)[:MAX_NETWORK_SEGMENTS]
    return net.arc_lonlat(ids) if len(ids) else []


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
    @app.middleware("http")
    async def log_request(request: Request, call_next):
        start = perf_counter()
        response = await call_next(request)
        app.state.logger.info(
            "%s %r status=%d latency_s=%.4f",
            request.method, request.url.path, response.status_code, perf_counter() - start,
        )
        return response

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

    model_cache: dict = {}

    @app.get("/models")
    def models():
        catalog = _model_catalog()
        deployed = (json.loads((artifact_dir / "metadata.json").read_text()) if (artifact_dir / "metadata.json").is_file()
                    else {}).get("model_name")
        hmm_rows = {}
        hc = config.RESULTS_DIR / "hmm" / "hmm_comparison.csv"
        if hc.is_file():
            import csv
            with open(hc, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r["trip_id"] == "ALL" and r["variant"] == "C+H":
                        hmm_rows[r["model"]] = {"st_top1_hmm": float(r["top1"]), "st_continuity_hmm": float(r["continuity"])}
        for k, v in catalog.items():
            v.update(hmm_rows.get(k, {}))
        return {"deployed": deployed, "models": list(catalog.values())}

    def resolve_model(name: str | None):
        runtime = app.state.runtime
        if not name or name == "deployed":
            return runtime.model, runtime.hmm_cfg, "deployed"
        path = MODELS_DIR / name / "model.joblib"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Model not found")
        if name not in model_cache:
            model_cache[name] = _read_model(path)
        cfg_path = config.RESULTS_DIR / "hmm" / "decoder_per_model" / f"{name}_config.json"
        cfg = _read_hmm_config(cfg_path) if cfg_path.is_file() else runtime.hmm_cfg
        return model_cache[name], cfg, name

    @app.get("/metrics")
    def metrics():
        """Precomputed experiment summary (results/summary.json): model comparison, tuning, HMM and
        final-evaluation metrics. Contains aggregate scores only, never reference routes or labels."""
        path = config.RESULTS_DIR / "summary.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Run scripts/build_summary.py to create the results summary")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        """Static results page generated by scripts/build_summary.py."""
        path = config.RESULTS_DIR / "dashboard.html"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Run scripts/build_summary.py to create the dashboard")
        return HTMLResponse(path.read_text(encoding="utf-8"))

    @app.post("/predict", response_model=Prediction)
    def predict(request: PredictRequest):
        trip_dir = get_trips().get(request.trip_id)
        if trip_dir is None:
            raise HTTPException(status_code=404, detail="Trip not found")
        runtime = app.state.runtime
        if runtime.issues:
            raise HTTPException(status_code=503, detail="; ".join(runtime.issues))

        start = perf_counter()
        logger = app.state.logger
        try:
            trip = load_trip(trip_dir, with_route=False)
            load_s = perf_counter() - start
            model_obj, hmm_cfg, model_name = resolve_model(request.model)
            out = predict_trip(trip, model_obj, hmm_cfg)
            net = out["net"]
            point_arcs = np.asarray(out["point_arcs"], dtype=np.int64)
            if point_arcs.shape != (trip.n_points,) or (
                (point_arcs < -1) | (point_arcs >= net.n_arcs)
            ).any():
                raise ValueError("Pipeline must return one valid arc ID or -1 per observation")

            score_start = perf_counter()
            metrics = _score_route(
                trip_dir, point_arcs, net, runtime.route_metrics,
                int(out["info"].get("n_breaks", 0)), logger,
            )
            timings = {key: float(value) for key, value in out["timings"].items()}
            timings["evaluation_s"] = perf_counter() - score_start
            timings["inference_s"] = timings["total_s"]
            timings["load_s"] = load_s
            gps = np.asarray(out["gps"], dtype=float)
            if len(gps) > MAX_GPS_POINTS:
                indices = np.linspace(0, len(gps) - 1, MAX_GPS_POINTS, dtype=int)
                gps = gps[indices]
            timings["total_s"] = perf_counter() - start
            result = Prediction(
                trip_id=trip.trip_id, n_points=trip.n_points,
                gps=gps.tolist(), pred_route=np.asarray(out["route_coords"], dtype=float).tolist(),
                point_arcs=point_arcs.tolist(), timings=timings, metrics=metrics,
                network=_nearby_network(net), model=model_name,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Reconstruction failed for trip=%s", request.trip_id)
            raise HTTPException(status_code=500, detail="Route reconstruction failed") from exc

        logger.info(
            "PREDICT trip=%s model=%s n_points=%d top1=%s continuity=%s total_s=%.4f gps_sent=%d",
            trip.trip_id, result.model, trip.n_points, metrics.top1_accuracy, metrics.continuity,
            timings["total_s"], len(gps),
        )
        return result

    return app


app = create_app()
