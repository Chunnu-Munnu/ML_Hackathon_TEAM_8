"""Generate offline Folium previews from the permitted PathSnap partitions."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import joblib

from pathsnap import config
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.pipeline import predict_trip
from pathsnap.viz import trip_map


def write_map(fmap, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = fmap.get_root().render()
    path.write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n", encoding="utf-8")
    print(path)


def checked_trip(root: Path, trip_id: str, parser: argparse.ArgumentParser) -> Path:
    trips = {trip.name: trip for trip in list_trips(root)}
    if trip_id not in trips:
        parser.error(f"Unknown trip {trip_id!r} in {root.name}")
    return trips[trip_id]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trip", default="00000001")
    parser.add_argument("--train-only", action="store_true", help="Generate the training preview only")
    parser.add_argument("--student-trips", nargs="+", default=["00000000", "00000007"])
    parser.add_argument("--model", type=Path, default=config.ARTIFACTS_DIR / "model.joblib")
    parser.add_argument("--hmm-config", type=Path, default=config.ARTIFACTS_DIR / "hmm_config.json")
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR / "maps")
    args = parser.parse_args()

    train_dir = checked_trip(config.TRAIN_DIR, args.train_trip, parser)
    trip = load_trip(train_dir)
    write_map(trip_map(trip), args.output_dir / f"train_{trip.trip_id}_raw.html")
    if args.train_only:
        return

    try:
        import_module("pathsnap.hmm")
        model = joblib.load(args.model)
        hmm_cfg = json.loads(args.hmm_config.read_text(encoding="utf-8"))
        if not isinstance(hmm_cfg, dict):
            raise ValueError("HMM configuration must contain a JSON object")
    except (ImportError, OSError, ValueError) as exc:
        parser.error(f"Cannot generate student-test predictions: {exc}. Use --train-only for the raw map.")

    for trip_id in args.student_trips:
        trip_dir = checked_trip(config.STUDENT_TEST_DIR, trip_id, parser)
        unlabeled_trip = load_trip(trip_dir, with_route=False)
        prediction = predict_trip(unlabeled_trip, model, hmm_cfg)
        labeled_trip = replace(unlabeled_trip, route=load_route(trip_dir))
        fmap = trip_map(labeled_trip, net=prediction["net"], pred_arcs=prediction["route_arcs"])
        write_map(fmap, args.output_dir / f"st_{trip_id}_pred.html")


if __name__ == "__main__":
    main()
