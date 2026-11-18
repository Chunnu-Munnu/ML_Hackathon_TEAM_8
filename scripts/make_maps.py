"""Generate offline Folium previews from the permitted PathSnap partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

from pathsnap import config
from pathsnap.io import list_trips, load_trip
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
    parser.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR / "maps")
    args = parser.parse_args()

    train_dir = checked_trip(config.TRAIN_DIR, args.train_trip, parser)
    trip = load_trip(train_dir)
    write_map(trip_map(trip), args.output_dir / f"train_{trip.trip_id}_raw.html")


if __name__ == "__main__":
    main()
