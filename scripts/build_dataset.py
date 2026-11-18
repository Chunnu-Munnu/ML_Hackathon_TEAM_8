"""Build point/candidate feature tables for the train and student_test partitions.

Outputs (results/features/ is gitignored; everything else is committed):
  results/features/{train,student_test}.parquet          one row per (point, candidate)
  results/features/points_{train,student_test}.parquet   one row per GPS point
  results/features/excluded_train_trips.json             train trips with unreliable reference labels
  results/eda/label_alignment.csv                        per-trip GPS-to-route alignment stats
  results/eda/candidate_miss_rate.csv                    miss rate vs radius / K (train only)

The evaluator partition is never touched here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import config  # noqa: E402
from pathsnap.candidates import get_candidates  # noqa: E402
from pathsnap.features import build_trip_features  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.labeling import align_route  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402

SWEEP_R = [20.0, 30.0, 50.0, 75.0]
SWEEP_K = [4, 8, 12]


def process(trip_dir: Path):
    t0 = time.time()
    trip = load_trip(trip_dir, with_route=True)
    net = RoadNetwork(trip)
    lab, ad, pos = align_route(net, trip.route)

    # miss-rate sweep on the widest setting, then filter
    wide = get_candidates(net, r=max(SWEEP_R), k=max(SWEEP_K))
    wide["rank"] = wide.groupby("point_idx").cumcount()
    is_lab = wide["arc_id"].to_numpy() == lab[wide["point_idx"].to_numpy()]
    sweep = []
    for r in SWEEP_R:
        for k in SWEEP_K:
            m = (wide["dist"].to_numpy() <= r) & (wide["rank"].to_numpy() < k)
            hit = np.zeros(len(lab), dtype=bool)
            hit[wide["point_idx"].to_numpy()[m & is_lab]] = True
            n_c = np.bincount(wide["point_idx"].to_numpy()[m], minlength=len(lab))
            sweep.append({"trip_id": trip.trip_id, "split": trip.split, "r": r, "k": k,
                          "n_points": len(lab), "n_miss": int((~hit).sum()), "mean_cands": float(n_c.mean())})

    cands = wide[(wide["dist"] <= config.CAND_RADIUS_M) & (wide["rank"] < config.CAND_K)].drop(columns="rank")
    rows, points = build_trip_features(trip, net, cands=cands.reset_index(drop=True), labels=(lab, ad))
    rows.insert(1, "split", trip.split)

    align = {"trip_id": trip.trip_id, "split": trip.split, "n_points": len(lab), "n_route": len(trip.route),
             "align_median_m": float(np.median(ad)), "align_p95_m": float(np.percentile(ad, 95)),
             "align_max_m": float(ad.max()), "frac_gt_30m": float(np.mean(ad > 30)),
             "frac_gt_50m": float(np.mean(ad > 50)),
             "route_coverage": float(len(np.unique(pos)) / len(trip.route)),
             "miss_rate": float(1 - points["label_in_cands"].mean()),
             "n_rows": len(rows), "epsg": net.proj.epsg, "seconds": round(time.time() - t0, 2)}
    return rows, points, sweep, align


def main(n_jobs: int = 4):
    (config.FEATURES_DIR).mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "eda").mkdir(parents=True, exist_ok=True)
    all_sweep, all_align = [], []
    for split, sdir in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        dirs = list_trips(sdir)
        print(f"[{split}] {len(dirs)} trips ...", flush=True)
        t0 = time.time()
        out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(process)(d) for d in dirs)
        rows = pd.concat([o[0] for o in out], ignore_index=True)
        points = pd.concat([o[1] for o in out], ignore_index=True)
        for o in out:
            all_sweep += o[2]
            all_align.append(o[3])
        rows.to_parquet(config.FEATURES_DIR / f"{split}.parquet", index=False)
        points.to_parquet(config.FEATURES_DIR / f"points_{split}.parquet", index=False)
        print(f"[{split}] rows={len(rows):,} points={len(points):,} positives={int(rows.label.sum()):,} "
              f"miss_rate={1 - points.label_in_cands.mean():.4f} ({time.time() - t0:.0f}s)", flush=True)

    align = pd.DataFrame(all_align)
    align.to_csv(config.RESULTS_DIR / "eda" / "label_alignment.csv", index=False)

    sweep = pd.DataFrame(all_sweep)
    sw = (sweep[sweep.split == "train"].groupby(["r", "k"])
          .agg(n_points=("n_points", "sum"), n_miss=("n_miss", "sum"),
               mean_cands=("mean_cands", "mean")).reset_index())
    sw["miss_rate"] = sw["n_miss"] / sw["n_points"]
    sw.to_csv(config.RESULTS_DIR / "eda" / "candidate_miss_rate.csv", index=False)
    print(sw.to_string(index=False))

    bad = align[(align.split == "train") & (align.align_median_m > config.BAD_LABEL_MEDIAN_M)]
    excl = {"rule": f"train trips whose median GPS-to-reference-route alignment distance > "
                    f"{config.BAD_LABEL_MEDIAN_M} m (GPS and map do not line up; labels unreliable). "
                    f"Rows kept in train.parquet; dropped before model fitting.",
            "trips": sorted(bad.trip_id.tolist())}
    (config.FEATURES_DIR / "excluded_train_trips.json").write_text(json.dumps(excl, indent=2))
    (config.RESULTS_DIR / "eda" / "excluded_train_trips.json").write_text(json.dumps(excl, indent=2))
    print("excluded train trips:", excl["trips"])


if __name__ == "__main__":
    main()
