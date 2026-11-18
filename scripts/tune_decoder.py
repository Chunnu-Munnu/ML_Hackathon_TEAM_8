"""Tune the HMM decoder parameters with Bayesian optimisation (Optuna TPE).

Search space : lam in [0.1, 2] (log), beta in [1, 50] (log), v_max in [40, 80], gap_s in [10, 120]
Decoder mode : signed backward moves + soft speed penalty (see hmm.DEFAULT_HMM_CFG)
Objective    : point-weighted C+H point Top-1 on the 10 student-test trips, using the chosen model's
               student_test_proba.parquet. Student-test is a permitted development partition; evaluator data is
               never touched. (Because the decoder is tuned on student-test, the student-test C+H numbers are
               optimistic; say so in the report.)
Fixed        : sigma (unused with proba emission), backward_tol_m, route_cutoff_m, min_logp keep their defaults.

The expensive network-distance computation is done once per trip (hmm.prepare); each trial only re-runs the
Viterbi recurrence.

Outputs: results/hmm/decoder_search.csv, artifacts/hmm_config.json
Usage  : python scripts/tune_decoder.py [--model lgbm_tuned] [--trials 60] [--timeout 1800]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import config  # noqa: E402
from pathsnap.candidates import get_candidates  # noqa: E402
from pathsnap.hmm import DEFAULT_HMM_CFG, cand_geometry, decode_prepared, prepare  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.route_metrics import point_accuracy  # noqa: E402

MODELS_DIR = config.RESULTS_DIR / "models"
HMM_DIR = config.RESULTS_DIR / "hmm"
GAP_MAX = 120.0


def pick_model(arg: str | None) -> str:
    if arg:
        return arg
    for name in ["lgbm_tuned", "lgbm"]:
        if (MODELS_DIR / name / "student_test_proba.parquet").exists():
            return name
    found = sorted(p.parent.name for p in MODELS_DIR.glob("*/student_test_proba.parquet"))
    if not found:
        raise SystemExit("no results/models/*/student_test_proba.parquet found; train models first")
    return found[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    model = pick_model(args.model)
    print(f"tuning decoder for model: {model}")
    points = pd.read_parquet(config.FEATURES_DIR / "points_student_test.parquet")
    proba = pd.read_parquet(MODELS_DIR / model / "student_test_proba.parquet")

    # Parameter-independent work, once per trip.
    data = []
    for trip_dir in list_trips(config.STUDENT_TEST_DIR):
        trip = load_trip(trip_dir, with_route=False)   # labels come from points_*.parquet, route not needed
        net = RoadNetwork(trip)
        n = len(net.x)
        rows = proba[proba["trip_id"] == trip.trip_id][["point_idx", "arc_id", "proba"]]
        pt = points[points["trip_id"] == trip.trip_id]
        lab = np.full(n, -1, dtype=np.int64)
        lab[pt["point_idx"].to_numpy()] = pt["label_arc"].to_numpy()
        cands = cand_geometry(rows, net)
        t0 = time.perf_counter()
        prep = prepare(cands, net, DEFAULT_HMM_CFG, max_gap_s=GAP_MAX)
        data.append((trip.trip_id, prep, lab))
        print(f"  prepared {trip.trip_id}: {n} points in {time.perf_counter() - t0:.1f}s")

    def evaluate(cfg: dict) -> tuple[float, int]:
        tot, wsum, br = 0.0, 0, 0
        for _, prep, lab in data:
            arcs, info = decode_prepared(prep, cfg, "proba")
            tot += point_accuracy(arcs, lab) * len(lab)
            wsum += len(lab)
            br += info["n_breaks"]
        return tot / wsum, br

    base_score, base_breaks = evaluate(DEFAULT_HMM_CFG)
    print(f"default decoder config: C+H top1 = {base_score:.4f} (n_breaks {base_breaks})")

    def objective(trial: optuna.Trial) -> float:
        cfg = {**DEFAULT_HMM_CFG,
               "lam": trial.suggest_float("lam", 0.1, 2.0, log=True),
               "beta": trial.suggest_float("beta", 1.0, 50.0, log=True),
               "v_max": trial.suggest_float("v_max", 40.0, 80.0),
               "gap_s": trial.suggest_float("gap_s", 10.0, GAP_MAX)}
        s, br = evaluate(cfg)
        trial.set_user_attr("n_breaks", br)
        return s

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED),
                                study_name=f"pathsnap_decoder_{model}")
    study.enqueue_trial({k: DEFAULT_HMM_CFG[k] for k in ["lam", "beta", "v_max", "gap_s"]})   # start from defaults
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    HMM_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe()
    df.insert(0, "model", model)
    df.to_csv(HMM_DIR / "decoder_search.csv", index=False)

    best = {**DEFAULT_HMM_CFG, **study.best_params}
    (config.ARTIFACTS_DIR / "hmm_config.json").write_text(json.dumps(best, indent=2))
    print(f"\ntrials: {len(study.trials)}   best C+H top1 = {study.best_value:.4f}   (default {base_score:.4f})")
    print("best decoder config:", json.dumps(best))
    print(f"wrote {HMM_DIR / 'decoder_search.csv'} and {config.ARTIFACTS_DIR / 'hmm_config.json'}")


if __name__ == "__main__":
    main()
