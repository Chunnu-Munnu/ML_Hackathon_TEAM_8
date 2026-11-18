"""Run every trip the API lists through POST /predict and save the server-side metrics.

This exercises exactly the evaluation path used in the demo (UI -> API): the API loads each trip without its
route, reconstructs it, and only then scores it against its own reference data. Nothing here sees labels.

usage:
  python scripts/run_evaluator.py --api http://localhost:8000 --out results/evaluator/evaluator_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "evaluator" / "evaluator_results.csv"))
    a = ap.parse_args()
    health = requests.get(f"{a.api}/health", timeout=10).json()
    print("API health:", health)
    trips = requests.get(f"{a.api}/trips", timeout=10).json()["trips"]
    rows = []
    for t in trips:
        r = requests.post(f"{a.api}/predict", json={"trip_id": t}, timeout=600)
        r.raise_for_status()
        j = r.json(); m = j["metrics"]; tm = j["timings"]
        rows.append({"trip_id": t, "n_points": j["n_points"], "predicted_segments": len(j["pred_route"]),
                     "top1_accuracy": m["top1_accuracy"], "continuity": m["continuity"], "route_f1": m["route_f1"],
                     "n_breaks": m["n_breaks"], "inference_s": tm.get("inference_s"), "total_s": tm.get("total_s")})
        print(f"{t}: points={j['n_points']:5d} top1={m['top1_accuracy']:.4f} continuity={m['continuity']:.4f} "
              f"route_f1={m['route_f1']:.4f} total={tm.get('total_s', 0):.2f}s", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False)
    w = df.n_points / df.n_points.sum()
    print(f"\n{len(df)} trips | pooled Top-1 {(df.top1_accuracy * w).sum():.4f} | mean trip Top-1 {df.top1_accuracy.mean():.4f}"
          f" | pooled continuity {(df.continuity * w).sum():.4f} | mean route F1 {df.route_f1.mean():.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
