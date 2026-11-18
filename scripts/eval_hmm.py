"""Evaluate route reconstruction on the 10 student-test trips.

Variants
  B0   nearest arc, no ML, no HMM
  B1   nearest-arc emission + HMM (emission="dist")
  C    classifier only (argmax probability per point)           -- for every model with student_test_proba.parquet
  C+H  classifier probabilities + HMM/Viterbi (emission="proba") -- same models

Uses student-test trips only (never evaluator data). Decoder parameters come from --cfg, else
artifacts/hmm_config.json if it exists, else DEFAULT_HMM_CFG.

Outputs
  results/hmm/hmm_comparison.csv   model, variant, trip_id, top1, continuity, route_precision, route_recall,
                                   route_f1, n_breaks, decode_s   (+ rows with trip_id == "ALL")
  results/hmm/hmm_gain.png
  results/models/model_comparison.csv   -> fills the `route_continuity` column (C+H continuity, ALL)
  results/maps/hmm_helped_<trip>.html x2, hmm_hurt_<trip>.html x1   (needs pathsnap.viz.trip_map)

"ALL" rows: top1 / continuity / route_* are point-weighted means over trips; n_breaks and decode_s are sums.
decode_s for C+H includes the network-distance preparation (shared across models) plus Viterbi itself.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import config  # noqa: E402
from pathsnap.candidates import get_candidates  # noqa: E402
from pathsnap.hmm import DEFAULT_HMM_CFG, cand_geometry, decode_prepared, prepare  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.route_metrics import continuity, point_accuracy, route_prf  # noqa: E402

HMM_DIR = config.RESULTS_DIR / "hmm"
MODELS_DIR = config.RESULTS_DIR / "models"
MAPS_DIR = config.RESULTS_DIR / "maps"
COLS = ["model", "variant", "trip_id", "top1", "continuity", "route_precision", "route_recall",
        "route_f1", "n_breaks", "decode_s"]


def load_cfg(path: str | None) -> dict:
    p = Path(path) if path else config.ARTIFACTS_DIR / "hmm_config.json"
    if p.exists():
        print(f"decoder config: {p}")
        return {**DEFAULT_HMM_CFG, **json.loads(p.read_text())}
    print("decoder config: DEFAULT_HMM_CFG (no hmm_config.json yet)")
    return dict(DEFAULT_HMM_CFG)


def score(model: str, variant: str, trip_id: str, arcs, lab, route_canon, net, n_breaks, decode_s) -> dict:
    prf = route_prf(arcs, route_canon)
    return {"model": model, "variant": variant, "trip_id": trip_id,
            "top1": point_accuracy(arcs, lab), "continuity": continuity(arcs, net),
            "route_precision": prf["precision"], "route_recall": prf["recall"], "route_f1": prf["f1"],
            "n_breaks": n_breaks, "decode_s": decode_s, "_n": len(lab)}


def argmax_arcs(rows: pd.DataFrame, n: int, col: str = "proba") -> np.ndarray:
    out = np.full(n, -1, dtype=np.int64)
    if len(rows):
        best = rows.loc[rows.groupby("point_idx")[col].idxmax()]
        out[best["point_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return out


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, v), g in df.groupby(["model", "variant"], sort=False):
        w = g["_n"].to_numpy(dtype=float)

        def wm(c):
            x = g[c].to_numpy(dtype=float)
            ok = ~np.isnan(x)
            return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
        out.append({"model": m, "variant": v, "trip_id": "ALL", "top1": wm("top1"), "continuity": wm("continuity"),
                    "route_precision": wm("route_precision"), "route_recall": wm("route_recall"),
                    "route_f1": wm("route_f1"), "n_breaks": int(g["n_breaks"].sum()),
                    "decode_s": float(g["decode_s"].sum()), "_n": int(w.sum())})
    return pd.concat([df, pd.DataFrame(out)], ignore_index=True)


def plot_gain(allrows: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = allrows[allrows["trip_id"] == "ALL"]
    models = [m for m in a["model"].unique() if m != "baseline"]
    if not models:
        return
    c = a[a["variant"] == "C"].set_index("model")
    ch = a[a["variant"] == "C+H"].set_index("model")
    models = [m for m in models if m in c.index and m in ch.index]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.1 * len(models) + 5), 4.5))
    for ax, col, title in zip(axes, ["top1", "continuity"], ["Point Top-1 accuracy", "Route continuity"]):
        ax.bar(x - 0.2, [c.loc[m, col] for m in models], 0.4, label="classifier only (C)")
        ax.bar(x + 0.2, [ch.loc[m, col] for m in models], 0.4, label="classifier + HMM (C+H)")
        for name, ls in [("B0", ":"), ("B1", "--")]:
            r = a[(a["model"] == "baseline") & (a["variant"] == name)]
            if len(r):
                ax.axhline(float(r[col].iloc[0]), color="gray", ls=ls, lw=1.2, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right")
        ax.set_title(f"{title} (student-test, 10 trips)")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None, help="decoder config json (default artifacts/hmm_config.json)")
    ap.add_argument("--map-model", default=None, help="model used for the helped/hurt maps (default: best C+H Top-1)")
    ap.add_argument("--no-maps", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    HMM_DIR.mkdir(parents=True, exist_ok=True)

    points = pd.read_parquet(config.FEATURES_DIR / "points_student_test.parquet")
    proba_files = {p.parent.name: p for p in sorted(MODELS_DIR.glob("*/student_test_proba.parquet"))}
    print(f"{len(proba_files)} models with student_test_proba.parquet: {list(proba_files)}")
    probas = {name: pd.read_parquet(p) for name, p in proba_files.items()}

    records: list[dict] = []
    keep_preds: dict[tuple[str, str], dict[str, np.ndarray]] = {}   # (model, variant) -> trip_id -> arcs
    trips = {}
    for trip_dir in list_trips(config.STUDENT_TEST_DIR):
        trip = load_trip(trip_dir, with_route=True)          # development partition: labels allowed here
        tid = trip.trip_id
        net = RoadNetwork(trip)
        n = len(net.x)
        trips[tid] = (trip, net)
        pt = points[points["trip_id"] == tid]
        lab = np.full(n, -1, dtype=np.int64)
        lab[pt["point_idx"].to_numpy()] = pt["label_arc"].to_numpy()
        route_canon = net.canon[trip.route]

        cands = cand_geometry(get_candidates(net), net)
        t0 = time.perf_counter()
        prep = prepare(cands, net, cfg)
        prep_s = time.perf_counter() - t0

        # B0 / B1
        t0 = time.perf_counter()
        b0 = np.full(n, -1, dtype=np.int64)
        first = cands.groupby("point_idx").head(1)            # sorted by distance
        b0[first["point_idx"].to_numpy()] = first["arc_id"].to_numpy()
        records.append(score("baseline", "B0", tid, b0, lab, route_canon, net, 0, time.perf_counter() - t0))
        t0 = time.perf_counter()
        b1, info = decode_prepared(prep, cfg, "dist")
        records.append(score("baseline", "B1", tid, b1, lab, route_canon, net, info["n_breaks"],
                             prep_s + time.perf_counter() - t0))

        # classifier variants
        for name, pr in probas.items():
            rows = pr[pr["trip_id"] == tid][["point_idx", "arc_id", "proba"]]
            if rows.empty:
                print(f"  [warn] {name}: no rows for trip {tid}")
                continue
            t0 = time.perf_counter()
            c_arcs = argmax_arcs(rows, n)
            records.append(score(name, "C", tid, c_arcs, lab, route_canon, net, 0, time.perf_counter() - t0))

            merged = cands.merge(rows, on=["point_idx", "arc_id"], how="left")
            if len(merged) == len(cands) and merged["proba"].notna().all():
                mprep, extra = dataclasses.replace(prep, proba=merged["proba"].to_numpy(dtype=float)), prep_s
            else:   # classifier saw a different candidate set: prepare separately on its own rows
                own = cand_geometry(rows, net)
                t0 = time.perf_counter()
                mprep = prepare(own, net, cfg)
                extra = time.perf_counter() - t0
            t0 = time.perf_counter()
            h_arcs, info = decode_prepared(mprep, cfg, "proba")
            records.append(score(name, "C+H", tid, h_arcs, lab, route_canon, net, info["n_breaks"],
                                 extra + time.perf_counter() - t0))
            keep_preds.setdefault((name, "C"), {})[tid] = c_arcs
            keep_preds.setdefault((name, "C+H"), {})[tid] = h_arcs
        print(f"trip {tid}: {n} points done")

    df = aggregate(pd.DataFrame(records))
    df[COLS].to_csv(HMM_DIR / "hmm_comparison.csv", index=False)
    print(f"wrote {HMM_DIR / 'hmm_comparison.csv'}")
    plot_gain(df, HMM_DIR / "hmm_gain.png")

    # summary on screen
    allr = df[df["trip_id"] == "ALL"][["model", "variant", "top1", "continuity", "route_f1", "n_breaks", "decode_s"]]
    print(allr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # fill route_continuity in the model comparison table
    mc = MODELS_DIR / "model_comparison.csv"
    if mc.exists():
        comp = pd.read_csv(mc)
        cont = df[(df["trip_id"] == "ALL") & (df["variant"] == "C+H")].set_index("model")["continuity"]
        new = comp["name"].map(cont)
        comp["route_continuity"] = new.combine_first(comp["route_continuity"]) if "route_continuity" in comp else new
        comp.to_csv(mc, index=False)
        print("filled route_continuity in model_comparison.csv")
    else:
        print("model_comparison.csv not found yet; re-run after the experiment harness has written it")

    if args.no_maps or not keep_preds:
        return
    best = df[(df["trip_id"] == "ALL") & (df["variant"] == "C+H") & (df["model"] != "baseline")]
    map_model = args.map_model or (best.sort_values("top1", ascending=False)["model"].iloc[0] if len(best) else None)
    if map_model is None or (map_model, "C+H") not in keep_preds:
        return
    try:
        from pathsnap.viz import trip_map
    except ImportError:
        print("pathsnap.viz not available yet: skipping maps; re-run when it is pushed")
        return
    per = df[(df["model"] == map_model) & (df["trip_id"] != "ALL")].pivot(index="trip_id", columns="variant", values="top1")
    delta = (per["C+H"] - per["C"]).sort_values()
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    picks = [("hmm_helped", t) for t in delta[delta > 0].index[::-1][:2]]
    if len(delta) and delta.iloc[0] < 0:
        picks.append(("hmm_hurt", delta.index[0]))
    for prefix, tid in picks:
        trip, net = trips[tid]
        m = trip_map(trip, net, pred_arcs=keep_preds[(map_model, "C+H")][tid], show_route=True)
        m.save(str(MAPS_DIR / f"{prefix}_{tid}.html"))
        print(f"map: {prefix}_{tid}.html  (delta top1 = {delta[tid]:+.4f}, model {map_model})")


if __name__ == "__main__":
    main()
