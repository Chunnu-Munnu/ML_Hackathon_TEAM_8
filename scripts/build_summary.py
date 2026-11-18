"""Collect every final result into results/summary.json and a static results/dashboard.html.

Reads only files that the pipeline scripts already wrote (no recomputation, no ground truth):
  results/eda/*.csv, results/models/*/{metrics,config}.json, results/optuna/*, results/hmm/hmm_comparison.csv,
  results/evaluator/evaluator_results.csv (written by scripts/run_evaluator.py through the API),
  artifacts/{metadata,hmm_config}.json
The API serves summary.json at GET /metrics; the UI's Results tab and the static dashboard render it.
"""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import config  # noqa: E402

R = config.RESULTS_DIR
NICE = {"lgbm_tuned": "LightGBM (Optuna-tuned)", "lgbm": "LightGBM", "rf": "Random Forest", "dtree": "Decision Tree",
        "logreg_C0.1": "Logistic Regression (C=0.1)", "logreg_C1.0": "Logistic Regression (C=1)",
        "knn": "KNN (distance-weighted)", "gnb_vs1e-06": "Gaussian NB (vs=1e-6)", "gnb_vs1e-09": "Gaussian NB (vs=1e-9)",
        "svm_linear_C0.1": "Linear SVM (C=0.1)", "svm_linear_C1.0": "Linear SVM (C=1)",
        "mlp_64_32": "MLP (64, 32)", "mlp_128_64_32": "MLP (128, 64, 32)", "b0_nearest": "Nearest arc (baseline)"}


def _read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def build() -> dict:
    out: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    al = pd.read_csv(R / "eda" / "label_alignment.csv", dtype={"trip_id": str})
    tr = al[al.split == "train"]
    excl = (_read_json(R / "eda" / "excluded_train_trips.json") or {}).get("trips", [])
    ok = tr[~tr.trip_id.isin(excl)]
    sw = pd.read_csv(R / "eda" / "candidate_miss_rate.csv")
    out["dataset"] = {
        "train_trips": int(len(tr)), "student_test_trips": int((al.split == "student_test").sum()),
        "train_points": int(tr.n_points.sum()), "excluded_train_trips": excl,
        "train_rows": int(ok.n_rows.sum()), "utm_zones": int(tr.epsg.nunique()),
        "median_alignment_m": round(float(ok.align_median_m.median()), 2),
        "candidate_radius_m": config.CAND_RADIUS_M, "candidate_k": config.CAND_K,
        "train_candidate_miss_rate_usable_trips": round(float((ok.miss_rate * ok.n_points).sum() / ok.n_points.sum()), 4),
        "candidate_sweep_all_train": sw.round(4).to_dict(orient="records"),
    }

    from pathsnap.features import FEATURES
    out["features"] = {"n": len(FEATURES), "names": FEATURES}

    models = []
    for f in sorted((R / "models").glob("*/metrics.json")):
        name = f.parent.name
        m = json.loads(f.read_text()); c = _read_json(f.parent / "config.json") or {}
        models.append({"name": name, "label": NICE.get(name, name), "family": c.get("family"),
                       "config": c.get("config"), "subsample_points": c.get("subsample_points"),
                       **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k != "st_confusion"},
                       "st_confusion": m.get("st_confusion")})
    hc = R / "hmm" / "hmm_comparison.csv"
    hmm_all = pd.read_csv(hc) if hc.exists() else pd.DataFrame()
    if len(hmm_all):
        a = hmm_all[hmm_all.trip_id == "ALL"]
        for m in models:
            for var, key in [("C", "st_top1_classifier_only"), ("C+H", "st_top1_with_hmm")]:
                r = a[(a.model == m["name"]) & (a.variant == var)]
                if len(r):
                    m[key] = round(float(r.top1.iloc[0]), 5)
                    if var == "C+H":
                        m["st_continuity_with_hmm"] = round(float(r.continuity.iloc[0]), 5)
                        m["st_route_f1_with_hmm"] = round(float(r.route_f1.iloc[0]), 5)
        out["hmm_variants_student_test"] = a.round(5).to_dict(orient="records")
    out["models"] = sorted(models, key=lambda m: -(m.get("cv_top1_mean") or 0))

    out["optuna"] = _read_json(R / "optuna" / "best_params.json")
    tv = R / "optuna" / "tuned_vs_untuned.csv"
    out["tuned_vs_untuned"] = pd.read_csv(tv).round(5).to_dict(orient="records") if tv.exists() else None

    out["deployed"] = {"metadata": _read_json(config.ARTIFACTS_DIR / "metadata.json"),
                       "hmm_config": _read_json(config.ARTIFACTS_DIR / "hmm_config.json")}

    ev = R / "evaluator" / "evaluator_results.csv"
    if ev.exists():
        e = pd.read_csv(ev, dtype={"trip_id": str})
        w = e.n_points / e.n_points.sum()
        out["evaluator"] = {
            "trips": e.round(5).to_dict(orient="records"),
            "pooled_top1": round(float((e.top1_accuracy * w).sum()), 5),
            "mean_trip_top1": round(float(e.top1_accuracy.mean()), 5),
            "pooled_continuity": round(float((e.continuity * w).sum()), 5),
            "mean_route_f1": round(float(e.route_f1.mean()), 5),
            "mean_latency_s": round(float(e.total_s.mean()), 3),
        }
    st = R / "student_test_final.csv"
    if st.exists():
        s = pd.read_csv(st, dtype={"trip_id": str}); w = s.n_points / s.n_points.sum()
        out["student_test_final"] = {"trips": s.round(5).to_dict(orient="records"),
                                     "pooled_top1": round(float((s.top1_accuracy * w).sum()), 5),
                                     "mean_trip_top1": round(float(s.top1_accuracy.mean()), 5),
                                     "pooled_continuity": round(float((s.continuity * w).sum()), 5)}
    return out


# ------------------------------------------------------------------ static dashboard
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16202c;--muted:#5b6675;--line:#e3e7ec;--acc:#1f5fbf;--good:#1d8a4e;--warn:#b5582b}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#11161d;--card:#18202a;--ink:#e6ebf1;--muted:#98a4b3;--line:#2a3542;--acc:#6ea2ff;--good:#4cc282;--warn:#f0a070}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 60px}h1{font-size:28px;margin:0}h2{font-size:19px;margin:34px 0 10px}
.sub{color:var(--muted);margin:4px 0 22px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}.k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.v{font-size:26px;font-weight:650;margin-top:2px}.v.good{color:var(--good)}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13.5px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:color-mix(in srgb,var(--acc) 8%,var(--card));font-weight:600}
tr.best td{font-weight:650}.wrap{overflow-x:auto}.bar{height:8px;background:var(--line);border-radius:4px;min-width:90px}.bar>i{display:block;height:100%;background:var(--acc);border-radius:4px}
.note{color:var(--muted);font-size:13px;margin-top:8px}code{font-size:12.5px}
"""


def pct(v):
    return "—" if v is None or v != v else f"{100 * v:.1f}%"


def num(v, d=3):
    return "—" if v is None or v != v else f"{v:.{d}f}"


def dashboard(s: dict) -> str:
    e = s.get("evaluator"); ds = s["dataset"]; best = s["models"][0] if s["models"] else {}
    dep = (s.get("deployed") or {}).get("metadata") or {}
    stf = s.get("student_test_final") or {}
    cards = [("Evaluator Top-1 (pooled)", pct(e and e["pooled_top1"]), True),
             ("Evaluator route continuity", pct(e and e["pooled_continuity"]), False),
             ("Evaluator mean route F1", pct(e and e["mean_route_f1"]), False),
             ("Student-test Top-1 (pooled)", pct(stf.get("pooled_top1")), False),
             ("Best CV Top-1 (train)", pct(best.get("cv_top1_mean")), False),
             ("Deployed model", html.escape(NICE.get(str(dep.get("model_name")), str(dep.get("model_name", "—")))), False)]
    h = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>PathSnap Results</title><style>{CSS}</style></head><body><main>",
         "<h1>PathSnap results</h1>",
         f"<p class=sub>Team 8 · map-matching and route reconstruction · generated {s['generated_utc']} from the files in <code>results/</code></p>",
         "<div class=grid>" + "".join(f"<div class=card><div class=k>{k}</div><div class='v{' good' if g else ''}'>{v}</div></div>" for k, v, g in cards) + "</div>"]

    if e:
        h.append("<h2>Final evaluation: 10 evaluator trips (through the API)</h2><div class=wrap><table><tr><th>Trip</th><th>GPS points</th><th>Top-1 accuracy</th><th></th><th>Continuity</th><th>Route F1</th><th>HMM breaks</th><th>Latency (s)</th></tr>")
        for t in e["trips"]:
            h.append(f"<tr><td>{t['trip_id']}</td><td>{t['n_points']:,}</td><td>{pct(t['top1_accuracy'])}</td>"
                     f"<td><div class=bar><i style='width:{100 * t['top1_accuracy']:.0f}%'></i></div></td><td>{pct(t['continuity'])}</td>"
                     f"<td>{pct(t['route_f1'])}</td><td>{t['n_breaks']}</td><td>{num(t['total_s'], 2)}</td></tr>")
        h.append(f"<tr class=best><td>All trips</td><td>{sum(t['n_points'] for t in e['trips']):,}</td><td>{pct(e['pooled_top1'])}</td><td></td>"
                 f"<td>{pct(e['pooled_continuity'])}</td><td>{pct(e['mean_route_f1'])}</td><td></td><td>{num(e['mean_latency_s'], 2)}</td></tr></table></div>"
                 "<p class=note>Top-1 = share of GPS observations whose predicted arc equals the aligned reference arc. The evaluator partition was used only for this final run.</p>")

    h.append("<h2>Model comparison (train: GroupKFold(5) by trip · student-test: development comparison)</h2><div class=wrap><table>"
             "<tr><th>Model</th><th>Family</th><th>CV Top-1</th><th>CV macro F1</th><th>CV log loss</th><th>Train Top-1</th><th>ST Top-1 (classifier)</th><th>ST Top-1 (+HMM)</th><th>ST macro F1</th><th>ST log loss</th><th>Fit (s)</th><th>Infer (s/trip)</th></tr>")
    for i, m in enumerate(s["models"]):
        h.append(f"<tr class={'best' if i == 0 else ''}><td>{html.escape(m['label'])}</td><td>{html.escape(str(m.get('family')))}</td>"
                 f"<td>{pct(m.get('cv_top1_mean'))} ± {num(100 * (m.get('cv_top1_std') or 0), 1)}</td><td>{num(m.get('cv_macro_f1_mean'))}</td>"
                 f"<td>{num(m.get('cv_log_loss_mean'))}</td><td>{pct(m.get('train_top1'))}</td><td>{pct(m.get('st_top1'))}</td>"
                 f"<td>{pct(m.get('st_top1_with_hmm'))}</td><td>{num(m.get('st_macro_f1'))}</td><td>{num(m.get('st_log_loss'))}</td>"
                 f"<td>{num(m.get('fit_s'), 1)}</td><td>{num(m.get('infer_s_per_trip'), 3)}</td></tr>")
    h.append("</table></div>")

    if s.get("tuned_vs_untuned"):
        h.append("<h2>Bayesian optimisation (Optuna TPE, objective = grouped-CV Top-1 on train)</h2><div class=wrap><table><tr><th>Model</th><th>CV macro F1</th><th>CV Top-1</th><th>ST Top-1</th><th>ST macro F1</th><th>ST log loss</th><th>Infer (s/trip)</th></tr>")
        for r in s["tuned_vs_untuned"]:
            h.append(f"<tr><td>{r['model']}</td><td>{num(r['cv_macro_f1'])}</td><td>{pct(r['cv_top1'])}</td><td>{pct(r['st_top1'])}</td><td>{num(r['st_macro_f1'])}</td><td>{num(r['st_log_loss'])}</td><td>{num(r['infer_s_per_trip'])}</td></tr>")
        h.append("</table></div>")
        o = s.get("optuna") or {}
        if o:
            h.append(f"<p class=note>{o.get('n_trials')} trials ({o.get('n_complete')} complete, {o.get('n_pruned')} pruned), best CV Top-1 {pct(o.get('best_cv_top1'))}. Best parameters: <code>{html.escape(json.dumps(o.get('best_params')))}</code></p>")

    if s.get("hmm_variants_student_test"):
        h.append("<h2>Route reconstruction: classifier only vs classifier + HMM (student-test)</h2><div class=wrap><table><tr><th>Model</th><th>Variant</th><th>Top-1</th><th>Continuity</th><th>Route F1</th><th>Breaks</th></tr>")
        for r in s["hmm_variants_student_test"]:
            h.append(f"<tr><td>{html.escape(str(r['model']))}</td><td>{r['variant']}</td><td>{pct(r['top1'])}</td><td>{pct(r['continuity'])}</td><td>{pct(r['route_f1'])}</td><td>{int(r['n_breaks'])}</td></tr>")
        h.append("</table></div><p class=note>B0 = nearest arc, B1 = nearest arc + HMM, C = classifier argmax, C+H = classifier probabilities decoded with the HMM.</p>")

    h.append(f"<h2>Data and preprocessing (train)</h2><div class=grid>"
             f"<div class=card><div class=k>Train trips / points</div><div class=v>{ds['train_trips']} / {ds['train_points']:,}</div></div>"
             f"<div class=card><div class=k>UTM zones</div><div class=v>{ds['utm_zones']}</div></div>"
             f"<div class=card><div class=k>Median alignment</div><div class=v>{ds['median_alignment_m']} m</div></div>"
             f"<div class=card><div class=k>Candidates (r, K)</div><div class=v>{int(ds['candidate_radius_m'])} m, {ds['candidate_k']}</div></div>"
             f"<div class=card><div class=k>Candidate miss rate</div><div class=v>{pct(ds['train_candidate_miss_rate_usable_trips'])}</div></div>"
             f"<div class=card><div class=k>Features</div><div class=v>{s['features']['n']}</div></div></div>"
             f"<p class=note>Excluded from fitting (unreliable reference labels): {', '.join(ds['excluded_train_trips'])}. Features: <code>{', '.join(s['features']['names'])}</code></p>")
    h.append("</main></body></html>")
    return "\n".join(h)


def main():
    s = build()
    (R / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    (R / "dashboard.html").write_text(dashboard(s), encoding="utf-8")
    print("wrote", R / "summary.json", "and", R / "dashboard.html")


if __name__ == "__main__":
    main()
