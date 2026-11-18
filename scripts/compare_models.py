"""Model comparison plots + selection. Reads results/models/*/metrics.json (actual runs only).

Selection rule (decided before seeing results): highest student-test Top-1 after HMM decoding
(classifier-only Top-1 until results/hmm/hmm_comparison.csv exists); tiebreaker lower student-test log loss;
CV std and inference time reported alongside. Never training accuracy.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config

MD = config.RESULTS_DIR / "models"
HMM_CSV = config.RESULTS_DIR / "hmm" / "hmm_comparison.csv"


def load() -> pd.DataFrame:
    recs = []
    for f in sorted(MD.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        c = json.loads((f.parent / "config.json").read_text())
        recs.append({"name": f.parent.name, "family": c["family"], **{k: v for k, v in m.items() if k != "st_confusion"}})
    return pd.DataFrame(recs).sort_values("st_top1", ascending=False).reset_index(drop=True)


def md_table(d: pd.DataFrame) -> str:
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, *body])


def bar(ax, names, vals, err=None, color="tab:blue"):
    ax.bar(names, vals, yerr=err, capsize=3, color=color)
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")


def main():
    df = load()
    if df.empty:
        raise SystemExit("no results/models/*/metrics.json yet")
    n = df["name"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar(ax, n, df["cv_macro_f1_mean"], df["cv_macro_f1_std"])
    ax.set_ylabel("CV macro F1 (GroupKFold 5, train)"); ax.set_title("Cross-validation macro F1 (mean ± std)")
    lo = np.nanmin(df["cv_macro_f1_mean"] - df["cv_macro_f1_std"].fillna(0)); ax.set_ylim(max(0, lo - 0.05), 1)
    fig.tight_layout(); fig.savefig(MD / "cv_macro_f1.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(len(df)); w = 0.38
    ax.bar(x - w / 2, df["st_top1"], w, label="Top-1 accuracy (points)")
    ax.bar(x + w / 2, df["st_macro_f1"], w, label="Macro F1 (rows, thr 0.5)")
    ax.set_xticks(x, n, rotation=45, ha="right"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Student-test: Top-1 accuracy and macro F1"); fig.tight_layout()
    fig.savefig(MD / "st_top1_macro_f1.png", dpi=120); plt.close(fig)

    ll = df.dropna(subset=["st_log_loss"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar(ax, ll["name"], ll["st_log_loss"], color="tab:red")
    ax.set_ylabel("log loss (lower is better)"); ax.set_title("Student-test log loss (models with probabilities)")
    fig.tight_layout(); fig.savefig(MD / "log_loss.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar(ax, n, df["infer_s_per_trip"] * 1000, color="tab:green")
    ax.set_yscale("log"); ax.set_ylabel("ms per trip (predict_proba only, log scale)")
    ax.set_title("Inference time on student-test"); fig.tight_layout()
    fig.savefig(MD / "inference_time.png", dpi=120); plt.close(fig)

    df["st_top1_hmm"] = np.nan
    if HMM_CSV.exists():
        h = pd.read_csv(HMM_CSV, dtype={"trip_id": str})
        h = h[(h["trip_id"] == "ALL")]
        piv = h.pivot_table(index="model", columns="variant", values="top1")
        if "C+H" in piv:
            df["st_top1_hmm"] = df["name"].map(piv["C+H"])
            p = piv.dropna(subset=[c for c in ["C", "C+H"] if c in piv]).sort_values("C+H", ascending=False)
            fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(len(p))
            ax.bar(x - w / 2, p.get("C", np.nan), w, label="classifier only")
            ax.bar(x + w / 2, p["C+H"], w, label="classifier + HMM/Viterbi")
            ax.set_xticks(x, p.index, rotation=45, ha="right"); ax.set_ylim(0, 1); ax.legend()
            ax.set_title("Student-test Top-1 before vs after HMM"); fig.tight_layout()
            fig.savefig(MD / "top1_before_after_hmm.png", dpi=120); plt.close(fig)

    cand = df[df["family"] != "Baseline (no ML)"].copy()
    key = "st_top1_hmm" if cand["st_top1_hmm"].notna().any() else "st_top1"
    cand = cand.sort_values([key, "st_log_loss"], ascending=[False, True], na_position="last")
    best = cand.iloc[0]
    cols = ["name", "family", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "train_top1",
            "st_top1", "st_top1_hmm", "st_macro_f1", "st_log_loss", "infer_s_per_trip"]
    md = [f"# Model selection\n",
          f"Rule (fixed before results): highest student-test Top-1 "
          f"{'after HMM' if key == 'st_top1_hmm' else '(classifier only; HMM results not available yet)'}; "
          f"tiebreak lower student-test log loss; CV std and inference time also considered. "
          f"Training accuracy is not used.\n",
          f"**Selected: `{best['name']}`** ({best['family']}): {key} = {best[key]:.4f}, "
          f"CV Top-1 = {best['cv_top1_mean']:.4f} ± {best['cv_top1_std']:.4f}, "
          f"ST log loss = {best['st_log_loss']}.\n",
          md_table(df[cols].round(4))]
    (MD / "selection.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
