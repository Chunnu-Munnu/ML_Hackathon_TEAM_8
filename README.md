# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University · Team 8

PathSnap matches noisy 1 Hz GPS tracks to the road network. For every GPS observation it generates nearby road
segments (candidates), a LightGBM classifier scores each (observation, candidate) pair, and an HMM/Viterbi decoder
turns those scores plus road connectivity into one connected route. A FastAPI service (Laptop 1) runs the full
pipeline and scores it server-side; a Streamlit UI (Laptop 2) lets the evaluator pick a trip and view the result.

**Final results:** see [`results/dashboard.html`](results/dashboard.html) (static page) or the **Results** tab in the UI.

| Final result (deployed: Random Forest + HMM/Viterbi) | Top-1 accuracy | Route continuity |
|---|---|---|
| **Evaluator test, 10 trips, through the API** | **94.1%** pooled · 95.0% mean per trip | 96.2% |
| Student test, 10 trips, through the API | 92.8% pooled · 93.9% mean per trip | 97.2% |
| Train, GroupKFold(5) CV, best classifier alone (tuned LightGBM) | 90.4% | — |
| Nearest-arc baseline (no ML), train CV | 54.8% | — |


---

## 1. Setup

```bash
git clone https://github.com/Chunnu-Munnu/ML_Hackathon_TEAM_8.git
cd ML_Hackathon_TEAM_8
py -3.11 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Data
```
data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          80 trips (committed)
data/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   10 trips (committed)
data/evaluator_test/<trip_id>/...                                10 trips: place on the API laptop only (gitignored)
```
`track` = lon, lat, t · `nodes` = lon, lat (row = node id) · `arcs` = from, to (row = arc id) ·
`route` = reference arc ids in driving order (no timestamps).
The evaluator partition is read **only** by the API (`EVAL_DIR`, default `data/evaluator_test` if present, else
`data/student_test`). No training, tuning or analysis script reads it.

## 2. Reproduce everything (in order)

| # | Command | Output |
|---|---|---|
| 1 | `python scripts/build_dataset.py` | `results/features/*.parquet` (local), `results/eda/label_alignment.csv`, `candidate_miss_rate.csv`, `excluded_train_trips.json` |
| 2 | `jupyter nbconvert --execute notebooks/01_eda.ipynb` | EDA figures/tables in `results/eda/`, Folium map |
| 3 | `python scripts/models/baseline_nearest.py` then `train_logreg.py`, `train_gnb.py`, `train_knn.py`, `train_dtree.py`, `train_lgbm.py`, `train_rf.py`, `train_mlp.py`, `train_svm.py` | `results/models/<name>/` (config, CV folds, metrics, confusion matrix, probabilities, model) + `model_comparison.csv` |
| 4 | `python scripts/compare_models.py` | comparison plots + `selection.md` |
| 5 | `python scripts/tune_optuna.py --trials 40 --subsample 0.5` | `results/optuna/` (trials, best params, plots, tuned vs untuned) + `artifacts/model.joblib` |
| 6 | `python scripts/eval_hmm.py` | `results/hmm/hmm_comparison.csv` (classifier only vs + HMM, all models) |
| 7 | `python scripts/tune_decoder.py --model lgbm_tuned` | `results/hmm/decoder_search.csv`, `artifacts/hmm_config.json` |
| 8 | `python scripts/export_artifacts.py` | `artifacts/feature_list.json`, `metadata.json` |
| 9 | `python scripts/make_maps.py` | Folium maps in `results/maps/` |
| 10 | start the API, then `python scripts/run_evaluator.py --out results/evaluator/evaluator_results.csv` | final per-trip metrics through the API |
| 11 | `python scripts/build_summary.py` | `results/summary.json` (served at `GET /metrics`) + `results/dashboard.html` |

Tests: `python -m unittest tests.test_api`

## 3. Method

**Preprocessing**
- Per-trip UTM projection (trips span 25 UTM zones).
- GPS flags (duplicate, jump > 70 m/s, gap > 5 s): points are kept, never deleted.
- Canonical ids for duplicate arcs; self-loops and zero-length arcs are never candidates.
- Each arc is linked to its reverse twin.
- STRtree spatial index and directed graph with cached Dijkstra.

**Labels**
- The route has no timestamps, so a monotone dynamic programme assigns each GPS point a route arc. It only moves forward along the route and minimises total distance.
- Typical alignment error is about 5 m.
- Train trips with median alignment > 50 m (00000033, 00000054) are excluded from fitting.

**Candidates**
- All arcs within 50 m, at most 8 per point.
- Train sweep: about 98% of points keep their reference arc (r and K chosen on train only).

**Features (45)**
- Geometry: distance, rank, z-score, projection, overshoot, arc length.
- Direction: heading over ±2/±5/±10 points vs arc bearing, reverse-twin comparison.
- Heading reliability: heading consistency, path straightness.
- Signed motion along the arc.
- Kinematics.
- Temporal context: neighbours' distance to the same arc, connectivity to the previous point's candidates.
- Network degree and candidate-set density.
- No lon/lat or trip id.

**Models (8 families + baseline)**
- Nearest arc (baseline), Logistic Regression, Gaussian NB, KNN, Decision Tree, LightGBM, Random Forest, MLP, Linear SVM.
- All trained through one harness: GroupKFold(5) by trip on train, refit, development comparison on student-test, seed 42.
- Primary metric: observation-level Top-1, with log loss as tiebreak.

**Bayesian optimisation**
- Optuna TPE on LightGBM.
- Objective: grouped-CV Top-1 on train only.

**HMM / Viterbi**
- States: candidate arcs.
- Emission: λ·log p(classifier).
- Transition: −|d_route − d_gps| / β, with driving distance from the directed graph.
- Backward moves along the same/preceding arc (1 Hz GPS jitter) are scored as negative distance instead of a detour, so the chain doesn't break while a wrong-direction run is still penalised every step.
- Gaps over gap_s split the chain.
- Points without candidates take the nearest decoded arc.
- Decoder parameters tuned on train/student-test only.

## 4. Two-laptop deployment

**Laptop 1 (model + API + reference data)**
```bash
git pull
.venv\Scripts\activate
REM put the evaluator trips in data\evaluator_test (or set EVAL_DIR=...)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Find the IP with `ipconfig` and allow Python through the Windows firewall (private network). Keep this terminal
visible: every request is logged here and in `logs/api.log`.

**Laptop 2 (UI)**
```bash
.venv\Scripts\activate
set PATHSNAP_API=http://<laptop1-ip>:8000
streamlit run ui/app.py
```
The sidebar shows *API ready* and the trip set. Choose a trip and press **Reconstruct route**. The **Results** tab
shows every final metric. The reference route never leaves Laptop 1; the UI only receives predicted geometry and
server-computed scores.

API: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`, `GET /metrics`.

## 5. Repository layout
```
src/pathsnap/   io, geo, network, candidates, labeling, features, metrics, experiment, hmm, route_metrics, pipeline, viz
scripts/        build_dataset, models/train_*.py, compare_models, tune_optuna, eval_hmm, tune_decoder,
                export_artifacts, make_maps, run_evaluator, build_summary
notebooks/      01_eda.ipynb
api/ ui/        FastAPI backend · Streamlit UI
artifacts/      deployed model, feature list, decoder config, metadata
results/        eda/, models/, optuna/, hmm/, maps/, evaluator/, summary.json, dashboard.html
tests/          API contract tests and live smoke test
```

## 6. Leakage and integrity
- No trip is split across partitions; CV folds are grouped by trip.
- The candidate radius, exclusion rule, features, models, hyperparameters and decoder parameters were chosen on train (and student-test for development comparison / decoder tuning). The evaluator partition was used once, for the final run through the API.
- `predict_trip` drops any reference route before predicting; the API loads the route separately, after reconstruction, only to compute accuracy, and never returns it.
- No hardcoded predictions or trip-specific logic.
