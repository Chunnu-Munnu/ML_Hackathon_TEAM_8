# PathSnap — Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon (PES University). PathSnap matches noisy 1 Hz GPS tracks to a road network:
candidate road segments are generated per GPS point, a classifier scores each (point, candidate) pair, and an
HMM/Viterbi decoder turns those scores into a connected route. A FastAPI backend (Laptop 1) serves predictions to a
Streamlit UI (Laptop 2).

> Work in progress.

## Setup

```bash
git clone https://github.com/Chunnu-Munnu/ML_HACKATHON.git
cd ML_HACKATHON
py -3.11 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                  # makes `import pathsnap` work everywhere
```

### Data

The raw data is not committed. Copy the provided folders so the layout is:

```
data/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}         # 80 trips
└── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}  # 10 trips
```

(Set `PATHSNAP_DATA` to use a different location.) Evaluator trips are never used during development; on demo day
the API reads them from the folder given in `EVAL_DIR`.

## Pipeline (run in order)

| Step | Command | Output |
|---|---|---|
| 1. Build features + labels | `python scripts/build_dataset.py` | `results/features/*.parquet`, `results/eda/candidate_miss_rate.csv`, `results/eda/label_alignment.csv` |
| 2. Train the 8 models | `python scripts/models/train_<name>.py` | `results/models/<name>/` |
| 3. Compare models | `python scripts/compare_models.py` | `results/models/model_comparison.csv`, plots |
| 4. Bayesian HPO | `python scripts/tune_optuna.py` | `results/optuna/`, `artifacts/model.joblib` |
| 5. HMM evaluation | `python scripts/eval_hmm.py` | `results/hmm/` |
| 6. Decoder tuning | `python scripts/tune_decoder.py` | `artifacts/hmm_config.json` |
| 7. Maps | `python scripts/make_maps.py` | `results/maps/*.html` |

## Deployment (two laptops)

Laptop 1 (model + API):
```bash
set EVAL_DIR=path\to\evaluator_trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Laptop 2 (UI):
```bash
set PATHSNAP_API=http://<laptop1-ip>:8000
streamlit run ui/app.py
```

## Repository layout

```
src/pathsnap/   core library (io, geo, network, candidates, labeling, features, metrics, experiment, hmm, pipeline, viz)
scripts/        dataset build, model training (scripts/models/), comparison, tuning, HMM eval, maps
notebooks/      EDA
api/  ui/       FastAPI backend and Streamlit evaluator UI
artifacts/      exported model + configs used by the API
results/        all CSVs, plots, logs and maps backing the report
docs/           project documentation
```
