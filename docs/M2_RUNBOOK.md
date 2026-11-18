# Reproduce the PathSnap M2 work

Run these PowerShell commands from the root of `m2-eda-api-ui`. The branch uses the dataset-bearing base `26784d3`. The replay pins the M3 harness and M4 decoder that produced the recorded results. The team later integrated both modules in `main` at `8725bea`, and the live check was repeated against that checkout.

## Python environment

```powershell
uv python install 3.11.17
uv venv --python 3.11.17 .venv
$repoM2 = (Get-Location).Path
$python = Join-Path $repoM2 '.venv/Scripts/python.exe'
uv pip install --python $python -r requirements.txt -e .
```

The recorded runs used Python 3.11.17, NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1, PyArrow 26.0.1, joblib 1.6.0, Shapely 2.2.0 and pyproj 3.7.2. The outputs used Matplotlib 3.11.2, Folium 1.0.0, FastAPI 0.163.2, Streamlit 1.67.0 and streamlit-folium 0.27.4.

## Build features and train the models

Both trainers call M3's `run_experiment()`. The shared harness uses seed 42 and five `GroupKFold` folds keyed by trip ID. It excludes training trips `00000033` and `00000054`, samples whole GPS points at 20% for KNN, and evaluates the final fits on the 10 student-test trips.

```powershell
git fetch origin
git fetch --no-tags origin 11bb601ca9fa1cbc77371bca616b57012c376af3 bf22596ec1bc7cbc42b784e16cb1ae5d644eb72b
$parent = Split-Path $repoM2 -Parent
$m3 = Join-Path $parent 'pathsnap-replay-m3'
git worktree add --detach --no-checkout $m3 11bb601
git -C $m3 sparse-checkout set --no-cone '/*' '!/data/'
git -C $m3 read-tree -mu HEAD

$env:PATHSNAP_DATA = Join-Path $repoM2 'data'
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $m3 'src'
$env:OMP_NUM_THREADS = '8'
$env:OPENBLAS_NUM_THREADS = '2'
Push-Location $m3
try {
    & $python scripts/build_dataset.py
    if ($LASTEXITCODE) { throw 'Feature generation failed' }
    & $python (Join-Path $repoM2 'scripts/models/train_knn.py')
    if ($LASTEXITCODE) { throw 'KNN training failed' }
    & $python (Join-Path $repoM2 'scripts/models/train_dtree.py')
    if ($LASTEXITCODE) { throw 'Decision tree training failed' }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
```

This writes local feature tables under the M3 worktree's `results/features/` and regenerates both model folders under its `results/models/`. The checked-in M2 outputs include fit configuration, five CV folds, metrics, confusion matrices, probability Parquets and model files. [KNN results](../results/models/knn/) · [Decision tree results](../results/models/dtree/)

- KNN mean CV point Top-1 was **87.44%** and mean CV candidate macro-F1 was **0.9234**. Student-test point Top-1 was **87.66%**, candidate macro-F1 was **0.9205**, and log loss was **0.1591**.
- Decision tree mean CV point Top-1 was **88.73%** and mean CV candidate macro-F1 was **0.9200**. Student-test point Top-1 was **88.82%**, candidate macro-F1 was **0.9124**, and log loss was **0.1344**.

## Execute the EDA notebook

```powershell
$env:PATHSNAP_EDA_FEATURES = Join-Path $m3 'results/features'
& $python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=240 `
    --output-dir $env:PATHSNAP_EDA_FEATURES --output 01_eda.executed.ipynb notebooks/01_eda.ipynb
& $python scripts/make_maps.py --train-only
```

The notebook writes its plots and tables to `results/eda/` and leaves M1's original sweep and alignment tables in place. The notebook source and extracted figures are versioned. The executed notebook is a local output. [EDA notebook](../notebooks/01_eda.ipynb) · [Data quality report](../results/eda/data_quality.md)

## Prepare the student-test service artifacts

The live test used the saved decision tree and M4's default HMM configuration from `bf22596`. This checkout layout allows the Python package to use M4's published sources during the replay. [M4 source at the recorded commit](https://github.com/Chunnu-Munnu/ML_Hackathon_TEAM_8/tree/bf22596ec1bc7cbc42b784e16cb1ae5d644eb72b/m4_file)

```powershell
$m4 = Join-Path $parent 'pathsnap-replay-m4'
git worktree add --detach $m4 bf22596
$env:PATHSNAP_M4_SOURCE = Join-Path $m4 'm4_file/src/pathsnap'
$env:PATHSNAP_ARTIFACTS = Join-Path $m3 'm2-smoke-artifacts'
$env:EVAL_DIR = Join-Path $repoM2 'data/student_test'
New-Item -ItemType Directory -Path $env:PATHSNAP_ARTIFACTS -Force | Out-Null
Copy-Item (Join-Path $m3 'results/models/dtree/model.joblib') $env:PATHSNAP_ARTIFACTS

@'
import json
import os
from pathlib import Path

import pathsnap

pathsnap.__path__.append(os.environ["PATHSNAP_M4_SOURCE"])
from pathsnap.hmm import DEFAULT_HMM_CFG

artifact_dir = Path(os.environ["PATHSNAP_ARTIFACTS"])
(artifact_dir / "hmm_config.json").write_text(json.dumps(DEFAULT_HMM_CFG), encoding="utf-8")
'@ | & $python -
```

Generate the two student-test prediction maps and the training map from these artifacts. A local output directory keeps the recorded maps available for comparison.

```powershell
$env:PATHSNAP_REPLAY_MAPS = Join-Path $m3 'm2-smoke-maps'
@'
import os
import runpy
import sys
from pathlib import Path

import pathsnap

pathsnap.__path__.append(os.environ["PATHSNAP_M4_SOURCE"])
artifacts = Path(os.environ["PATHSNAP_ARTIFACTS"])
sys.argv = [
    "scripts/make_maps.py", "--model", str(artifacts / "model.joblib"),
    "--hmm-config", str(artifacts / "hmm_config.json"),
    "--output-dir", os.environ["PATHSNAP_REPLAY_MAPS"],
]
runpy.run_path(sys.argv[0], run_name="__main__")
'@ | & $python -
```

## Run the API and dashboard

Start the API in the same PowerShell window as the artifact setup.

```powershell
@'
import os

import pathsnap
import uvicorn

pathsnap.__path__.append(os.environ["PATHSNAP_M4_SOURCE"])
uvicorn.run("api.main:app", host="127.0.0.1", port=8000)
'@ | & $python -
```

Open a second PowerShell window at the repository root for the dashboard.

```powershell
$env:PATHSNAP_API = 'http://127.0.0.1:8000'
.venv/Scripts/python.exe -m streamlit run ui/app.py --server.port 8501
```

In a third PowerShell window run the automated checks.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_m2_api.py'
.venv/Scripts/python.exe tests/m2_live_smoke.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501 --output results/features/replayed_api_smoke.csv
```

The live check exercised all 19,160 observations across 10 student-test trips. The decision tree with the default HMM produced **89.65% point-weighted Top-1**. Trip continuity ranged from **94.37% to 100%**, and pipeline inference took **0.353 to 3.880 s per trip** on the test machine. Streamlit's app test selected a trip, called the running FastAPI service over HTTP, and checked the Folium component, metric cards and timing table. `n_breaks` counts decoder restarts after infeasible transitions. Continuity measures direct connectivity of adjacent matched observations. [Per-trip API evidence](../results/m2_api_smoke.csv) · [Offline maps](../results/maps/)

## Team integration

Team commit `8725bea` contains M3's harness and M4's decoder and scoring modules in their normal package paths. The M2 API and Streamlit replay with those modules reproduced all recorded per-trip scores and counts. After preparing the student-test artifacts above, this command starts the same combination with standard imports from the integrated checkout.

```powershell
$team = Join-Path $parent 'pathsnap-replay-team'
git worktree add --detach --no-checkout $team 8725bea
git -C $team sparse-checkout set --no-cone '/*' '!/data/'
git -C $team read-tree -mu HEAD
$env:PYTHONPATH = Join-Path $team 'src'
& $python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The remaining handoff is to bring the final M2 changes into the team's integration history and publish the selected deployment artifacts. Place the team's selected model and decoder configuration at `artifacts/model.joblib` and `artifacts/hmm_config.json`. With the deployment trip directory in `EVAL_DIR`, run `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000` from the integrated checkout. Set `PATHSNAP_API` on the Streamlit laptop to that server's URL. During inference the API reads GPS and network inputs. The scoring step reads the optional reference route afterward and publishes aggregate metrics.
