"""Project-wide paths and constants. Import from here instead of hardcoding."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
# Evaluator trips: only the API reads this, only on demo day.
EVAL_DIR = Path(os.environ.get("EVAL_DIR", STUDENT_TEST_DIR))

RESULTS_DIR = ROOT / "results"
FEATURES_DIR = RESULTS_DIR / "features"
ARTIFACTS_DIR = ROOT / "artifacts"

SEED = 42

# Candidate generation (chosen from the train-only miss-rate sweep, see results/eda/candidate_miss_rate.csv)
CAND_RADIUS_M = 50.0
CAND_K = 8

# Cleaning thresholds
JUMP_SPEED_MS = 70.0      # steps faster than this (~250 km/h) are flagged as GPS jumps
GAP_S = 5.0               # dt above this is flagged as a gap
STATIONARY_MS = 1.0       # slower than this: heading is unreliable

# Train trips whose median GPS-to-route alignment distance exceeds this are treated as
# unreliable labels and excluded from model fitting (decided on train data only).
BAD_LABEL_MEDIAN_M = 50.0
