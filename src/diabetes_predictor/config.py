"""Single source of truth for paths, the random seed and feature definitions."""

from pathlib import Path

# --- Paths (resolved from the repo root, so scripts work from any working directory) ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "diabetes.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# --- Reproducibility ---
SEED = 42

# --- Experiment settings ---
TEST_SIZE = 0.20  # stratified hold-out, touched once in evaluate.py
N_SPLITS = 5  # StratifiedKFold folds used for every CV step on the train set
N_ITER_SEARCH = 40  # RandomizedSearchCV candidates per tuned model
N_MODELS_TO_TUNE = 2
TARGET_RECALL = 0.80  # screening requirement: catch at least 80% of diabetics
N_BOOTSTRAP = 1000  # resamples for test-set confidence intervals

# --- Columns ---
TARGET = "Diabetic"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
# A value of 0 is physiologically impossible for these measurements: it encodes "not measured".
# (Pregnancies = 0 is a real value and is deliberately NOT in this list.)
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
