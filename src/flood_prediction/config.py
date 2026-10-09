"""Paths, constants, and column schema for the Flood Prediction dataset.

Dataset: Kaggle Playground Series S4E5, "Regression with a Flood Prediction
Dataset" (https://www.kaggle.com/competitions/playground-series-s4e5),
itself a synthetic version of a 50,000-row original "Flood Prediction"
dataset — mirrored as `flood.csv` in
https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset.

The single most useful fact about this data (verified against `flood.csv`):
in the original, the target is *exactly* 0.005 x the sum of the 20 factor
scores, with zero noise. The competition's synthetic generator perturbed the
features, so the modeling problem is really "recover the original row sum
from noisy features" — which is why row-level statistics (features.py)
matter far more than any individual factor.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_CSV = DATA_RAW_DIR / "train.csv"
TEST_CSV = DATA_RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_RAW_DIR / "sample_submission.csv"
# Bundled with the repository so the app starts without a Kaggle download:
# a 5,000-row random sample of train.csv, and the CC0-licensed original
# dataset (see ORIGINAL_DATASET below). See scripts/build_sample.py.
TRAIN_SAMPLE_CSV = DATA_SAMPLE_DIR / "train_sample.csv.gz"
FULL_TRAIN_ROWS = 1_117_957  # train.csv, verified directly
ORIGINAL_SAMPLE_CSV = DATA_SAMPLE_DIR / "original.csv.gz"
MODEL_PATH = MODELS_DIR / "model.joblib"

# The 50k-row original dataset the competition was generated from (see the
# module docstring) — used for EDA and as optional extra training rows.
ORIGINAL_CSV = DATA_RAW_DIR / "original" / "flood.csv"
ORIGINAL_DATASET = "naiyakhalid/flood-prediction-dataset"

# The original's exact target formula: FloodProbability = 0.005 * row sum.
ORIGINAL_TARGET_PER_POINT = 0.005

# Out-of-fold + test predictions written by scripts/make_ensemble_submission.py,
# read back by the notebook/report/app to show how the leaderboard model
# was assembled without retraining it.
ENSEMBLE_PREDICTIONS_PATH = DATA_PROCESSED_DIR / "ensemble_predictions.npz"

# Frozen final leaderboards (fetched by scripts/fetch_leaderboard.py) used to
# turn a late-submission score into a "would have ranked" position.
LEADERBOARD_DIR = DATA_RAW_DIR / "leaderboard"

# --- Kaggle competition ---------------------------------------------------

KAGGLE_COMPETITION = "playground-series-s4e5"

# --- Target ----------------------------------------------------------------

ID_COL = "id"
TARGET_COL = "FloodProbability"

RANDOM_SEED = 42

# --- Column schema -----------------------------------------------------
# All 20 features are integer "risk factor" scores (verified against the real
# training data: every column is a non-negative integer, mostly 0-10 with a
# thin synthetic tail up to ~18). Higher always means "more flood risk" —
# e.g. a high DrainageSystems score means *worse* drainage, not better.

FEATURE_COLS = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]

RAW_FEATURE_COLS = FEATURE_COLS
