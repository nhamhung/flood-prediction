"""Cached data/model loaders shared across every page.

Centralized here (rather than re-loaded per page) so the training data and
trained pipeline are each read from disk exactly once per session, and every
page sees the same defaults.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from flood_prediction import config, data, features, interpretability, model  # noqa: E402

PALETTE = ["#2c5cc5", "#5b8def", "#8fb4f2", "#e07a5f", "#81b29a"]

# The 20 factors grouped by theme for the calculator form. Every factor is
# scored so that higher = more flood risk.
FACTOR_GROUPS = {
    "🌧️ Climate & terrain": [
        "MonsoonIntensity", "ClimateChange", "TopographyDrainage",
        "CoastalVulnerability", "Landslides", "Watersheds",
    ],
    "🏗️ Water infrastructure": [
        "RiverManagement", "DamsQuality", "DrainageSystems",
        "DeterioratingInfrastructure", "Siltation",
    ],
    "🌱 Land use": [
        "Deforestation", "Urbanization", "AgriculturalPractices",
        "Encroachments", "WetlandLoss",
    ],
    "🏛️ Governance & people": [
        "IneffectiveDisasterPreparedness", "InadequatePlanning",
        "PoliticalFactors", "PopulationScore",
    ],
}
assert sorted(sum(FACTOR_GROUPS.values(), [])) == sorted(config.FEATURE_COLS)


def factor_label(col: str) -> str:
    """'IneffectiveDisasterPreparedness' -> 'Ineffective disaster preparedness'."""
    words = "".join(f" {c}" if c.isupper() else c for c in col).strip().split(" ")
    return " ".join([words[0]] + [w.lower() for w in words[1:]])


def _configure_kaggle_credentials() -> None:
    """Wire Kaggle API credentials from Streamlit secrets into the
    environment variables the `kaggle` package reads, so a deployment
    without a pre-baked Docker image (e.g. Streamlit Community Cloud)
    can fetch the competition data automatically when
    USE_FULL_KAGGLE_DATA is set — see `data._download_from_kaggle`. A no-op
    if real environment variables are already set (e.g. running locally) or
    no secret is configured (the bundled sample is used instead).
    """
    try:
        if st.secrets.get("USE_FULL_KAGGLE_DATA"):
            os.environ["USE_FULL_KAGGLE_DATA"] = "true"
    except Exception:
        pass
    if os.environ.get("KAGGLE_API_TOKEN") or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        return
    try:
        token = st.secrets.get("KAGGLE_API_TOKEN")
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            return
    except Exception:
        pass
    try:
        os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
        os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]
    except Exception:
        pass


_configure_kaggle_credentials()


@st.cache_resource
def get_pipeline():
    return model.load_pipeline()


@st.cache_data
def get_train_df() -> pd.DataFrame:
    return data.load_train()


@st.cache_data
def get_original_df() -> pd.DataFrame | None:
    """The 50k-row original dataset (downloaded or bundled), or None."""
    return data.load_original()


@st.cache_data
def get_default_row() -> dict:
    """A "typical" area: every factor at its training-set median."""
    medians = get_train_df()[config.FEATURE_COLS].median()
    return {col: int(round(medians[col])) for col in config.FEATURE_COLS}


@st.cache_data
def get_engineered_sample(n: int = 50_000) -> pd.DataFrame:
    """A random sample of training rows with engineered features and the
    target attached, for the EDA pages (the full 1.1M rows would make every
    scatter plot slow without changing what it shows)."""
    train_df = get_train_df()
    train_df = train_df.sample(min(n, len(train_df)), random_state=config.RANDOM_SEED)
    engineered = features.RowStatsFeatures().fit_transform(train_df[config.FEATURE_COLS])
    engineered[config.TARGET_COL] = train_df[config.TARGET_COL].values
    return engineered


@st.cache_data(show_spinner="Computing SHAP values (first load only)...")
def get_shap_explanation(sample_size: int = 500):
    pipeline = get_pipeline()
    X = get_train_df()[config.FEATURE_COLS]
    return interpretability.compute_shap_values(pipeline, X, max_samples=sample_size)


@st.cache_resource
def get_explainer():
    import shap

    return shap.TreeExplainer(get_pipeline().named_steps["model"])


@st.cache_data
def get_ensemble_predictions() -> dict | None:
    """Saved leaderboard-blend predictions, or None if
    scripts/make_ensemble_submission.py hasn't been run."""
    if not config.ENSEMBLE_PREDICTIONS_PATH.exists():
        return None
    with np.load(config.ENSEMBLE_PREDICTIONS_PATH) as saved:
        return {key: saved[key] for key in saved.files}
