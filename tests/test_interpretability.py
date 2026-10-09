"""Tests for SHAP interpretability helpers (synthetic data, small LightGBM)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, features, interpretability, model  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.integers(0, 11, size=(300, len(config.FEATURE_COLS))), columns=config.FEATURE_COLS
    )
    y = config.ORIGINAL_TARGET_PER_POINT * X.sum(axis=1)
    pipeline = model.train_pipeline(X, y, LGBMRegressor(n_estimators=50, verbosity=-1))
    return pipeline, X


def test_shap_values_have_one_column_per_engineered_feature(fitted):
    pipeline, X = fitted
    explanation, X_transformed = interpretability.compute_shap_values(pipeline, X, max_samples=40)
    assert explanation.values.shape == (40, len(config.FEATURE_COLS) + len(features.ENGINEERED_COLS))
    assert list(X_transformed.columns) == list(explanation.feature_names)


def test_shap_values_add_up_to_the_prediction(fitted):
    pipeline, X = fitted
    explanation, _ = interpretability.compute_shap_values(pipeline, X.iloc[:10])
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, pipeline.predict(X.iloc[:10]), atol=1e-6)


def test_top_features_are_ranked(fitted):
    pipeline, X = fitted
    explanation, _ = interpretability.compute_shap_values(pipeline, X, max_samples=100)
    top = interpretability.top_shap_features(explanation, top_n=5)
    assert len(top) == 5
    assert top["mean_abs_shap"].is_monotonic_decreasing
