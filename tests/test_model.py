"""Tests for model construction, training, blending, and persistence.

Uses a synthetic regression frame built the way the original dataset was
(target = 0.005 x row sum, plus noise) — no Kaggle download needed.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, model  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        rng.integers(0, 11, size=(n, len(config.FEATURE_COLS))), columns=config.FEATURE_COLS
    )
    y = pd.Series(
        config.ORIGINAL_TARGET_PER_POINT * X.sum(axis=1) + rng.normal(0, 0.01, size=n),
        name=config.TARGET_COL,
    )
    return X, y


@pytest.fixture(autouse=True)
def fast_estimators(monkeypatch):
    """Shrink the boosted models so the suite runs in seconds; the real
    hyperparameters are exercised by scripts/train.py on real data."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    monkeypatch.setitem(model.MODEL_FACTORIES, "LightGBM", lambda: LGBMRegressor(n_estimators=20, verbosity=-1))
    monkeypatch.setitem(model.MODEL_FACTORIES, "XGBoost", lambda: XGBRegressor(n_estimators=20))
    monkeypatch.setitem(
        model.MODEL_FACTORIES, "CatBoost",
        lambda: CatBoostRegressor(iterations=20, verbose=False, allow_writing_files=False),
    )


class TestModelFactories:
    @pytest.mark.parametrize("name", list(model.MODEL_FACTORIES.keys()))
    def test_each_factory_trains_and_predicts(self, synthetic_data, name):
        X, y = synthetic_data
        pipeline = model.train_pipeline(X, y, estimator=model.MODEL_FACTORIES[name]())
        predictions = pipeline.predict(X.iloc[:5])
        assert predictions.shape == (5,)
        assert np.isfinite(predictions).all()

    def test_factories_return_fresh_instances(self):
        assert model.lightgbm_estimator() is not model.lightgbm_estimator()

    def test_default_pipeline_is_lightgbm(self):
        from lightgbm import LGBMRegressor

        assert isinstance(model.build_pipeline().named_steps["model"], LGBMRegressor)

    def test_linear_model_learns_the_sum(self, synthetic_data):
        X, y = synthetic_data
        scores = model.cross_validate_pipeline(X, y, estimator=model.linear_estimator(), cv=3)
        assert scores["r2_mean"] > 0.9
        assert set(scores) == {"r2_mean", "r2_std", "rmse_mean", "rmse_std"}


class TestEnsemble:
    def test_out_of_fold_predictions_cover_every_row(self, synthetic_data):
        X, y = synthetic_data
        X_test = X.iloc[:7]
        oof, test = model.out_of_fold_predictions(
            X, y, X_test, members=["LightGBM", "Linear Regression"], n_splits=3, log=lambda _: None
        )
        assert list(oof) == ["LightGBM", "Linear Regression"]
        for name in oof:
            assert oof[name].shape == (len(X),)
            assert (oof[name] != 0).all()  # every row got an out-of-fold prediction
            assert test[name].shape == (7,)

    def test_blend_weights_are_non_negative_and_sum_to_one(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=500)
        oof = {
            "good": y + rng.normal(0, 0.1, size=500),
            "noisy": y + rng.normal(0, 1.0, size=500),
            "anti": -y,
        }
        weights = model.fit_blend_weights(oof, y)
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0).all()
        assert weights[0] > weights[1]  # the accurate member dominates
        assert weights[2] == 0  # an anti-correlated member is dropped, never subtracted

    def test_blend_is_weighted_average(self):
        predictions = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        np.testing.assert_allclose(model.blend(predictions, np.array([0.25, 0.75])), [2.5, 3.5])


class TestPersistence:
    def test_save_and_load_round_trip(self, synthetic_data):
        X, y = synthetic_data
        pipeline = model.train_pipeline(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            model.save_pipeline(pipeline, path)
            loaded = model.load_pipeline(path)
        np.testing.assert_allclose(loaded.predict(X.iloc[:5]), pipeline.predict(X.iloc[:5]))

    def test_load_missing_model_explains_how_to_train(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="scripts/train.py"):
            model.load_pipeline(tmp_path / "missing.joblib")


class TestWithSeed:
    def test_sets_the_right_seed_parameter_per_library(self):
        from catboost import CatBoostRegressor
        from lightgbm import LGBMRegressor

        assert model.with_seed(LGBMRegressor(), 7).get_params()["random_state"] == 7
        assert model.with_seed(CatBoostRegressor(), 7).get_params()["random_seed"] == 7

    def test_seedless_estimator_is_unchanged(self):
        from sklearn.linear_model import LinearRegression

        assert model.with_seed(LinearRegression(), 7).get_params() == LinearRegression().get_params()

    def test_seed_bagging_averages_into_one_prediction_per_row(self, synthetic_data):
        X, y = synthetic_data
        oof, test = model.out_of_fold_predictions(
            X, y, X.iloc[:4], members=["XGBoost"], n_splits=3, n_seeds=2, log=lambda _: None
        )
        assert oof["XGBoost"].shape == (len(X),)
        assert test["XGBoost"].shape == (4,)
