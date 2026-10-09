"""Tests for the leaderboard stack and the rich feature set (synthetic data)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, features, stacking  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_X() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.integers(0, 16, size=(40, len(config.FEATURE_COLS))), columns=config.FEATURE_COLS)


class TestRichFeatures:
    def test_extends_standard_features(self, synthetic_X):
        rich = features.RichRowStatsFeatures().fit_transform(synthetic_X)
        standard = features.RowStatsFeatures().fit_transform(synthetic_X)
        assert list(rich.columns) == list(standard.columns) + features.RICH_EXTRA_COLS
        pd.testing.assert_frame_equal(rich[standard.columns], standard)

    def test_value_counts_sum_to_number_of_factors_below_16(self, synthetic_X):
        rich = features.RichRowStatsFeatures().fit_transform(synthetic_X)
        counts = rich[[f"count_{v}" for v in features.VALUE_COUNT_RANGE]].sum(axis=1)
        assert (counts == len(config.FEATURE_COLS)).all()

    def test_all_zero_row_has_zero_cv_not_nan(self):
        row = pd.DataFrame([[0] * len(config.FEATURE_COLS)], columns=config.FEATURE_COLS)
        out = features.RichRowStatsFeatures().fit_transform(row).iloc[0]
        assert out["row_cv"] == 0 and np.isfinite(out.to_numpy(dtype=float)).all()


class TestStack:
    def test_weights_are_non_negative_and_favour_the_accurate_member(self):
        rng = np.random.default_rng(0)
        y = pd.Series(rng.normal(size=2000))
        oof = {"good": y + rng.normal(0, 0.1, 2000), "noisy": y + rng.normal(0, 1.0, 2000), "anti": -y}
        stack = stacking.fit_stack(oof, y)
        assert (stack.weights >= 0).all()
        assert stack.weights[0] > stack.weights[1]
        assert stack.weights[2] == pytest.approx(0, abs=1e-9)

    def test_stack_score_is_cross_validated(self):
        rng = np.random.default_rng(1)
        y = pd.Series(rng.normal(size=500))
        oof = {"noise_a": rng.normal(size=500), "noise_b": rng.normal(size=500)}
        # Pure-noise members: an in-sample fit would report R^2 >= 0; held-out scoring can't.
        assert stacking.fit_stack(oof, y).oof_r2 <= 0.01

    def test_predict_applies_weights_and_intercept(self):
        stack = stacking.Stack(members=["a", "b"], weights=np.array([0.25, 0.75]), intercept=0.1,
                               oof_r2=0.0, member_oof_r2={})
        np.testing.assert_allclose(stacking.predict_stack(stack, {"a": np.array([1.0]), "b": np.array([3.0])}), [2.6])


def test_member_slugs_are_unique_and_pipelines_build():
    assert len(stacking.MEMBERS) == len({m.slug for m in stacking.MEMBERS.values()})
    for member in stacking.MEMBERS.values():
        pipeline = stacking.build_member_pipeline(member)
        assert pipeline.named_steps["engineer"] is not None
