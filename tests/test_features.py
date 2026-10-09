"""Tests for the row-statistics feature engineering.

Uses small hand-built and synthetic frames (no Kaggle download needed).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, features  # noqa: E402


def _row(values: list[int]) -> pd.DataFrame:
    return pd.DataFrame([values], columns=config.FEATURE_COLS)


@pytest.fixture(scope="module")
def synthetic_X() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.integers(0, 11, size=(50, len(config.FEATURE_COLS))), columns=config.FEATURE_COLS
    )


class TestRowStatsFeatures:
    def test_output_columns_are_raw_then_engineered(self, synthetic_X):
        out = features.RowStatsFeatures().fit_transform(synthetic_X)
        assert list(out.columns) == config.FEATURE_COLS + features.ENGINEERED_COLS
        assert list(features.RowStatsFeatures().get_feature_names_out()) == list(out.columns)

    def test_preserves_index_and_row_count(self, synthetic_X):
        X = synthetic_X.set_index(pd.Index(range(100, 150), name="id"))
        out = features.RowStatsFeatures().fit_transform(X)
        assert out.index.equals(X.index)

    def test_known_row_statistics(self):
        values = list(range(20))  # 0..19
        out = features.RowStatsFeatures().fit_transform(_row(values)).iloc[0]
        assert out["row_sum"] == 190
        assert out["row_min"] == 0
        assert out["row_max"] == 19
        assert out["row_range"] == 19
        assert out["row_median"] == 9.5
        assert out["row_n_unique"] == 20
        assert out["row_std"] == pytest.approx(np.std(values))
        assert out["row_skew"] == pytest.approx(0.0)

    def test_sorted_values_are_ascending_and_order_invariant(self, synthetic_X):
        out = features.RowStatsFeatures().fit_transform(synthetic_X)
        sorted_block = out[features.SORTED_COLS].to_numpy()
        assert (np.diff(sorted_block, axis=1) >= 0).all()

        # Shuffling which factor holds which value must not change any
        # engineered feature — the target (a sum) is order-invariant too.
        shuffled = synthetic_X.copy()
        shuffled[config.FEATURE_COLS] = np.random.default_rng(1).permuted(
            synthetic_X.to_numpy(), axis=1
        )
        out_shuffled = features.RowStatsFeatures().fit_transform(shuffled)
        pd.testing.assert_frame_equal(
            out[features.ENGINEERED_COLS], out_shuffled[features.ENGINEERED_COLS]
        )

    def test_constant_row_has_zero_skew_and_kurtosis_not_nan(self):
        out = features.RowStatsFeatures().fit_transform(_row([5] * 20)).iloc[0]
        assert out["row_std"] == 0
        assert out["row_skew"] == 0
        assert out["row_kurtosis"] == 0
        assert out["row_n_unique"] == 1
        assert not out.isna().any()

    def test_ignores_extra_columns_and_column_order(self, synthetic_X):
        reordered = synthetic_X[config.FEATURE_COLS[::-1]].assign(extra=1)
        out = features.RowStatsFeatures().fit_transform(reordered)
        expected = features.RowStatsFeatures().fit_transform(synthetic_X)
        pd.testing.assert_frame_equal(out, expected)

    def test_original_formula_recovered_by_row_sum(self):
        # In the original dataset FloodProbability == 0.005 * row sum exactly;
        # row_sum must be that sum (sanity check of the column the whole
        # project's story rests on).
        X = _row([3] * 10 + [7] * 10)
        out = features.RowStatsFeatures().fit_transform(X).iloc[0]
        assert out["row_sum"] * config.ORIGINAL_TARGET_PER_POINT == pytest.approx(0.5)
