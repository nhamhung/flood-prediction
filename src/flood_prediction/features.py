"""Feature engineering and the shared preprocessing pipeline.

This is the single source of truth for turning raw competition columns into
model-ready features. The notebook, `scripts/train.py`, both submission
scripts, and the Streamlit app all call `build_feature_pipeline()` (wrapped
inside the fitted pipeline saved to `models/model.joblib`) so none of them
can silently diverge from how the model was actually trained.

Why row statistics, not per-column features: in the original dataset the
target is exactly 0.005 x the row sum (see `config.py`). The competition's
synthetic generator perturbed individual factor values, so any single column
is a noisy, weak signal — but statistics computed *across* a row (its sum,
spread, shape, and sorted values) let a model estimate how much each row was
perturbed and recover the underlying sum. Verified on a 20% holdout:
LightGBM on the 20 raw columns alone reaches R^2 0.842; adding these
row statistics lifts it to 0.869.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from . import config

# Names of the derived columns RowStatsFeatures adds, in the order they're
# created. Declared once here so callers (the app's Feature Engineering page,
# tests) can refer to them without re-deriving them.
ROW_STAT_COLS = [
    "row_sum",
    "row_std",
    "row_max",
    "row_min",
    "row_median",
    "row_range",
    "row_skew",
    "row_kurtosis",
    "row_n_unique",
    "row_q25",
    "row_q75",
]

# The row's 20 values sorted ascending: sorted_00 is its smallest factor
# score, sorted_19 its largest. Order-invariant by construction, matching
# the target's own order invariance (a sum doesn't care which factor is
# which).
SORTED_COLS = [f"sorted_{i:02d}" for i in range(len(config.FEATURE_COLS))]

ENGINEERED_COLS = ROW_STAT_COLS + SORTED_COLS


class RowStatsFeatures(BaseEstimator, TransformerMixin):
    """Appends row-level statistics and sorted values to the raw factors.

    Kept as a proper scikit-learn transformer (not a bare function) so it
    composes into a `Pipeline` and is persisted with the fitted model —
    inference always applies the exact same derivation as training.
    Stateless: `fit` learns nothing, so there is no train/test leakage risk.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "RowStatsFeatures":
        return self

    def __sklearn_is_fitted__(self) -> bool:
        # Stateless, so always "fitted" — without this, scikit-learn's
        # check_is_fitted (run when a fitted pipeline is sliced, e.g. by the
        # SHAP helpers) finds no learned `*_` attributes and raises.
        return True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = X[config.FEATURE_COLS].to_numpy(dtype=float)
        sorted_values = np.sort(values, axis=1)

        stats = pd.DataFrame(index=X.index)
        stats["row_sum"] = values.sum(axis=1)
        stats["row_std"] = values.std(axis=1)
        stats["row_max"] = sorted_values[:, -1]
        stats["row_min"] = sorted_values[:, 0]
        stats["row_median"] = np.median(values, axis=1)
        stats["row_range"] = stats["row_max"] - stats["row_min"]
        # pandas' skew/kurt (bias-corrected, matching scipy's bias=False);
        # a constant row has undefined skew/kurtosis, mapped to 0.
        row_frame = pd.DataFrame(values)
        stats["row_skew"] = row_frame.skew(axis=1).fillna(0.0).to_numpy()
        stats["row_kurtosis"] = row_frame.kurt(axis=1).fillna(0.0).to_numpy()
        stats["row_n_unique"] = (np.diff(sorted_values, axis=1) != 0).sum(axis=1) + 1
        stats["row_q25"] = np.quantile(values, 0.25, axis=1)
        stats["row_q75"] = np.quantile(values, 0.75, axis=1)

        sorted_frame = pd.DataFrame(sorted_values, index=X.index, columns=SORTED_COLS)
        return pd.concat([X[config.FEATURE_COLS], stats, sorted_frame], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array(config.FEATURE_COLS + ENGINEERED_COLS)


VALUE_COUNT_RANGE = range(0, 16)  # factor scores observed in the data run 0..18; >15 is very rare
RICH_EXTRA_COLS = (
    [f"count_{v}" for v in VALUE_COUNT_RANGE]
    + [f"row_q{q}" for q in (10, 20, 30, 40, 60, 70, 80, 90)]
    + ["row_mean_sq", "row_mean_cube", "row_cv", "row_n_above_10"]
)


class RichRowStatsFeatures(RowStatsFeatures):
    """RowStatsFeatures plus a wider set of row descriptors, for ensemble diversity:
    how many factors take each value 0-15, finer quantiles, higher moments, the
    coefficient of variation, and how many factors exceed 10 (scores above 10 are
    rare in the original data, so they hint at synthetic perturbation).

    Ideas from the competition's public notebooks; on their own they barely move a
    single model (holdout R^2 +0.0001), but models trained on a different view of
    the row make different mistakes, which is what a stack needs.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        base = super().transform(X)
        values = X[config.FEATURE_COLS].to_numpy(dtype=float)
        extra = {f"count_{v}": (values == v).sum(axis=1) for v in VALUE_COUNT_RANGE}
        for q in (10, 20, 30, 40, 60, 70, 80, 90):
            extra[f"row_q{q}"] = np.quantile(values, q / 100, axis=1)
        mean = values.mean(axis=1)
        extra["row_mean_sq"] = (values**2).mean(axis=1)
        extra["row_mean_cube"] = (values**3).mean(axis=1)
        extra["row_cv"] = np.divide(values.std(axis=1), mean, out=np.zeros_like(mean), where=mean > 0)
        extra["row_n_above_10"] = (values > 10).sum(axis=1)
        return pd.concat([base, pd.DataFrame(extra, index=X.index)], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array(list(super().get_feature_names_out()) + RICH_EXTRA_COLS)


def build_feature_pipeline() -> Pipeline:
    """RowStatsFeatures on its own, without a final estimator.

    Tree models need no scaling or encoding here (every input is already a
    clean integer score with no missing values — verified on the real data),
    so this is the whole preprocessing story. Useful on its own for
    inspecting transformed features (e.g. in the notebook's EDA section)
    without needing a trained model.
    """
    return Pipeline(steps=[("engineer", RowStatsFeatures())])
