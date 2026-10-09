"""Model pipeline construction, training, evaluation, and persistence.

Two models, two jobs (the same split as the disaster_tweets_nlp project):

- **The app model** (`models/model.joblib`, built by `scripts/train.py`): one
  LightGBM regressor inside a scikit-learn pipeline. Trains in about a
  minute, loads instantly, and SHAP can explain every prediction it makes —
  so the notebook, the app, and `scripts/make_submission.py` all use it.
- **The leaderboard model** (`scripts/make_ensemble_submission.py`): a K-fold,
  seed-bagged blend of LightGBM + XGBoost + CatBoost, weighted by
  `fit_blend_weights` on out-of-fold predictions. Slower to build and not
  explainable as a whole. The blend's edge over its best member is small
  (final OOF R^2 0.86943 vs 0.86941 — the members' errors are >99%
  correlated), but on this leaderboard even the fifth decimal moves ranks.
"""

from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from . import config
from .features import build_feature_pipeline

# --- Individual model factories -----------------------------------------
# Each returns a fresh, unfitted estimator. Kept as small named functions
# (rather than one dict of instances) so every call gets an independent
# estimator instance — sharing one instance across folds would let fitted
# state leak between them.
#
# Tree counts are fixed (no early stopping inside the pipeline), chosen from
# early-stopped runs on a 20% holdout of the real training data: early
# stopping needs a validation set passed through `fit`, which a plain
# `Pipeline.fit(X, y)` can't do without leaking a fold's own rows into its
# stopping decision.


def linear_estimator() -> BaseEstimator:
    """Ordinary least squares on the same features — the honest baseline.

    Since the original target is a linear function of the row sum, this is
    much stronger than a typical "linear baseline" (holdout R^2 ~0.845), and
    every tree model has to justify itself against it.
    """
    return LinearRegression()


# The single most important hyperparameter on this dataset is column
# sampling: every tree must be able to split on `row_sum` and the sorted
# values, so `colsample_bytree=1.0`. Sampling half the columns per tree (a
# common default) cost 0.0005 holdout R^2 for LightGBM — more than the gap
# between leaderboard rank 50 and rank 500.


def lightgbm_estimator() -> BaseEstimator:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=600,
        learning_rate=0.012,
        num_leaves=250,
        max_depth=10,
        min_child_samples=114,
        colsample_bytree=1.0,
        # Row subsampling makes the seed matter, so `out_of_fold_predictions`
        # can average several seeds (a small, free variance reduction);
        # verified no loss vs. no subsampling on the holdout (0.86921 vs 0.86920).
        subsample=0.8,
        subsample_freq=1,
        random_state=config.RANDOM_SEED,
        verbosity=-1,
    )


def xgboost_estimator() -> BaseEstimator:
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=360,
        learning_rate=0.02,
        max_depth=10,
        min_child_weight=50,
        subsample=0.9,
        colsample_bytree=1.0,
        tree_method="hist",
        random_state=config.RANDOM_SEED,
    )


def catboost_estimator() -> BaseEstimator:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=1400,
        learning_rate=0.06,
        depth=8,
        random_seed=config.RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )


MODEL_FACTORIES: dict[str, Callable[[], BaseEstimator]] = {
    "Linear Regression": linear_estimator,
    "LightGBM": lightgbm_estimator,
    "XGBoost": xgboost_estimator,
    "CatBoost": catboost_estimator,
}

# The members of the leaderboard blend, in a fixed order (blend weights are
# stored positionally).
ENSEMBLE_MEMBERS = ["LightGBM", "XGBoost", "CatBoost"]


def build_pipeline(estimator: BaseEstimator | None = None) -> Pipeline:
    """Feature engineering + regressor."""
    pipeline = build_feature_pipeline()
    steps = list(pipeline.steps)
    # `is None`, not `estimator or ...`: some sklearn estimators define
    # `__len__` but not `__bool__`, so truthiness can raise on an unfitted one.
    steps.append(("model", estimator if estimator is not None else lightgbm_estimator()))
    return Pipeline(steps=steps)


def cross_validate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: BaseEstimator | None = None,
    cv: int = 5,
) -> dict:
    """Shuffled K-fold cross-validation; returns mean/std R^2 and RMSE.

    R^2 is the competition metric. RMSE is reported alongside it because it
    is in the target's own units (flood probability), which is easier to
    reason about: an RMSE of 0.0185 means a typical miss of ~1.9 points.
    """
    pipeline = build_pipeline(estimator)
    folds = KFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_SEED)
    scores = cross_validate(
        pipeline, X, y, cv=folds, scoring=["r2", "neg_root_mean_squared_error"]
    )
    return {
        "r2_mean": scores["test_r2"].mean(),
        "r2_std": scores["test_r2"].std(),
        "rmse_mean": -scores["test_neg_root_mean_squared_error"].mean(),
        "rmse_std": scores["test_neg_root_mean_squared_error"].std(),
    }


def train_pipeline(
    X: pd.DataFrame, y: pd.Series, estimator: BaseEstimator | None = None
) -> Pipeline:
    """Fit a fresh pipeline on the full given data."""
    pipeline = build_pipeline(estimator)
    pipeline.fit(X, y)
    return pipeline


# --- Leaderboard ensemble -----------------------------------------------


def with_seed(estimator: BaseEstimator, seed: int) -> BaseEstimator:
    """Set whichever random-seed parameter this estimator uses (CatBoost
    calls it `random_seed`, everything else `random_state`); estimators
    with neither (e.g. LinearRegression) are returned unchanged.

    CatBoost is matched by type, not via `get_params()`: its `get_params()`
    lists only explicitly-set parameters, so `random_seed` is absent from a
    default-constructed model.
    """
    if type(estimator).__module__.startswith("catboost"):
        return estimator.set_params(random_seed=seed)
    if "random_state" in estimator.get_params():
        return estimator.set_params(random_state=seed)
    return estimator


def out_of_fold_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    members: list[str] = ENSEMBLE_MEMBERS,
    n_splits: int = 5,
    n_seeds: int = 1,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """K-fold train each member; return (oof, test) prediction dicts.

    `oof[name][i]` is member `name`'s prediction for training row `i` from
    the fold model(s) that never saw row `i` — an honest estimate of how
    that member does on unseen data, and the only legitimate input for
    learning blend weights. `test[name]` averages all fold models'
    predictions on `X_test`.

    `n_seeds > 1` trains each member that many times per fold with
    different random seeds and averages them (seed bagging): the same model,
    with less of the run-to-run noise that matters at this leaderboard's
    fifth decimal.
    """
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    oof = {name: np.zeros(len(X)) for name in members}
    test = {name: np.zeros(len(X_test)) for name in members}
    for fold, (train_idx, valid_idx) in enumerate(folds.split(X)):
        for name in members:
            for seed in range(n_seeds):
                estimator = with_seed(MODEL_FACTORIES[name](), config.RANDOM_SEED + seed)
                pipeline = train_pipeline(X.iloc[train_idx], y.iloc[train_idx], estimator)
                oof[name][valid_idx] += pipeline.predict(X.iloc[valid_idx]) / n_seeds
                test[name] += pipeline.predict(X_test) / (n_splits * n_seeds)
            fold_r2 = r2_score(y.iloc[valid_idx], oof[name][valid_idx])
            log(f"  fold {fold + 1}/{n_splits}  {name:<9} R^2 = {fold_r2:.5f}")
    return oof, test


def fit_blend_weights(oof: dict[str, np.ndarray], y: pd.Series) -> np.ndarray:
    """Non-negative least-squares blend weights over the members' OOF
    predictions, normalised to sum to 1.

    Non-negative so a member can be dropped (weight 0) but never subtracted —
    negative weights on highly correlated models fit OOF noise and tend not
    to transfer to the test set. Normalised so the blend stays a weighted
    average and its scale matches the target's.
    """
    matrix = np.column_stack(list(oof.values()))
    weights, _ = nnls(matrix, np.asarray(y, dtype=float))
    if weights.sum() == 0:
        return np.full(len(oof), 1 / len(oof))
    return weights / weights.sum()


def blend(predictions: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted average of member predictions, members in dict order."""
    return np.column_stack(list(predictions.values())) @ weights


# --- Persistence ---------------------------------------------------------


def save_pipeline(pipeline: Pipeline, path: Path = config.MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path = config.MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train a model first: `python scripts/train.py`."
        )
    return joblib.load(path)
