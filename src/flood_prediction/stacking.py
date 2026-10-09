"""Leaderboard stack: many diverse members, combined by a Ridge regression.

The approach of the competition's 1st-place solution (public notebook
"PSS4E05 #1st place solution - Ensemble with ridge"): train many models that
see the data differently, keep each one's out-of-fold (OOF) predictions, and
learn how to combine them with a linear model. This module is this project's
own implementation of that idea.

Each member's OOF and test predictions are cached under
data/processed/oof/, so adding a member never retrains the others. The stack's
own score is itself cross-validated (Ridge fit on 4/5 of the rows, scored on
the 5th), so it can't flatter itself by fitting its weights on the rows it is
scored on.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from . import config
from .features import RichRowStatsFeatures, RowStatsFeatures
from .model import catboost_estimator, lightgbm_estimator, with_seed, xgboost_estimator

OOF_DIR = config.DATA_PROCESSED_DIR / "oof"


@dataclass(frozen=True)
class Member:
    name: str
    estimator: Callable[[], BaseEstimator]
    features: str = "standard"  # "standard" (RowStatsFeatures) or "rich" (RichRowStatsFeatures)

    @property
    def slug(self) -> str:
        return "_".join("".join(c if c.isalnum() else " " for c in self.name.lower()).split())


def _lgbm(**overrides) -> Callable[[], BaseEstimator]:
    return lambda: lightgbm_estimator().set_params(**overrides)


def _xgb(**overrides) -> Callable[[], BaseEstimator]:
    return lambda: xgboost_estimator().set_params(**overrides)


def _cat(**overrides) -> Callable[[], BaseEstimator]:
    return lambda: catboost_estimator().set_params(**overrides)


MEMBERS = {m.slug: m for m in [
    Member("LightGBM", _lgbm()),
    Member("XGBoost", _xgb()),
    Member("CatBoost", _cat()),
    Member("LightGBM rich", _lgbm(), "rich"),
    Member("XGBoost rich", _xgb(), "rich"),
    Member("CatBoost rich", _cat(), "rich"),
    Member("LightGBM wide", _lgbm(num_leaves=127, max_depth=-1, min_child_samples=200, n_estimators=1000)),
    Member("XGBoost shallow", _xgb(max_depth=6, n_estimators=1100)),
]}


def build_member_pipeline(member: Member, seed: int = config.RANDOM_SEED) -> Pipeline:
    features = RichRowStatsFeatures() if member.features == "rich" else RowStatsFeatures()
    return Pipeline([("engineer", features), ("model", with_seed(member.estimator(), seed))])


def _cache_path(member: Member, n_splits: int, n_seeds: int) -> Path:
    return OOF_DIR / f"{member.slug}_{n_splits}f{n_seeds}s.npz"


def member_predictions(
    member: Member, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame,
    n_splits: int = 7, n_seeds: int = 1, log: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray]:
    """(oof, test) predictions for one member — from cache if available."""
    # Reuse any cached run with the same folds; prefer the one with the most seeds.
    cached = sorted(OOF_DIR.glob(f"{member.slug}_{n_splits}f*s.npz"),
                    key=lambda p: int(p.stem.rsplit("f", 1)[1].rstrip("s")), reverse=True)
    for path in cached:
        with np.load(path) as saved:
            if len(saved["oof"]) == len(X) and len(saved["test"]) == len(X_test):
                return saved["oof"], saved["test"]
    path = _cache_path(member, n_splits, n_seeds)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    oof, test = np.zeros(len(X)), np.zeros(len(X_test))
    for fold, (train_idx, valid_idx) in enumerate(folds.split(X)):
        for seed in range(n_seeds):
            pipeline = build_member_pipeline(member, config.RANDOM_SEED + seed)
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[valid_idx] += pipeline.predict(X.iloc[valid_idx]) / n_seeds
            test += pipeline.predict(X_test) / (n_splits * n_seeds)
        log(f"  {member.name}: fold {fold + 1}/{n_splits} R^2 {r2_score(y.iloc[valid_idx], oof[valid_idx]):.5f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, oof=oof, test=test)
    return oof, test


@dataclass
class Stack:
    members: list[str]
    weights: np.ndarray
    intercept: float
    oof_r2: float                 # cross-validated score of the stack itself
    member_oof_r2: dict[str, float]


def fit_stack(oof: dict[str, np.ndarray], y: pd.Series, alpha: float = 1.0) -> Stack:
    """Ridge with non-negative weights on members' OOF predictions; its own
    score is cross-validated over rows (5-fold) before the final fit."""
    names = list(oof)
    matrix = np.column_stack([oof[n] for n in names])
    target = np.asarray(y, dtype=float)
    stacked_oof = np.zeros(len(target))
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=config.RANDOM_SEED + 1).split(matrix):
        ridge = Ridge(alpha=alpha, positive=True).fit(matrix[train_idx], target[train_idx])
        stacked_oof[valid_idx] = ridge.predict(matrix[valid_idx])
    final = Ridge(alpha=alpha, positive=True).fit(matrix, target)
    return Stack(
        members=names, weights=final.coef_, intercept=float(final.intercept_),
        oof_r2=float(r2_score(target, stacked_oof)),
        member_oof_r2={n: float(r2_score(target, oof[n])) for n in names},
    )


def predict_stack(stack: Stack, test: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([test[n] for n in stack.members]) @ stack.weights + stack.intercept
