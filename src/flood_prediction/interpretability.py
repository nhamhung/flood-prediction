"""SHAP-based model interpretability.

SHAP (SHapley Additive exPlanations) attributes each prediction to individual
feature contributions, based on the game-theoretic idea of fairly splitting
credit among "players" (here, features) for a shared payout (here, the
prediction). `shap.TreeExplainer` computes exact Shapley values efficiently
for tree-based models by walking the tree structure directly, rather than the
slow model-agnostic sampling every other SHAP explainer needs.

That efficient path only works for a *single* tree model, not for the
leaderboard blend — so these helpers explain the app model
(`models/model.joblib`, LightGBM by default).

What to expect on this dataset: credit concentrates on `row_sum` and the
sorted values rather than on any named factor. That's the honest answer, not
a flaw — the target is (originally) a sum, so "which factor matters most"
has no meaningful answer beyond "all 20 equally, through their total".
"""

import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 500, random_state: int = 42
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values for a fitted single-tree-model pipeline.

    Returns `(explanation, X_transformed)`: the SHAP explanation (one row per
    sampled input, one column per engineered feature) and the transformed
    feature matrix it was computed against (with real column names) —
    useful for dependence plots that need both the SHAP values and the
    underlying feature values.

    `max_samples` subsamples `X` before computing SHAP values: exact tree
    SHAP is fast per-row, but still linear in row count, and a summary plot
    doesn't need all 1.1M training rows to be informative.
    """
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)

    preprocessing = pipeline[:-1]
    X_transformed = preprocessing.transform(X)

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    explanation = explainer(X_transformed)
    return explanation, X_transformed


def top_shap_features(explanation: shap.Explanation, top_n: int = 15) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value (overall importance)."""
    importance = abs(explanation.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
