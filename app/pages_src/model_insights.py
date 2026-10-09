"""Model Insights page: what the app model learned (SHAP), and how the
leaderboard blend was put together (if its saved predictions exist).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import r2_score

from . import shared
from flood_prediction import config
from flood_prediction.features import ENGINEERED_COLS
from flood_prediction.interpretability import top_shap_features


def render():
    st.title("🧠 Model Insights")
    st.caption(
        "What the saved app model (LightGBM) relies on, and how the stronger "
        "leaderboard blend was assembled."
    )

    st.subheader("What does the model rely on? (SHAP)")
    st.caption("Computed live from the saved model on 500 training rows — first load takes a few seconds.")
    explanation, X_transformed = shared.get_shap_explanation(sample_size=500)
    top = top_shap_features(explanation, top_n=15).sort_values("mean_abs_shap")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [shared.PALETTE[3] if f in ENGINEERED_COLS else shared.PALETTE[0] for f in top["feature"]]
    ax.barh(top["feature"], top["mean_abs_shap"] * 100, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Mean |SHAP value| (percentage points, log scale)")
    ax.set_title("Top 15 features (orange = engineered row statistic)")
    sns.despine(ax=ax)
    st.pyplot(fig)
    importance = abs(explanation.values).mean(axis=0)
    row_sum_share = importance[list(explanation.feature_names).index("row_sum")] / importance.sum()
    st.info(
        f"`row_sum` alone carries **{row_sum_share:.0%}** of all the attribution "
        "(note the log scale); the spread statistics and sorted values share most "
        "of the rest. None of the 20 named factors matters much *on its own*, "
        "because the original target is a plain sum that weights every factor equally."
    )

    st.subheader("How the prediction responds to the row sum")
    sum_index = list(X_transformed.columns).index("row_sum")
    fig2, ax2 = plt.subplots(figsize=(7, 3.8))
    ax2.scatter(X_transformed["row_sum"], explanation.values[:, sum_index] * 100, s=8, alpha=0.5,
                color=shared.PALETTE[0])
    ax2.set_xlabel("row_sum")
    ax2.set_ylabel("SHAP contribution (pp)")
    sns.despine(ax=ax2)
    st.pyplot(fig2)
    st.caption(
        "Close to a straight line through the bulk of the data, as the "
        "original formula (0.005 per point) predicts — flattening at the "
        "extremes, where the synthetic data is thin."
    )

    st.subheader("The leaderboard blend")
    saved = shared.get_ensemble_predictions()
    if saved is None:
        st.warning(
            "Run `python scripts/make_ensemble_submission.py` to build the "
            "leaderboard blend; this section then shows each member's "
            "out-of-fold score and its blend weight."
        )
        return

    y = saved["y"]
    rows = [
        {"model": name, "OOF R²": r2_score(y, saved[f"oof_{name}"]), "blend weight": weight}
        for name, weight in zip(saved["members"], saved["weights"])
    ]
    blended = np.column_stack([saved[f"oof_{n}"] for n in saved["members"]]) @ saved["weights"]
    rows.append({"model": "Blend", "OOF R²": r2_score(y, blended), "blend weight": 1.0})
    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.format({"OOF R²": "{:.5f}", "blend weight": "{:.3f}"}),
        hide_index=True, width="stretch",
    )
    st.caption(
        "Out-of-fold R²: each row is scored by the fold model that never saw it. "
        f"Computed on all {len(y):,} training rows (target: `{config.TARGET_COL}`)."
    )
