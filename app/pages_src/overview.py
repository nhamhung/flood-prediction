"""Dataset Overview page: what the data looks like, and the one discovery
that shapes the whole project — the target is (originally) just a sum.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from . import shared
from flood_prediction import config, data


def render():
    st.title("📊 Dataset Overview")
    st.caption(
        "Kaggle's Playground Series S4E5 — a synthetic dataset generated from a "
        "50,000-row original 'Flood Prediction' dataset."
    )

    train_df = shared.get_train_df()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", f"{len(train_df):,}")
    c2.metric("Risk factors", len(config.FEATURE_COLS))
    c3.metric("Missing values", int(train_df.isna().sum().sum()))
    c4.metric("Distinct target values", train_df[config.TARGET_COL].nunique())
    if data.using_sample_data():
        st.info(
            f"Showing the bundled {len(train_df):,}-row random sample of the 1,117,957-row "
            "competition training set, so the app starts without a Kaggle download. Set "
            "`USE_FULL_KAGGLE_DATA=true` (with Kaggle credentials) to load the full data."
        )

    st.subheader("The target only takes a handful of values")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    counts = train_df[config.TARGET_COL].value_counts().sort_index()
    ax.bar(counts.index, counts.values, width=0.004, color=shared.PALETTE[0])
    ax.set_xlabel("FloodProbability")
    ax.set_ylabel("Rows")
    sns.despine(ax=ax)
    st.pyplot(fig)
    st.caption(
        "Every value is a multiple of 0.005 — a strong hint that it was computed, "
        "not measured."
    )

    st.subheader("The key discovery: it's (originally) just a sum")
    original_df = shared.get_original_df()
    sample = shared.get_engineered_sample()
    fig2, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    if original_df is not None:
        original_sum = original_df[config.FEATURE_COLS].sum(axis=1)
        axes[0].scatter(original_sum, original_df[config.TARGET_COL], s=3, alpha=0.3, color=shared.PALETTE[4])
        axes[0].set_title("Original data (50k rows): a perfect line")
    else:
        axes[0].text(0.5, 0.5, "Original dataset not downloaded\n(see README → Get the data)",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Original data")
    axes[0].set_xlabel("Sum of the 20 factors")
    axes[0].set_ylabel("FloodProbability")
    axes[1].scatter(sample["row_sum"], sample[config.TARGET_COL], s=2, alpha=0.08, color=shared.PALETTE[0])
    axes[1].set_title(f"Competition data ({len(sample):,}-row sample): noisy")
    axes[1].set_xlabel("Sum of the 20 factors")
    for ax in axes:
        sns.despine(ax=ax)
    st.pyplot(fig2)
    st.info(
        "In the original dataset, **FloodProbability = 0.005 × (sum of all 20 "
        "factors)** — exactly, for every row. Kaggle's synthetic generator added "
        "noise to the factor values, so the competition is really asking: *given "
        "noisy factors, recover the original sum.* That's why the engineered "
        "features are row-level statistics, not per-factor transformations."
    )

    st.subheader("Each factor on its own is a weak signal")
    correlations = (
        sample[config.FEATURE_COLS + ["row_sum"]].corrwith(sample[config.TARGET_COL]).sort_values()
    )
    fig3, ax3 = plt.subplots(figsize=(7, 5.5))
    colors = [shared.PALETTE[3] if c == "row_sum" else shared.PALETTE[1] for c in correlations.index]
    ax3.barh(correlations.index, correlations.values, color=colors)
    ax3.set_xlabel("Correlation with FloodProbability")
    sns.despine(ax=ax3)
    st.pyplot(fig3)
    st.caption(
        "All 20 factors correlate about equally (and weakly) with the target; "
        "their sum (orange) correlates far more strongly than any of them."
    )
