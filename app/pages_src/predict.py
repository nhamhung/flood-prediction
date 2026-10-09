"""Flood Risk Calculator page: score an area's 20 risk factors, see the
model's flood probability, and see which inputs pushed it up or down.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from . import shared
from flood_prediction import config, features

SLIDER_MAX = 16  # the synthetic data's factors reach ~16-18; 0-10 covers ~99%


def _key(col: str) -> str:
    return f"factor_{col}"


def _set_row(row: dict, actual: float | None = None) -> None:
    for col in config.FEATURE_COLS:
        st.session_state[_key(col)] = int(min(row[col], SLIDER_MAX))
    st.session_state["loaded_actual"] = actual


def _load_random_area() -> None:
    sample = shared.get_train_df().sample(1).iloc[0]
    _set_row(sample.to_dict(), actual=float(sample[config.TARGET_COL]))


def render():
    st.title("🌊 Flood Risk Calculator")
    st.caption(
        "Score an area on 20 flood-risk factors (0 = no risk, 10 = severe — every "
        "factor is scored so that higher means *more* risk), and the model "
        "estimates its flood probability."
    )

    if _key(config.FEATURE_COLS[0]) not in st.session_state:
        _set_row(shared.get_default_row())

    c1, c2, _ = st.columns([1, 1, 2])
    c1.button("🎲 Load a random real area", on_click=_load_random_area, width="stretch")
    c2.button(
        "↺ Reset to typical", on_click=_set_row, args=(shared.get_default_row(),),
        width="stretch",
    )

    columns = st.columns(len(shared.FACTOR_GROUPS))
    for column, (group, cols) in zip(columns, shared.FACTOR_GROUPS.items()):
        with column:
            st.markdown(f"**{group}**")
            for col in cols:
                st.slider(shared.factor_label(col), 0, SLIDER_MAX, key=_key(col))

    row = pd.DataFrame(
        [{col: st.session_state[_key(col)] for col in config.FEATURE_COLS}]
    )
    prediction = float(shared.get_pipeline().predict(row)[0])
    factor_sum = int(row.iloc[0].sum())
    original_formula = config.ORIGINAL_TARGET_PER_POINT * factor_sum

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Model's flood probability", f"{prediction:.1%}")
    m2.metric(
        "Original dataset's formula", f"{original_formula:.1%}",
        help="In the original 50k-row dataset, flood probability is exactly "
        "0.005 x the sum of the 20 factors. The model learned something close "
        "to this from the noisy synthetic data alone.",
    )
    actual = st.session_state.get("loaded_actual")
    if actual is not None:
        m3.metric("This area's actual label", f"{actual:.1%}", delta=f"{prediction - actual:+.1%} model error",
                  delta_color="off")
    else:
        m3.metric("Sum of all 20 factors", factor_sum)

    st.subheader("Why this prediction?")
    st.caption(
        "SHAP contributions for this exact input: how far each feature moved "
        "the prediction away from the training-set average. Expect the "
        "row-level statistics (sum, sorted values) to dominate — the target is "
        "fundamentally a sum, so no single named factor matters on its own."
    )
    engineered = features.RowStatsFeatures().transform(row)
    explanation = shared.get_explainer()(engineered)
    contributions = pd.Series(explanation.values[0], index=engineered.columns)
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(10)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [shared.PALETTE[3] if v > 0 else shared.PALETTE[0] for v in top.values]
    ax.barh(top.index, top.values * 100, color=colors)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("Contribution to flood probability (percentage points)")
    ax.set_title(f"Baseline {float(np.ravel(explanation.base_values)[0]):.1%} → prediction {prediction:.1%}")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    st.pyplot(fig)
    st.caption("Orange pushes the probability up, blue pushes it down.")
