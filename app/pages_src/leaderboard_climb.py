"""Leaderboard Climb page: every late submission, where it would have
ranked on the final (private) leaderboard, and how tightly packed the top is.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from . import shared
from flood_prediction import leaderboard


@st.cache_data
def _private_scores() -> pd.Series:
    return leaderboard.load_scores(leaderboard.PRIVATE_LEADERBOARD_CSV)


def render():
    st.title("🏆 Leaderboard Climb")
    st.caption(
        "The competition closed on 31 May 2024, so every submission here is a "
        "*late* submission: Kaggle scores it, but doesn't place it. Each score "
        "is compared against the frozen final leaderboard instead."
    )

    history = pd.DataFrame(leaderboard.SUBMISSION_HISTORY)
    placements = [
        leaderboard.would_rank(row.private, row.public) for row in history.itertuples()
    ]
    history["private rank"] = [
        f"#{p[0].rank_best}" if p[0].rank_best == p[0].rank_worst
        else f"#{p[0].rank_best}–{p[0].rank_worst}"
        for p in placements
    ]
    history["top %"] = [f"{p[0].percentile:.1f}%" for p in placements]

    best = history.loc[history["private"].idxmax()]
    best_placement = leaderboard.would_rank(best["private"], best["public"])[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best private R²", f"{best['private']:.5f}")
    c2.metric("Would have ranked", f"#{best_placement.rank_best}", help=best_placement.describe())
    c3.metric("Teams in the competition", f"{best_placement.n_teams:,}")

    st.subheader("Every submission")
    st.dataframe(
        history.rename(columns={"oof_r2": "OOF R²", "public": "public R²", "private": "private R²"}),
        hide_index=True, width="stretch",
        column_config={
            "OOF R²": st.column_config.NumberColumn(format="%.5f"),
            "public R²": st.column_config.NumberColumn(format="%.5f"),
            "private R²": st.column_config.NumberColumn(format="%.5f"),
        },
    )
    st.caption(
        "OOF R²: this project's own out-of-fold cross-validation estimate. Private R²: "
        "Kaggle's final score on the ~80% of the test set hidden during the "
        "competition — the one that decides the ranking."
    )

    st.subheader("How tightly packed the top is")
    scores = _private_scores()
    top = scores.head(600)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(range(1, len(top) + 1), top.values, color=shared.PALETTE[0], linewidth=1.5)
    for row in history.itertuples():
        rank = leaderboard.place(row.private, scores, "private").rank_best
        ax.scatter([rank], [round(row.private, 5)], color=shared.PALETTE[3], zorder=3, s=30)
    ax.axvline(50, color="#888", linestyle="--", linewidth=0.8)
    ax.text(52, top.values[-1], "top 50", color="#666", fontsize=8, va="bottom")
    ax.set_xlabel("Final (private) rank")
    ax.set_ylabel("Private R²")
    sns.despine(ax=ax)
    st.pyplot(fig)
    st.info(
        f"Rank 1 scored {scores.iloc[0]:.5f}; rank 50 scored {scores.iloc[49]:.5f}; "
        f"rank 500 scored {scores.iloc[499]:.5f}. The whole top 500 fits in a "
        f"{scores.iloc[0] - scores.iloc[499]:.4f} band of R² — on a dataset this "
        "large, the differences are real, but tiny. Orange dots are this "
        "project's submissions."
    )
