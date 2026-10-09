"""Turn a late-submission score into a "would have ranked" position.

The competition closed on 2024-05-31, so a late submission gets public and
private scores but no leaderboard position. These helpers compare a score
against a frozen snapshot of the final leaderboards (fetched by
`scripts/fetch_leaderboard.py`) to report where it *would* have placed.

Which board matters: the **private** leaderboard is the final ranking (it
scores the ~80% of the test set nobody saw during the competition); the
public one is what competitors saw live. The private board is reported
first everywhere in this project.

Ties: this competition is packed so tightly that dozens of teams share each
5-decimal score (e.g. 0.86899 covers ranks 13-56 on the private board). A
late score is placed at the *best* rank among teams it ties with — Kaggle
breaks ties by earliest submission, which a late entry would lose, so the
range is reported too (`rank_best`..`rank_worst`).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

PRIVATE_LEADERBOARD_CSV = config.LEADERBOARD_DIR / "private_leaderboard.csv"
PUBLIC_LEADERBOARD_CSV = config.LEADERBOARD_DIR / "public_leaderboard.csv"


# Every late submission this project made, in order — transcribed from
# `kaggle competitions submissions playground-series-s4e5` (scores are
# Kaggle's, not local estimates). The app's Leaderboard Climb page and the
# report both read this, so the climb is told from one source of truth.
SUBMISSION_HISTORY = [
    {
        "step": "v1: LightGBM + XGBoost + CatBoost, plain average",
        "change": "Row statistics + sorted values; first-guess hyperparameters (colsample 0.5)",
        "oof_r2": 0.86916,
        "public": 0.86909,
        "private": 0.86870,
    },
    {
        "step": "v2: same three models, tuned, NNLS-weighted blend",
        "change": "colsample_bytree 0.5 -> 1.0, deeper XGBoost (depth 10); blend weights learned on OOF",
        "oof_r2": 0.86940,
        "public": 0.86931,
        "private": 0.86893,
    },
    {
        "step": "v3: v2 with 7 folds x 2 seeds",
        "change": "Each model sees 6/7 of the data instead of 4/5; two seeds averaged per member",
        "oof_r2": 0.86943,
        "public": 0.86933,
        "private": 0.86893,
    },
]


@dataclass(frozen=True)
class Placement:
    board: str
    score: float
    rank_best: int  # 1 + number of teams strictly better
    rank_worst: int  # 1 + number of teams better or tied
    n_teams: int

    @property
    def percentile(self) -> float:
        """Top-X% at the best rank, e.g. 1.8 means "top 1.8%"."""
        return 100 * self.rank_best / self.n_teams

    def describe(self) -> str:
        span = (
            f"#{self.rank_best}"
            if self.rank_best == self.rank_worst
            else f"#{self.rank_best}-{self.rank_worst} (tied)"
        )
        return (
            f"{self.board} {self.score:.5f} -> would rank {span} "
            f"of {self.n_teams:,} (top {self.percentile:.1f}%)"
        )


def load_scores(path: Path) -> pd.Series:
    """One score per team, best first. Accepts both Kaggle CSV layouts:
    the `-d` download (`Score`) and the paged `-s -v` listing (`score`).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch the leaderboards first: "
            "`python scripts/fetch_leaderboard.py`."
        )
    frame = pd.read_csv(path)
    column = "Score" if "Score" in frame.columns else "score"
    return frame[column].astype(float).sort_values(ascending=False).reset_index(drop=True)


def place(score: float, scores: pd.Series, board: str) -> Placement:
    """Where `score` lands among `scores` (higher is better, R^2).

    Scores are compared at Kaggle's displayed 5-decimal precision, so a raw
    0.868994 counts as a tie with a displayed 0.86899.
    """
    rounded = round(score, 5)
    board_scores = scores.round(5)
    strictly_better = int((board_scores > rounded).sum())
    better_or_tied = int((board_scores >= rounded).sum())
    return Placement(
        board=board,
        score=rounded,
        rank_best=strictly_better + 1,
        rank_worst=max(better_or_tied, strictly_better + 1),
        n_teams=len(scores),
    )


def would_rank(private_score: float, public_score: float | None = None) -> list[Placement]:
    """Placements on the private board, and on the public one if given."""
    placements = [place(private_score, load_scores(PRIVATE_LEADERBOARD_CSV), "private")]
    if public_score is not None:
        placements.append(place(public_score, load_scores(PUBLIC_LEADERBOARD_CSV), "public"))
    return placements
