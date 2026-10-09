"""Tests for the "would have ranked" leaderboard placement."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import leaderboard  # noqa: E402

SCORES = pd.Series([0.86905, 0.86902, 0.86899, 0.86899, 0.86899, 0.86890, 0.85000])


class TestPlace:
    def test_better_than_everyone_is_rank_one(self):
        placement = leaderboard.place(0.87, SCORES, "private")
        assert (placement.rank_best, placement.rank_worst) == (1, 1)

    def test_tie_reports_the_full_rank_range(self):
        placement = leaderboard.place(0.86899, SCORES, "private")
        assert (placement.rank_best, placement.rank_worst) == (3, 5)
        assert "#3-5 (tied)" in placement.describe()

    def test_raw_score_is_compared_at_kaggles_five_decimals(self):
        # 0.868994 displays as 0.86899 on Kaggle, so it ties rather than loses.
        assert leaderboard.place(0.868994, SCORES, "private").rank_best == 3

    def test_between_scores(self):
        placement = leaderboard.place(0.8695 - 0.0005, SCORES, "public")  # 0.86900
        assert (placement.rank_best, placement.rank_worst) == (3, 3)
        assert placement.board == "public"

    def test_worst_score_ranks_last_plus_one(self):
        placement = leaderboard.place(0.1, SCORES, "private")
        assert placement.rank_best == len(SCORES) + 1
        assert placement.percentile == pytest.approx(100 * 8 / 7)


class TestLoadScores:
    @pytest.mark.parametrize("column", ["Score", "score"])
    def test_accepts_both_kaggle_csv_layouts(self, tmp_path, column):
        path = tmp_path / "board.csv"
        pd.DataFrame({column: [0.5, 0.9, 0.7]}).to_csv(path, index=False)
        assert list(leaderboard.load_scores(path)) == [0.9, 0.7, 0.5]

    def test_missing_snapshot_explains_how_to_fetch(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fetch_leaderboard.py"):
            leaderboard.load_scores(tmp_path / "missing.csv")
