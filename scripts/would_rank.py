#!/usr/bin/env python
"""Where would a late submission have ranked on the final leaderboard?

Usage:
    python scripts/would_rank.py                          # your latest Kaggle submissions
    python scripts/would_rank.py --private 0.86899 --public 0.86935

With no arguments, lists your submissions to this competition via the
Kaggle CLI and places each one. Needs the leaderboard snapshot first:
`python scripts/fetch_leaderboard.py`.
"""

import argparse
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, leaderboard  # noqa: E402


def my_submissions() -> pd.DataFrame:
    result = subprocess.run(
        ["kaggle", "competitions", "submissions", config.KAGGLE_COMPETITION, "-v"],
        capture_output=True, text=True, check=True,
    )
    frame = pd.read_csv(io.StringIO(result.stdout))
    frame = frame[frame["status"].astype(str).str.contains("COMPLETE", case=False)]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--private", type=float, help="Private leaderboard score (R^2).")
    parser.add_argument("--public", type=float, help="Public leaderboard score (R^2).")
    args = parser.parse_args()

    if args.private is not None:
        for placement in leaderboard.would_rank(args.private, args.public):
            print(placement.describe())
        return

    submissions = my_submissions()
    if submissions.empty:
        print("No scored submissions yet.")
        return
    for _, row in submissions.iterrows():
        print(f"{row['fileName']} — {row['description']} ({row['date']})")
        for placement in leaderboard.would_rank(float(row["privateScore"]), float(row["publicScore"])):
            print(f"  {placement.describe()}")


if __name__ == "__main__":
    main()
