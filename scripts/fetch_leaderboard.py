#!/usr/bin/env python
"""Fetch the competition's final public and private leaderboards.

Usage:
    python scripts/fetch_leaderboard.py

Writes data/raw/leaderboard/{public,private}_leaderboard.csv, which
`flood_prediction.leaderboard` reads to turn a late-submission score into a
"would have ranked" position.

Why two different Kaggle CLI calls: `kaggle competitions leaderboard -d`
downloads the *public* board as one CSV, but there's no download for the
*private* (final) board — `-s` shows it a page at a time, so this pages
through it (200 teams per page, Kaggle's maximum).
"""

import io
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, leaderboard  # noqa: E402

NEXT_PAGE_PREFIX = "Next Page Token = "


def _kaggle(*args: str) -> str:
    result = subprocess.run(
        ["kaggle", "competitions", "leaderboard", config.KAGGLE_COMPETITION, *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def fetch_public() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        _kaggle("-d", "-q", "-p", tmp)
        (zip_path,) = Path(tmp).glob("*.zip")
        with ZipFile(zip_path) as zf:
            (csv_name,) = [n for n in zf.namelist() if n.endswith(".csv")]
            with zf.open(csv_name) as f:
                return pd.read_csv(f)


def fetch_private() -> pd.DataFrame:
    pages, token = [], None
    while True:
        args = ["-s", "-v", "--page-size", "200"]
        if token:
            args += ["--page-token", token]
        output = _kaggle(*args)
        lines = output.splitlines()
        token = next(
            (line[len(NEXT_PAGE_PREFIX):] for line in lines if line.startswith(NEXT_PAGE_PREFIX)),
            None,
        )
        csv_text = "\n".join(line for line in lines if not line.startswith(NEXT_PAGE_PREFIX))
        pages.append(pd.read_csv(io.StringIO(csv_text)))
        print(f"  fetched {sum(len(p) for p in pages):,} private-board teams ...")
        if not token:
            break
    board = pd.concat(pages).drop_duplicates("teamId")
    return board.sort_values("score", ascending=False)


def main() -> None:
    config.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching public leaderboard ...")
    public = fetch_public()
    public.to_csv(leaderboard.PUBLIC_LEADERBOARD_CSV, index=False)
    print(f"  {len(public):,} teams -> {leaderboard.PUBLIC_LEADERBOARD_CSV}")
    print("Fetching private (final) leaderboard ...")
    private = fetch_private()
    private.to_csv(leaderboard.PRIVATE_LEADERBOARD_CSV, index=False)
    print(f"  {len(private):,} teams -> {leaderboard.PRIVATE_LEADERBOARD_CSV}")


if __name__ == "__main__":
    main()
