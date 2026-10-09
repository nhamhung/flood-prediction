#!/usr/bin/env python
"""Build the small data files bundled with the repository (data/sample/).

Usage:
    python scripts/build_sample.py

- train_sample.csv.gz: a 5,000-row random sample of the competition's
  train.csv, so the Streamlit app starts without a Kaggle download (the same
  approach as this portfolio's other projects).
- original.csv.gz: the full 50,000-row original dataset, which is CC0
  (public domain) — so the app's "it's a sum" discovery works out of the box.

Needs the full data first (see the README's "Get the data").
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config  # noqa: E402

SAMPLE_ROWS = 5_000


def main() -> None:
    config.DATA_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(config.TRAIN_CSV)
    sample = train.sample(SAMPLE_ROWS, random_state=config.RANDOM_SEED).sort_values(config.ID_COL)
    sample.to_csv(config.TRAIN_SAMPLE_CSV, index=False, compression="gzip")
    pd.read_csv(config.ORIGINAL_CSV).to_csv(config.ORIGINAL_SAMPLE_CSV, index=False, compression="gzip")
    for path in (config.TRAIN_SAMPLE_CSV, config.ORIGINAL_SAMPLE_CSV):
        print(f"wrote {path.relative_to(config.PROJECT_ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
