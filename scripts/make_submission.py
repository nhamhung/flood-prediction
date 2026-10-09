#!/usr/bin/env python
"""Generate a Kaggle-submittable submission.csv from the trained app model.

Usage:
    python scripts/make_submission.py [--output submission.csv]

Then submit with the Kaggle CLI (the competition is closed, so this is a
late submission — scored, but not placed on the leaderboard; see
`flood_prediction.leaderboard` for the "would have ranked" comparison):
    kaggle competitions submit -c playground-series-s4e5 -f submission.csv -m "message"

For the stronger leaderboard blend, use scripts/make_ensemble_submission.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, data, model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submission.csv",
        help="Where to write the submission file (default: ./submission.csv)",
    )
    args = parser.parse_args()

    print(f"Loading test data from {config.TEST_CSV} ...")
    test_df = data.load_test()
    missing = set(config.RAW_FEATURE_COLS) - set(test_df.columns)
    if missing:
        raise ValueError(f"Test data is missing expected columns: {sorted(missing)}")
    X_test = test_df[config.RAW_FEATURE_COLS].copy()

    print(f"Loading trained pipeline from {config.MODEL_PATH} ...")
    pipeline = model.load_pipeline()

    print(f"Predicting {len(X_test):,} rows ...")
    predictions = pipeline.predict(X_test)

    submission = test_df.reset_index()[[config.ID_COL]].copy()
    submission[config.TARGET_COL] = predictions

    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
