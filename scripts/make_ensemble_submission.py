#!/usr/bin/env python
"""Build the leaderboard model: a K-fold LightGBM + XGBoost + CatBoost blend.

Usage:
    python scripts/make_ensemble_submission.py                   # 5 folds x 1 seed (~8 min)
    python scripts/make_ensemble_submission.py --folds 7 --seeds 2   # the final leaderboard run

Timings are on a 10-core laptop (1.1M rows). Writes:
  - the submission CSV
  - data/processed/ensemble_predictions.npz — every member's out-of-fold
    and test predictions plus the blend weights, so the notebook, report,
    and app can show how the blend was assembled without retraining it.

Then submit (a late submission — see flood_prediction.leaderboard):
    kaggle competitions submit -c playground-series-s4e5 -f submission_ensemble.csv -m "message"
"""

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, data, model  # noqa: E402

# Flush every line, so progress shows up even when output is redirected to a log.
print = functools.partial(print, flush=True)  # noqa: A001


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submission_ensemble.csv",
        help="Where to write the submission file (default: ./submission_ensemble.csv)",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5).")
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Random seeds averaged per member per fold (seed bagging; default: 1).",
    )
    args = parser.parse_args()

    started = time.time()
    train_df = data.load_full_train()
    X, y = data.split_features_target(train_df)
    test_df = data.load_test()
    X_test = test_df[config.RAW_FEATURE_COLS].copy()
    print(f"Loaded {len(X):,} training rows and {len(X_test):,} test rows.")

    print(
        f"Training {', '.join(model.ENSEMBLE_MEMBERS)} with {args.folds}-fold CV, "
        f"{args.seeds} seed(s) each ..."
    )
    oof, test = model.out_of_fold_predictions(
        X, y, X_test, n_splits=args.folds, n_seeds=args.seeds, log=print
    )

    print("Out-of-fold R^2 per member:")
    for name, predictions in oof.items():
        print(f"  {name:<9} {r2_score(y, predictions):.5f}")

    weights = model.fit_blend_weights(oof, y)
    blended_oof = model.blend(oof, weights)
    print("Blend weights: " + ", ".join(f"{n} {w:.3f}" for n, w in zip(oof, weights)))
    print(f"Blended out-of-fold R^2: {r2_score(y, blended_oof):.5f}")

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.ENSEMBLE_PREDICTIONS_PATH,
        members=np.array(list(oof)),
        weights=weights,
        y=y.to_numpy(),
        **{f"oof_{name}": p for name, p in oof.items()},
        **{f"test_{name}": p for name, p in test.items()},
    )
    print(f"Saved member predictions to {config.ENSEMBLE_PREDICTIONS_PATH}")

    submission = test_df.reset_index()[[config.ID_COL]].copy()
    submission[config.TARGET_COL] = model.blend(test, weights)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output} ({(time.time() - started) / 60:.0f} min)")


if __name__ == "__main__":
    main()
