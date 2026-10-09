#!/usr/bin/env python
"""Train the app model on the full training set and save it to models/model.joblib.

Usage:
    python scripts/train.py                      # default: LightGBM (the app model)
    python scripts/train.py --model catboost     # pick any single model (see --help)
    python scripts/train.py --skip-cv            # skip the 5-fold CV report (faster)

This is the explainable single model the notebook, app, and
`scripts/make_submission.py` use. The leaderboard blend is built separately
by `scripts/make_ensemble_submission.py`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, data, model  # noqa: E402

# CLI-friendly aliases for model.MODEL_FACTORIES's keys.
MODEL_CHOICES = {
    "linear": "Linear Regression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHOICES),
        default="lightgbm",
        help="Which single model to train (default: lightgbm).",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip the 5-fold cross-validation report and just fit the final model.",
    )
    args = parser.parse_args()
    estimator_name = MODEL_CHOICES[args.model]

    print(f"Loading training data from {config.TRAIN_CSV} ...")
    train_df = data.load_full_train()
    X, y = data.split_features_target(train_df)
    print(f"Loaded {len(X):,} rows, {X.shape[1]} raw feature columns.")

    if not args.skip_cv:
        print(f"Cross-validating {estimator_name} (5-fold) ...")
        cv_scores = model.cross_validate_pipeline(
            X, y, estimator=model.MODEL_FACTORIES[estimator_name](), cv=5
        )
        print(f"  R^2  = {cv_scores['r2_mean']:.5f} (+/- {cv_scores['r2_std']:.5f})")
        print(f"  RMSE = {cv_scores['rmse_mean']:.5f} (+/- {cv_scores['rmse_std']:.5f})")

    print("Fitting final model on the full training set ...")
    pipeline = model.train_pipeline(X, y, estimator=model.MODEL_FACTORIES[estimator_name]())

    model.save_pipeline(pipeline)
    print(f"Saved trained pipeline to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
