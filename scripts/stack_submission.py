#!/usr/bin/env python
"""Build the leaderboard stack: many diverse members combined by Ridge.

Usage:
    python scripts/stack_submission.py --list                       # available members
    python scripts/stack_submission.py                              # all members (cached ones are reused)
    python scripts/stack_submission.py --members lightgbm xgboost catboost lightgbm_rich

Each member is trained with K-fold CV on the full training set (cached under
data/processed/oof/), then a non-negative Ridge regression learns how to
combine their out-of-fold predictions. Writes submission_stack.csv.

Then submit (a late submission — see flood_prediction.leaderboard):
    kaggle competitions submit -c playground-series-s4e5 -f submission_stack.csv -m "message"
"""

import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, data, stacking  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", nargs="*", default=list(stacking.MEMBERS), help="Member slugs (default: all).")
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=1, help="Seeds per member for members not yet cached.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularisation.")
    parser.add_argument("--list", action="store_true", help="List members and exit.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "submission_stack.csv")
    args = parser.parse_args()

    if args.list:
        for slug, member in stacking.MEMBERS.items():
            print(f"  {slug:<20} {member.name} ({member.features} features)")
        return

    started = time.time()
    train_df = data.load_full_train()
    X, y = data.split_features_target(train_df)
    test_df = data.load_test()
    X_test = test_df[config.RAW_FEATURE_COLS]

    oof, test = {}, {}
    for slug in args.members:
        member = stacking.MEMBERS[slug]
        print(f"{member.name} ({member.features} features) ...")
        oof[slug], test[slug] = stacking.member_predictions(member, X, y, X_test, args.folds, args.seeds, log=print)

    stack = stacking.fit_stack(oof, y, alpha=args.alpha)
    print("\nMember OOF R^2 and stack weights:")
    for name, weight in zip(stack.members, stack.weights):
        print(f"  {name:<20} OOF R^2 {stack.member_oof_r2[name]:.5f}   weight {weight:.3f}")
    print(f"Stack OOF R^2 (cross-validated): {stack.oof_r2:.5f}")

    submission = test_df.reset_index()[[config.ID_COL]].copy()
    submission[config.TARGET_COL] = stacking.predict_stack(stack, test)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} predictions to {args.output} ({(time.time() - started) / 60:.0f} min)")


if __name__ == "__main__":
    main()
