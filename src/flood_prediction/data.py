"""Data loading helpers.

Raw CSVs are not committed to the repo (Kaggle competition data may not be
redistributed). Download them first — see the project README for the
`kaggle competitions download` command — or let `_require_file` fetch
them automatically via the Kaggle API (used when deploying without a
Docker image that already bakes the files in; see
`app/pages_src/shared.py` for how deployed credentials get wired in).
Auto-fetching at deploy time is consistent with the competition's own
terms either way: it downloads to this account's own environment on
request, the same as the manual CLI command already documented below,
rather than redistributing the data anywhere.
"""

import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from . import config


def _download_from_kaggle() -> bool:
    """Best-effort automatic fetch via the Kaggle API. Competition
    downloads (unlike plain datasets) come back as a single zip with no
    built-in unzip option, so this extracts it manually. Returns
    whether the target file exists afterward. Silently does nothing
    (returns False) if the `kaggle` package isn't installed, no
    credentials are configured, or this account hasn't accepted the
    competition's rules on kaggle.com yet — callers fall back to the
    manual-download error message either way.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        api.competition_download_files(config.KAGGLE_COMPETITION, path=str(config.DATA_RAW_DIR), quiet=True)
        zip_path = config.DATA_RAW_DIR / f"{config.KAGGLE_COMPETITION}.zip"
        with ZipFile(zip_path) as zf:
            zf.extractall(config.DATA_RAW_DIR)
        zip_path.unlink(missing_ok=True)
    except Exception:
        return False
    return config.TRAIN_CSV.exists()


def _require_file(path: Path) -> Path:
    if not path.exists():
        _download_from_kaggle()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found, and automatic download via the Kaggle API "
            "didn't produce it either (no credentials configured, the "
            "`kaggle` package isn't installed, or this account hasn't "
            "accepted the competition's rules on kaggle.com yet). Download "
            "the competition data manually instead — see the README's 'Get "
            "the data' section, e.g.:\n"
            f"  kaggle competitions download -c {config.KAGGLE_COMPETITION} "
            f"-p {config.DATA_RAW_DIR}\n"
            f"  unzip -o {config.DATA_RAW_DIR / (config.KAGGLE_COMPETITION + '.zip')} "
            f"-d {config.DATA_RAW_DIR}"
        )
    return path


def using_sample_data() -> bool:
    """Whether the bundled Kaggle-derived sample is the active data source."""
    return config.TRAIN_SAMPLE_CSV.exists() and os.getenv("USE_FULL_KAGGLE_DATA", "").lower() not in {"1", "true", "yes"}


def load_train() -> pd.DataFrame:
    """Load fast sample data by default; opt into the full Kaggle download with USE_FULL_KAGGLE_DATA=true."""
    path = config.TRAIN_SAMPLE_CSV if using_sample_data() else _require_file(config.TRAIN_CSV)
    df = pd.read_csv(path)
    return df.set_index(config.ID_COL)


def load_full_train() -> pd.DataFrame:
    """The complete 1,117,957-row training set, never the bundled sample.

    Training, the leaderboard ensemble, the notebook and the report all need
    the full data; only the Streamlit app runs on the sample. Raises rather
    than silently training on fewer rows.
    """
    df = pd.read_csv(_require_file(config.TRAIN_CSV)).set_index(config.ID_COL)
    if len(df) != config.FULL_TRAIN_ROWS:
        raise RuntimeError(
            f"Expected the full training set ({config.FULL_TRAIN_ROWS:,} rows) at {config.TRAIN_CSV}, "
            f"got {len(df):,}. Download the competition data (see the README)."
        )
    return df


def load_original() -> pd.DataFrame | None:
    """The 50k-row original dataset: the downloaded copy if present, else the
    bundled one (CC0, so it may be redistributed); None if neither exists."""
    for path in (config.ORIGINAL_CSV, config.ORIGINAL_SAMPLE_CSV):
        if path.exists():
            return pd.read_csv(path)
    return None


def load_test() -> pd.DataFrame:
    """Load the unlabeled competition test set, indexed by id."""
    df = pd.read_csv(_require_file(config.TEST_CSV))
    return df.set_index(config.ID_COL)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labeled frame into (X, y), restricted to the known feature schema."""
    missing = set(config.RAW_FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Input frame is missing expected columns: {sorted(missing)}")
    X = df[config.RAW_FEATURE_COLS].copy()
    y = df[config.TARGET_COL].copy()
    return X, y
