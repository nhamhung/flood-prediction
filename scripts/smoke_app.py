"""Asset-aware deployment smoke check for the flood-prediction app."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flood_prediction import config, data, leaderboard, model  # noqa: E402


def check_model() -> None:
    pipeline = model.load_pipeline()
    row = pd.DataFrame([{col: 5 for col in config.FEATURE_COLS}])
    prediction = pipeline.predict(row)
    if len(prediction) != 1 or not 0 < float(prediction[0]) < 1:
        raise RuntimeError("The packaged model did not return one flood probability.")


def check_bundled_assets() -> None:
    if not data.using_sample_data():
        raise RuntimeError("The bundled data sample is missing (run scripts/build_sample.py).")
    if data.load_original() is None:
        raise RuntimeError("The bundled original dataset is missing (run scripts/build_sample.py).")
    leaderboard.load_scores(leaderboard.PRIVATE_LEADERBOARD_CSV)


def check_streamlit() -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless=true", "--server.port=8501"],
        cwd=PROJECT_ROOT,
    )
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming healthy.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(1)
        raise RuntimeError("Streamlit health endpoint did not become ready within 90 seconds.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    check_model()
    check_bundled_assets()
    check_streamlit()
    print("flood-prediction deployment smoke check passed")
