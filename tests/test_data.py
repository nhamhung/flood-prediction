"""Tests for bundled-sample vs. full-data loading (no download needed)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flood_prediction import config, data  # noqa: E402


def test_load_train_prefers_bundled_sample(monkeypatch, tmp_path):
    sample_csv = tmp_path / "sample.csv"
    sample_csv.write_text("id,FloodProbability\n7,0.5\n", encoding="utf-8")
    monkeypatch.setattr(config, "TRAIN_SAMPLE_CSV", sample_csv)
    monkeypatch.delenv("USE_FULL_KAGGLE_DATA", raising=False)
    monkeypatch.setattr(data, "_download_from_kaggle", lambda: pytest.fail("unexpected download"))
    assert data.load_train().loc[7, config.TARGET_COL] == 0.5


def test_full_data_flag_bypasses_sample(monkeypatch, tmp_path):
    sample_csv = tmp_path / "sample.csv"
    sample_csv.write_text("id,FloodProbability\n7,0.5\n", encoding="utf-8")
    full_csv = tmp_path / "train.csv"
    full_csv.write_text("id,FloodProbability\n9,0.6\n", encoding="utf-8")
    monkeypatch.setattr(config, "TRAIN_SAMPLE_CSV", sample_csv)
    monkeypatch.setattr(config, "TRAIN_CSV", full_csv)
    monkeypatch.setenv("USE_FULL_KAGGLE_DATA", "true")
    assert not data.using_sample_data()
    assert data.load_train().loc[9, config.TARGET_COL] == 0.6


def test_original_falls_back_to_bundled_copy(monkeypatch, tmp_path):
    bundled = tmp_path / "original.csv"
    bundled.write_text("FloodProbability\n0.5\n", encoding="utf-8")
    monkeypatch.setattr(config, "ORIGINAL_CSV", tmp_path / "missing.csv")
    monkeypatch.setattr(config, "ORIGINAL_SAMPLE_CSV", bundled)
    assert len(data.load_original()) == 1


def test_original_is_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ORIGINAL_CSV", tmp_path / "a.csv")
    monkeypatch.setattr(config, "ORIGINAL_SAMPLE_CSV", tmp_path / "b.csv")
    assert data.load_original() is None


def test_load_full_train_ignores_sample_and_checks_size(monkeypatch, tmp_path):
    sample_csv = tmp_path / "sample.csv"
    sample_csv.write_text("id,FloodProbability\n7,0.5\n", encoding="utf-8")
    full_csv = tmp_path / "train.csv"
    full_csv.write_text("id,FloodProbability\n9,0.6\n", encoding="utf-8")
    monkeypatch.setattr(config, "TRAIN_SAMPLE_CSV", sample_csv)
    monkeypatch.setattr(config, "TRAIN_CSV", full_csv)
    monkeypatch.delenv("USE_FULL_KAGGLE_DATA", raising=False)
    with pytest.raises(RuntimeError, match="full training set"):
        data.load_full_train()  # one row is not the full 1,117,957
    monkeypatch.setattr(config, "FULL_TRAIN_ROWS", 1)
    assert data.load_full_train().index.tolist() == [9]
