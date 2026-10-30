"""Checks for window_features.py. Run from the repository folder with: python tests/test_window_features.py

The module must reproduce the feature tables that the notebooks saved for the training windows.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from window_features import FEATURE_COLUMNS, apply_scaler, window_features  # noqa: E402


def training_features():
    """Features of the 92 saved training windows, computed by the module."""
    saved = pd.read_csv(ROOT / "data" / "features" / "window_features.csv")
    rows = []
    for _, row in saved.iterrows():
        window = pd.read_csv(ROOT / "data" / "processed" / row["activity"].lower() / row["file"])
        rows.append({"file": row["file"], "activity": row["activity"], **window_features(window)})
    return saved, pd.DataFrame(rows)


def test_features_match_the_saved_training_table():
    saved, computed = training_features()
    assert len(saved) == 92
    assert (saved["file"] == computed["file"]).all()
    assert np.allclose(saved[FEATURE_COLUMNS], computed[FEATURE_COLUMNS], rtol=0, atol=1e-9)


def test_normalization_matches_the_saved_normalized_table():
    _, computed = training_features()
    scaler = json.loads((ROOT / "data" / "features" / "scaler.json").read_text())
    saved = pd.read_csv(ROOT / "data" / "features" / "window_features_normalized.csv")
    normalized = apply_scaler(computed, scaler)
    assert np.allclose(saved[FEATURE_COLUMNS], normalized[FEATURE_COLUMNS], rtol=0, atol=1e-9)


def test_a_perfectly_flat_window_gives_finite_features():
    window = pd.DataFrame({"time_s": np.arange(255) * 0.02})
    for sensor in ["acc", "gyro"]:
        for axis in ["x", "y", "z"]:
            window[f"{sensor}_{axis}"] = 0.5
    values = np.array(list(window_features(window).values()))
    assert np.isfinite(values).all()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"{len(tests)} tests passed")
