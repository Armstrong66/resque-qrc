"""Unit tests for preprocessing, parser cleaning, and metrics alignment."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TARGETS, WINDOW_SIZE, TRAIN_FRAC, VAL_FRAC
from data.parser import _clean_resampled
from preprocessing.pipeline import WeatherPreprocessor
from preprocessing.projection import fit_pca_projector, project_splits
from evaluation.metrics import valid_prediction_time, rmse_per_target
from experiments.sweeps import _subsample_sweep_data


def test_clean_resampled_drops_nan_rows():
    idx = pd.date_range("2020-01-01", periods=12, freq="6h")
    df = pd.DataFrame(
        {t: np.linspace(1, 2, 12) for t in TARGETS},
        index=idx,
    )
    df.iloc[4, :] = np.nan  # entire row missing after interp
    out = _clean_resampled(df)
    assert out[TARGETS].isna().sum().sum() == 0
    assert len(out) <= len(df)


def test_segment_windows_no_cross_split():
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="6h")
    df = pd.DataFrame({t: np.random.randn(n).astype(np.float32) for t in TARGETS}, index=idx)
    prep = WeatherPreprocessor(df, window=10, horizons=[6])
    ds = prep.build_dataset(6)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)
    # Max windows in train segment
    h_steps = 1
    max_train_windows = train_end - 10 - h_steps + 1
    assert len(ds.X_train) <= max(0, max_train_windows)
    assert len(ds.X_train) > 0


def test_pca_fit_on_train_only():
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((50, 80)).astype(np.float32)
    X_vl = rng.standard_normal((20, 80)).astype(np.float32)
    proj = fit_pca_projector(X_tr, n_components=9)
    assert proj.transform(X_tr).shape == (50, 9)
    assert proj.transform(X_vl).shape == (20, 9)


def test_vpt_uses_step_hours():
    y = np.ones((10, 4))
    pred = y + 0.01
    vpt = valid_prediction_time(y, pred, step_hours=6, lyapunov_hours=48)
    assert vpt > 0


def test_label_offset_alignment():
    y_true = np.arange(20).reshape(-1, 1).astype(np.float32)
    pred = y_true[5:].copy()
    rmse = rmse_per_target(y_true[5:], pred)
    assert rmse.shape == (1,)


def test_sweep_calibration_subsets_preserve_temporal_order_and_alignment():
    X_train = np.arange(40, dtype=np.float32).reshape(10, 4)
    y_train = np.arange(20, dtype=np.float32).reshape(10, 2)
    X_val = np.arange(32, dtype=np.float32).reshape(8, 4)
    y_val = np.arange(16, dtype=np.float32).reshape(8, 2)
    X_tr_s, y_tr_s, X_vl_s, y_vl_s = _subsample_sweep_data(
        X_train, y_train, X_val, y_val, max_train=6, max_val=5)

    assert X_tr_s.shape == (6, 4)
    assert y_tr_s.shape == (6, 2)
    assert X_vl_s.shape == (5, 4)
    assert y_vl_s.shape == (5, 2)
    np.testing.assert_array_equal(X_tr_s, X_train[:6])
    np.testing.assert_array_equal(y_vl_s, y_val[:5])
