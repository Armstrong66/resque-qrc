"""
preprocessing/pipeline.py — Normalisation, windowing, and splits.

Enforces:
  - z-score computed on train split ONLY (no data leakage)
  - Strict temporal ordering (no shuffle)
  - Forecast horizon baked into labels at construction time
"""

import sys
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TARGETS, WINDOW_SIZE, HORIZONS,
                    TRAIN_FRAC, VAL_FRAC, DATA_PROC, STATION_NAME, RANDOM_SEED)
from utils import get_logger

logger = get_logger(__name__)
np.random.seed(RANDOM_SEED)


@dataclass
class Dataset:
    """
    Holds train/val/test arrays for one horizon.
    X: (N, window * n_features)  — flattened input window
    y: (N, n_targets)            — target values horizon steps ahead
    timestamps: (N,)             — timestamps of the forecast target step
    scaler_params: dict          — mean/std used for inverse-transform
    """
    horizon:       int
    X_train:       np.ndarray
    y_train:       np.ndarray
    X_val:         np.ndarray
    y_val:         np.ndarray
    X_test:        np.ndarray
    y_test:        np.ndarray
    ts_train:      np.ndarray = field(default_factory=lambda: np.array([]))
    ts_val:        np.ndarray = field(default_factory=lambda: np.array([]))
    ts_test:       np.ndarray = field(default_factory=lambda: np.array([]))
    scaler_mean:   Optional[np.ndarray] = None
    scaler_std:    Optional[np.ndarray] = None
    feature_names: list = field(default_factory=list)
    target_names:  list = field(default_factory=list)

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalised predictions back to physical units."""
        if self.scaler_mean is None or self.scaler_std is None:
            return y_norm
        # scaler_mean/std are for all features; targets are the last n_targets columns
        n = len(self.target_names)
        mean = self.scaler_mean[-n:]
        std  = self.scaler_std[-n:]
        return y_norm * std + mean

    def summary(self) -> str:
        return (f"Horizon={self.horizon}h  "
                f"train={len(self.X_train)}  val={len(self.X_val)}  test={len(self.X_test)}  "
                f"input_dim={self.X_train.shape[1]}  n_targets={self.y_train.shape[1]}")


class WeatherPreprocessor:
    """
    Converts a processed weather DataFrame into windowed ML-ready Datasets.

    Usage:
        prep = WeatherPreprocessor(df)
        datasets = prep.build_all()     # returns dict: {6: Dataset, 24: Dataset}
    """

    def __init__(self, df: pd.DataFrame,
                 targets: list = None,
                 window: int = None,
                 horizons: list = None,
                 train_frac: float = None,
                 val_frac: float = None):
        self.df       = df.copy()
        self.targets  = targets  or TARGETS
        self.window   = window   or WINDOW_SIZE
        self.horizons = horizons or HORIZONS
        self.train_f  = train_frac or TRAIN_FRAC
        self.val_f    = val_frac   or VAL_FRAC

        # Validate all targets present
        missing_cols = [t for t in self.targets if t not in df.columns]
        if missing_cols:
            raise ValueError(f"Target columns missing from DataFrame: {missing_cols}")

        logger.info(f"WeatherPreprocessor: {len(df)} timesteps, "
                    f"targets={self.targets}, window={self.window}, "
                    f"horizons={self.horizons}")

    def _split_indices(self, n: int) -> tuple[int, int]:
        """Return (train_end, val_end) indices for strict temporal split."""
        train_end = int(n * self.train_f)
        val_end   = train_end + int(n * self.val_f)
        return train_end, val_end

    def _normalise(self, arr: np.ndarray,
                   mean: np.ndarray,
                   std: np.ndarray) -> np.ndarray:
        std_safe = np.where(std < 1e-8, 1.0, std)
        return (arr - mean) / std_safe

    def build_dataset(self, horizon: int) -> Dataset:
        """
        Build windowed Dataset for a single forecast horizon (in timesteps,
        where 1 timestep = RESAMPLE_FREQ, e.g. 6h → horizon=1 for 6h-ahead).
        Caller passes horizon in HOURS; we convert to steps internally.
        """
        from config import RESAMPLE_FREQ
        freq_hours = int(RESAMPLE_FREQ.replace("h", ""))
        horizon_steps = horizon // freq_hours

        # Work with a clean numeric array
        values = self.df[self.targets].values.astype(np.float32)
        timestamps = self.df.index.values
        n = len(values)

        # ── Split BEFORE normalisation to prevent leakage ──────────────────
        train_end, val_end = self._split_indices(n)

        # Compute normalisation stats from train split only
        mean = values[:train_end].mean(axis=0)
        std  = values[:train_end].std(axis=0)
        norm = self._normalise(values, mean, std)

        # ── Build sliding windows ──────────────────────────────────────────
        X_list, y_list, ts_list = [], [], []
        max_start = n - self.window - horizon_steps
        if max_start <= 0:
            raise ValueError(
                f"Not enough data for window={self.window} + horizon={horizon_steps} "
                f"steps. Have {n} timesteps."
            )

        for i in range(max_start):
            window_slice = norm[i : i + self.window]           # (W, n_targets)
            target_val   = norm[i + self.window + horizon_steps - 1]  # (n_targets,)
            X_list.append(window_slice.flatten())
            y_list.append(target_val)
            ts_list.append(timestamps[i + self.window + horizon_steps - 1])

        X   = np.array(X_list, dtype=np.float32)
        y   = np.array(y_list, dtype=np.float32)
        ts  = np.array(ts_list)

        # ── Apply temporal split ────────────────────────────────────────────
        # The window construction shifts indices; use proportional cut on X
        n_samples = len(X)
        t1 = int(n_samples * self.train_f)
        t2 = t1 + int(n_samples * self.val_f)

        ds = Dataset(
            horizon      = horizon,
            X_train      = X[:t1],        y_train  = y[:t1],   ts_train = ts[:t1],
            X_val        = X[t1:t2],      y_val    = y[t1:t2], ts_val   = ts[t1:t2],
            X_test       = X[t2:],        y_test   = y[t2:],   ts_test  = ts[t2:],
            scaler_mean  = mean,
            scaler_std   = std,
            feature_names= self.targets * self.window,
            target_names = self.targets,
        )
        logger.info(f"Built dataset — {ds.summary()}")
        return ds

    def build_all(self) -> dict[int, Dataset]:
        """Build datasets for all configured horizons."""
        return {h: self.build_dataset(h) for h in self.horizons}

    def save(self, datasets: dict, out_dir: Path = None):
        out_dir = out_dir or (DATA_PROC / STATION_NAME)
        out_dir.mkdir(parents=True, exist_ok=True)
        for h, ds in datasets.items():
            path = out_dir / f"dataset_h{h}.pkl"
            with open(path, "wb") as f:
                pickle.dump(ds, f)
            logger.info(f"Saved dataset h={h} → {path}")

    @staticmethod
    def load(horizon: int, out_dir: Path = None) -> "Dataset":
        from config import DATA_PROC, STATION_NAME
        out_dir = out_dir or (DATA_PROC / STATION_NAME)
        path = out_dir / f"dataset_h{horizon}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No cached dataset at {path}. Run preprocessing first.")
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    import pandas as pd
    from data.parser import load_and_merge
    from data.downloader import download_all

    paths = download_all()
    df = load_and_merge(paths)
    prep = WeatherPreprocessor(df)
    datasets = prep.build_all()
    prep.save(datasets)
    for h, ds in datasets.items():
        print(ds.summary())