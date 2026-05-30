"""
preprocessing/pipeline.py — Normalisation, windowing, and splits.

Enforces:
  - z-score computed on train timeline only (no leakage)
  - Windows built inside each temporal segment (no cross-split leakage)
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
                    TRAIN_FRAC, VAL_FRAC, DATA_PROC, STATION_NAME,
                    RANDOM_SEED, PREPROCESS_VERSION)
from utils import get_logger

logger = get_logger(__name__)
np.random.seed(RANDOM_SEED)


@dataclass
class Dataset:
    """
    Holds train/val/test arrays for one horizon.
    X: (N, window * n_features)  — flattened input window
    y: (N, n_targets)            — target values horizon steps ahead
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
        n = len(self.target_names)
        mean = self.scaler_mean[:n]
        std = self.scaler_std[:n]
        return y_norm * std + mean

    def summary(self) -> str:
        return (f"Horizon={self.horizon}h  "
                f"train={len(self.X_train)}  val={len(self.X_val)}  test={len(self.X_test)}  "
                f"input_dim={self.X_train.shape[1]}  n_targets={self.y_train.shape[1]}")


class WeatherPreprocessor:
    """Converts a processed weather DataFrame into windowed ML-ready Datasets."""

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

        missing_cols = [t for t in self.targets if t not in df.columns]
        if missing_cols:
            raise ValueError(f"Target columns missing from DataFrame: {missing_cols}")

        logger.info(f"WeatherPreprocessor: {len(df)} timesteps, "
                    f"targets={self.targets}, window={self.window}, "
                    f"horizons={self.horizons}")

    def _split_indices(self, n: int) -> tuple[int, int]:
        train_end = int(n * self.train_f)
        val_end   = train_end + int(n * self.val_f)
        return train_end, val_end

    def _normalise(self, arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        std_safe = np.where(std < 1e-8, 1.0, std)
        return (arr - mean) / std_safe

    @staticmethod
    def _build_windows_in_segment(norm: np.ndarray,
                                 timestamps: np.ndarray,
                                 horizon_steps: int,
                                 window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sliding windows wholly inside one normalized timeline segment."""
        seg_n = len(norm)
        max_start = seg_n - window - horizon_steps + 1
        if max_start <= 0:
            return (
                np.empty((0, window * norm.shape[1]), dtype=np.float32),
                np.empty((0, norm.shape[1]), dtype=np.float32),
                np.array([]),
            )

        X_list, y_list, ts_list = [], [], []
        for i in range(max_start):
            window_slice = norm[i : i + window]
            target_val = norm[i + window + horizon_steps - 1]
            X_list.append(window_slice.flatten())
            y_list.append(target_val)
            ts_list.append(timestamps[i + window + horizon_steps - 1])

        return (
            np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(ts_list),
        )

    def build_dataset(self, horizon: int) -> Dataset:
        from config import RESAMPLE_FREQ
        freq_hours = int(RESAMPLE_FREQ.replace("h", ""))
        horizon_steps = horizon // freq_hours

        values = self.df[self.targets].values.astype(np.float32)
        timestamps = self.df.index.values
        n = len(values)

        train_end, val_end = self._split_indices(n)
        mean = values[:train_end].mean(axis=0)
        std = values[:train_end].std(axis=0)
        norm = self._normalise(values, mean, std)

        segments = [
            (norm[:train_end], timestamps[:train_end]),
            (norm[train_end:val_end], timestamps[train_end:val_end]),
            (norm[val_end:], timestamps[val_end:]),
        ]

        parts = [
            self._build_windows_in_segment(seg, ts, horizon_steps, self.window)
            for seg, ts in segments
        ]

        (X_train, y_train, ts_train) = parts[0]
        (X_val, y_val, ts_val) = parts[1]
        (X_test, y_test, ts_test) = parts[2]

        if len(X_train) == 0:
            raise ValueError(
                f"Train segment produced no windows (horizon={horizon}h). "
                f"Need more data or smaller window."
            )

        ds = Dataset(
            horizon=horizon,
            X_train=X_train, y_train=y_train, ts_train=ts_train,
            X_val=X_val, y_val=y_val, ts_val=ts_val,
            X_test=X_test, y_test=y_test, ts_test=ts_test,
            scaler_mean=mean,
            scaler_std=std,
            feature_names=[f"{t}_t{i}" for i in range(self.window) for t in self.targets],
            target_names=self.targets,
        )
        logger.info(f"Built dataset — {ds.summary()} (segment-wise windows, no boundary leakage)")
        return ds

    def build_all(self) -> dict[int, Dataset]:
        return {h: self.build_dataset(h) for h in self.horizons}

    def save(self, datasets: dict, out_dir: Path = None):
        out_dir = out_dir or (DATA_PROC / STATION_NAME)
        out_dir.mkdir(parents=True, exist_ok=True)
        for h, ds in datasets.items():
            path = out_dir / f"dataset_h{h}_{PREPROCESS_VERSION}.pkl"
            with open(path, "wb") as f:
                pickle.dump(ds, f)
            logger.info(f"Saved dataset h={h} -> {path}")

    @staticmethod
    def load(horizon: int, out_dir: Path = None) -> "Dataset":
        from config import DATA_PROC, STATION_NAME
        out_dir = out_dir or (DATA_PROC / STATION_NAME)
        path = out_dir / f"dataset_h{horizon}_{PREPROCESS_VERSION}.pkl"
        if not path.exists():
            legacy = out_dir / f"dataset_h{horizon}.pkl"
            if legacy.exists():
                path = legacy
            else:
                raise FileNotFoundError(f"No cached dataset at {path}. Run preprocessing first.")
        with open(path, "rb") as f:
            return pickle.load(f)
