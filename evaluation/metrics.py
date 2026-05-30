"""
evaluation/metrics.py — RMSE, MAE, VPT, physical units, and results tables.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (LYAPUNOV_TIME_HOURS, VPT_THRESHOLD, RESULTS,
                    RESAMPLE_HOURS, REPORT_PHYSICAL_UNITS)
from utils import get_logger

logger = get_logger(__name__)


def rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def mae_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)


def valid_prediction_time(y_true: np.ndarray, y_pred: np.ndarray,
                           step_hours: float = None,
                           lyapunov_hours: float = None,
                           threshold: float = None) -> float:
    """
    VPT for direct one-step forecasts on a regular grid.

    Counts test samples with normalised error below threshold, multiplies by
    the resample step (6h), and normalises by Lyapunov time.
    """
    step_hours = step_hours or RESAMPLE_HOURS
    lyapunov_hours = lyapunov_hours or LYAPUNOV_TIME_HOURS
    threshold = threshold or VPT_THRESHOLD
    eps = 1e-8

    norm_err = np.linalg.norm(y_pred - y_true, axis=1) / (
        np.linalg.norm(y_true, axis=1) + eps
    )
    valid_steps = int(np.sum(norm_err < threshold))
    vpt_hours = valid_steps * step_hours
    return float(vpt_hours / lyapunov_hours)


def build_results_table(results: dict,
                         y_true_val: np.ndarray,
                         y_true_test: np.ndarray,
                         target_names: list,
                         horizon_hours: int,
                         out_dir: Path = None,
                         dataset=None) -> pd.DataFrame:
    """Build results CSV with normalized and optional physical-unit metrics."""
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    inv_y = dataset.inverse_transform_y if dataset is not None else None

    rows = []
    for name, result in results.items():
        pred_val = getattr(result, "y_pred_val", None)
        pred_test = getattr(result, "y_pred_test", None)
        if pred_val is None:
            logger.warning(f"No y_pred stored for {name}; skipping")
            continue

        offset = int(getattr(result, "label_offset", 0))
        yv = y_true_val[offset:]
        yt = y_true_test[offset:]
        n_val = min(len(yv), len(pred_val))
        n_test = min(len(yt), len(pred_test))
        y_true_val_ = yv[:n_val]
        pred_val_ = pred_val[:n_val]
        y_true_test_ = yt[:n_test]
        pred_test_ = pred_test[:n_test]

        val_rmse = rmse_per_target(y_true_val_, pred_val_)
        test_rmse = rmse_per_target(y_true_test_, pred_test_)
        val_mae = mae_per_target(y_true_val_, pred_val_)
        test_mae = mae_per_target(y_true_test_, pred_test_)
        vpt = valid_prediction_time(y_true_test_, pred_test_)

        row = {"model": name, "horizon_hours": horizon_hours}
        for t, tname in enumerate(target_names):
            row[f"val_rmse_{tname}"] = round(float(val_rmse[t]), 4)
            row[f"test_rmse_{tname}"] = round(float(test_rmse[t]), 4)
            row[f"val_mae_{tname}"] = round(float(val_mae[t]), 4)
            row[f"test_mae_{tname}"] = round(float(test_mae[t]), 4)

        row["val_rmse_mean"] = round(float(val_rmse.mean()), 4)
        row["test_rmse_mean"] = round(float(test_rmse.mean()), 4)
        row["vpt_lyapunov"] = round(vpt, 3)

        if REPORT_PHYSICAL_UNITS and inv_y is not None:
            yt_phys = inv_y(y_true_test_)
            pt_phys = inv_y(pred_test_)
            phys_rmse = rmse_per_target(yt_phys, pt_phys)
            row["test_rmse_mean_phys"] = round(float(phys_rmse.mean()), 4)
            for t, tname in enumerate(target_names):
                row[f"test_rmse_phys_{tname}"] = round(float(phys_rmse[t]), 4)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("test_rmse_mean")
    csv_path = out_dir / f"results_h{horizon_hours}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results table saved -> {csv_path}")
    logger.info("\n" + df.to_string(index=False))
    return df
