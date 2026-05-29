"""
evaluation/metrics.py — RMSE, MAE, VPT/Lyapunov, and results table.

All metrics return per-target arrays. Mean across targets is the scalar summary.
VPT (Valid Prediction Time) is the chaotic-forecasting metric required by the challenge.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LYAPUNOV_TIME_HOURS, VPT_THRESHOLD, RESULTS
from utils import get_logger

logger = get_logger(__name__)


# ── Core metrics ──────────────────────────────────────────────────────────────

def rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Root Mean Squared Error per target. Returns shape (n_targets,)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def mae_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean Absolute Error per target. Returns shape (n_targets,)."""
    return np.mean(np.abs(y_true - y_pred), axis=0)


def valid_prediction_time(y_true: np.ndarray, y_pred: np.ndarray,
                           horizon_hours: int,
                           lyapunov_hours: float = None,
                           threshold: float = None) -> float:
    """
    Valid Prediction Time (VPT): the fraction of the test set for which
    normalised error < threshold, expressed in units of Lyapunov time.

    Normalised error = ||y_pred - y_true|| / (||y_true|| + ε)

    Standard in chaotic forecasting literature. A VPT of 1.0 means the
    model forecasts accurately for one full Lyapunov time.
    """
    lyapunov_hours = lyapunov_hours or LYAPUNOV_TIME_HOURS
    threshold      = threshold      or VPT_THRESHOLD
    eps = 1e-8

    # Normalised error per sample (averaged across targets)
    norm_err = np.linalg.norm(y_pred - y_true, axis=1) / (
        np.linalg.norm(y_true, axis=1) + eps
    )

    # VPT = number of steps where error < threshold × step_size / Lyapunov_time
    valid_steps = np.sum(norm_err < threshold)
    vpt_hours   = valid_steps * horizon_hours
    vpt_lyapunov = vpt_hours / lyapunov_hours

    return float(vpt_lyapunov)


def mnist_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy for MNIST benchmark (0–1 scale)."""
    labels_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    labels_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    return float(np.mean(labels_true == labels_pred))


# ── Results table ─────────────────────────────────────────────────────────────

def build_results_table(results: dict,
                         y_true_val: np.ndarray,
                         y_true_test: np.ndarray,
                         target_names: list,
                         horizon_hours: int,
                         out_dir: Path = None) -> pd.DataFrame:
    """
    Build and save a formatted results table.

    results: dict of model_name → BaselineResult or ReadoutResult
             Each must have y_pred_val and y_pred_test attributes.
    """
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, result in results.items():
        # Handle both BaselineResult and ReadoutResult
        pred_val  = getattr(result, "y_pred_val",  None)
        pred_test = getattr(result, "y_pred_test", None)

        if pred_val is None:
            logger.warning(f"No y_pred stored for {name}; skipping metrics row")
            continue

        # Align lengths — warmup trimming can produce off-by-one differences
        n_val  = min(len(y_true_val),  len(pred_val))
        n_test = min(len(y_true_test), len(pred_test))
        y_true_val_  = y_true_val[:n_val]
        pred_val_    = pred_val[:n_val]
        y_true_test_ = y_true_test[:n_test]
        pred_test_   = pred_test[:n_test]

        val_rmse  = rmse_per_target(y_true_val_,  pred_val_)
        test_rmse = rmse_per_target(y_true_test_, pred_test_)
        val_mae   = mae_per_target(y_true_val_,   pred_val_)
        test_mae  = mae_per_target(y_true_test_,  pred_test_)
        vpt       = valid_prediction_time(y_true_test_, pred_test_, horizon_hours)

        row = {"model": name}
        for t, tname in enumerate(target_names):
            row[f"val_rmse_{tname}"]  = round(float(val_rmse[t]),  4)
            row[f"test_rmse_{tname}"] = round(float(test_rmse[t]), 4)
            row[f"val_mae_{tname}"]   = round(float(val_mae[t]),   4)
            row[f"test_mae_{tname}"]  = round(float(test_mae[t]),  4)
        row["val_rmse_mean"]  = round(float(val_rmse.mean()),  4)
        row["test_rmse_mean"] = round(float(test_rmse.mean()), 4)
        row["vpt_lyapunov"]   = round(vpt, 3)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("test_rmse_mean")
    csv_path = out_dir / f"results_h{horizon_hours}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results table saved → {csv_path}")
    logger.info("\n" + df.to_string(index=False))
    return df


def save_metrics_json(metrics: dict, name: str, out_dir: Path = None):
    """Save a single model's metrics as JSON for the qBraid Skill agent."""
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metrics_{name}.json"
    with open(path, "w") as f:
        json.dump({k: float(v) if hasattr(v, "__float__") else v
                   for k, v in metrics.items()}, f, indent=2)
    logger.debug(f"Metrics JSON saved → {path}")