"""
readout/ridge_readout.py — Analytical ridge regression readout for QRC.

W* = (XᵀX + λI)⁻¹Xᵀy   ← closed-form, no gradient descent.

Supports:
  - Cold-start (standard QRC)
  - Warm-start: ESN W_out initialises QRC ridge (novel contribution)
  - Single-output and multi-output (joint W* matrix)
  - Independent per-target + ensemble fallback
  - Auto-selects best strategy by val RMSE and logs the decision
"""

import sys
import json
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RIDGE_LAMBDAS, RESULTS, RANDOM_SEED
from utils import get_logger

logger = get_logger(__name__)
np.random.seed(RANDOM_SEED)


@dataclass
class ReadoutResult:
    """Holds fitted weights and metadata for one readout configuration."""
    strategy:    str              # "joint", "independent", "ensemble"
    warm_start:  bool
    lambda_:     float
    W:           np.ndarray       # Shape: (n_features, n_targets)
    val_rmse:    np.ndarray       # Per-target RMSE on val set
    val_rmse_mean: float          # Scalar summary
    target_names: list = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, n_features) → y_pred: (N, n_targets)"""
        return X @ self.W

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.debug(f"Readout saved → {path}")

    @staticmethod
    def load(path: Path) -> "ReadoutResult":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Core analytical solver ────────────────────────────────────────────────────

def _ridge_solve(X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Analytical ridge regression: W* = (XᵀX + λI)⁻¹Xᵀy
    X: (N, d)  y: (N, k)  → W: (d, k)
    Uses SVD for numerical stability when d is large.
    """
    d = X.shape[1]
    try:
        # Gram matrix solve — efficient when N > d
        A = X.T @ X + lambda_ * np.eye(d)
        b = X.T @ y
        W = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: pseudo-inverse
        logger.warning("Singular matrix in ridge solve; falling back to lstsq")
        W, _, _, _ = np.linalg.lstsq(
            X.T @ X + lambda_ * np.eye(d), X.T @ y, rcond=None
        )
    return W.astype(np.float32)


def _rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns per-target RMSE array of shape (n_targets,)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


# ── Warm-start helper ─────────────────────────────────────────────────────────

def esn_warm_start_weights(X_train_esn: np.ndarray,
                            y_train: np.ndarray,
                            lambda_: float,
                            qrc_feature_dim: int) -> np.ndarray:
    """
    Train a ridge readout on classical ESN states, then resize/project
    the weights to match QRC feature dimension.
    Used to initialise QRC readout before the analytical solve (warm-start).

    If ESN dim == QRC dim: use directly.
    If ESN dim != QRC dim: project via random orthogonal matrix (fixed seed).
    """
    W_esn = _ridge_solve(X_train_esn, y_train, lambda_)  # (esn_dim, n_targets)
    esn_dim = W_esn.shape[0]

    if esn_dim == qrc_feature_dim:
        logger.debug("Warm-start: ESN and QRC dims match — direct transfer")
        return W_esn

    # Project ESN weights to QRC dimension
    rng = np.random.default_rng(RANDOM_SEED)
    if esn_dim < qrc_feature_dim:
        P = rng.standard_normal((esn_dim, qrc_feature_dim))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        W_init = P.T @ W_esn     # (qrc_dim, n_targets)
    else:
        P = rng.standard_normal((qrc_feature_dim, esn_dim))
        P /= np.linalg.norm(P, axis=0, keepdims=True)
        W_init = P @ W_esn       # (qrc_dim, n_targets)

    logger.debug(f"Warm-start: projected ESN ({esn_dim}d) → QRC ({qrc_feature_dim}d)")
    return W_init.astype(np.float32)


# ── Main readout fitter ───────────────────────────────────────────────────────

class RidgeReadout:
    """
    Fits and selects the best readout strategy for QRC states.

    Strategies:
      "joint"       — single W* mapping all features → all targets
      "independent" — one W* per target, independently
      "ensemble"    — average predictions of independent models

    Warm-start option available for each strategy.
    Auto-selects best by validation RMSE. Decision logged transparently.
    """

    def __init__(self, target_names: list,
                 lambdas: list = None,
                 modes: list = None,
                 warm_start: bool = True):
        self.target_names = target_names
        self.lambdas      = lambdas or RIDGE_LAMBDAS
        self.modes        = modes or ["joint", "independent", "ensemble"]
        self.warm_start   = warm_start
        self.best_result_: Optional[ReadoutResult] = None
        self._all_results_: list[ReadoutResult] = []

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray,
            X_train_esn: Optional[np.ndarray] = None) -> "ReadoutResult":
        """
        Fit all modes × all lambdas; select best by val_rmse_mean.
        X_train_esn: ESN reservoir states aligned with X_train (for warm-start).
        Returns best ReadoutResult.
        """
        n_targets = y_train.shape[1]
        results = []

        for mode in self.modes:
            for lam in self.lambdas:

                # ── Joint mode ─────────────────────────────────────────────
                if mode in ("joint", "ensemble"):
                    W = self._fit_joint(X_train, y_train, lam,
                                        X_train_esn, warm=self.warm_start)
                    val_pred = X_val @ W
                    val_rmse = _rmse_per_target(y_val, val_pred)
                    results.append(ReadoutResult(
                        strategy     = mode,
                        warm_start   = self.warm_start,
                        lambda_      = lam,
                        W            = W,
                        val_rmse     = val_rmse,
                        val_rmse_mean= float(val_rmse.mean()),
                        target_names = self.target_names,
                    ))

                # ── Independent mode ───────────────────────────────────────
                if mode == "independent":
                    W_all = []
                    for t in range(n_targets):
                        W_t = self._fit_joint(
                            X_train, y_train[:, t:t+1], lam,
                            X_train_esn,
                            warm=self.warm_start
                        )
                        W_all.append(W_t)
                    W = np.concatenate(W_all, axis=1)
                    val_pred = X_val @ W
                    val_rmse = _rmse_per_target(y_val, val_pred)
                    results.append(ReadoutResult(
                        strategy     = "independent",
                        warm_start   = self.warm_start,
                        lambda_      = lam,
                        W            = W,
                        val_rmse     = val_rmse,
                        val_rmse_mean= float(val_rmse.mean()),
                        target_names = self.target_names,
                    ))

        # ── Ensemble: average predictions of best independent + best joint ─
        # (Separate from the modes loop above — done after all individual fits)
        ind_results  = [r for r in results if r.strategy == "independent"]
        joint_results = [r for r in results if r.strategy == "joint"]
        if ind_results and joint_results:
            best_ind   = min(ind_results,   key=lambda r: r.val_rmse_mean)
            best_joint = min(joint_results, key=lambda r: r.val_rmse_mean)
            ens_pred   = 0.5 * (X_val @ best_ind.W + X_val @ best_joint.W)
            ens_rmse   = _rmse_per_target(y_val, ens_pred)
            # Store ensemble as a pseudo-result with W pointing to joint
            # (ensemble prediction uses both at inference time)
            ens_result = ReadoutResult(
                strategy      = "ensemble",
                warm_start    = self.warm_start,
                lambda_       = -1.0,   # Not applicable
                W             = (best_ind.W + best_joint.W) / 2.0,
                val_rmse      = ens_rmse,
                val_rmse_mean = float(ens_rmse.mean()),
                target_names  = self.target_names,
            )
            results.append(ens_result)

        # ── Auto-select best ───────────────────────────────────────────────
        self._all_results_ = results
        best = min(results, key=lambda r: r.val_rmse_mean)
        self.best_result_ = best

        self._log_selection(results, best)
        return best

    def _fit_joint(self, X: np.ndarray, y: np.ndarray, lam: float,
                   X_esn: Optional[np.ndarray], warm: bool) -> np.ndarray:
        """Fit ridge; optionally warm-start from ESN weights."""
        if warm and X_esn is not None:
            W_init = esn_warm_start_weights(X_esn, y, lam, X.shape[1])
            # Incorporate warm-start: shift X by W_init contribution
            # i.e. solve for residual δW around W_init
            residual = y - X @ W_init
            dW = _ridge_solve(X, residual, lam)
            return (W_init + dW).astype(np.float32)
        return _ridge_solve(X, y, lam)

    def _log_selection(self, all_results: list, best: ReadoutResult):
        """Transparently log the selection decision to console and log file."""
        logger.info("=" * 60)
        logger.info("READOUT SELECTION — val RMSE summary")
        logger.info(f"{'Strategy':<14} {'WarmStart':<10} {'Lambda':<10} {'Val RMSE (mean)'}")
        logger.info("-" * 60)
        for r in sorted(all_results, key=lambda x: x.val_rmse_mean):
            marker = " ← SELECTED" if r is best else ""
            logger.info(
                f"{r.strategy:<14} {str(r.warm_start):<10} {r.lambda_:<10.5g} "
                f"{r.val_rmse_mean:.4f}{marker}"
            )
        logger.info("=" * 60)
        logger.info(
            f"Best: strategy={best.strategy} warm={best.warm_start} "
            f"lambda={best.lambda_:.5g} val_rmse_mean={best.val_rmse_mean:.4f}"
        )
        per_target = {t: f"{v:.4f}" for t, v in
                      zip(self.target_names, best.val_rmse)}
        logger.info(f"Per-target val RMSE: {per_target}")

    def save_selection_log(self, out_dir: Path = None):
        """Save selection summary as JSON for the qBraid Skill agent."""
        out_dir = out_dir or RESULTS
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = [
            {
                "strategy":       r.strategy,
                "warm_start":     r.warm_start,
                "lambda":         r.lambda_,
                "val_rmse_mean":  r.val_rmse_mean,
                "val_rmse_per_target": {t: float(v) for t, v
                                        in zip(r.target_names, r.val_rmse)},
            }
            for r in sorted(self._all_results_, key=lambda x: x.val_rmse_mean)
        ]
        path = out_dir / "readout_selection.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Selection log saved → {path}")