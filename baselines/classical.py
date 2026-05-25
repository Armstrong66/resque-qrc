"""
baselines/classical.py — All classical baselines for benchmarking.

Baselines:
  - Persistence: y_pred = last observed value
  - ARIMA:       statsmodels AutoARIMA per target
  - ESN:         Echo State Network (classical QRC analogue) — critical baseline
  - LSTM/GRU:    swappable via model_type flag (single config, both supported)

ESN also provides warm-start weights for the QRC readout (see readout/ridge_readout.py).
"""

import sys
import json
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (ESN_HIDDEN_DIM, RNN_HIDDEN, RNN_LAYERS, RNN_EPOCHS,
                    RNN_LR, RNN_BATCH, RESULTS, RANDOM_SEED)
from utils import get_logger

logger = get_logger(__name__)
np.random.seed(RANDOM_SEED)

# Optional imports — guarded so the project loads even if not installed
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logger.warning("pmdarima not installed. ARIMA baseline will be skipped. "
                   "Install: pip install pmdarima statsmodels")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. LSTM/GRU baselines will be skipped. "
                   "Install: pip install torch")


@dataclass
class BaselineResult:
    name:         str
    y_pred_val:   np.ndarray
    y_pred_test:  np.ndarray
    val_rmse:     np.ndarray
    test_rmse:    np.ndarray
    meta:         dict

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Baseline result saved: {self.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Persistence
# ─────────────────────────────────────────────────────────────────────────────

def run_persistence(y_val: np.ndarray, y_test: np.ndarray,
                    X_val: np.ndarray, X_test: np.ndarray,
                    window: int) -> BaselineResult:
    """
    Forecast = last value in the input window (most recent observation).
    X is flattened (N, window * n_targets); last observation is at index -n_targets.
    """
    n_targets = y_val.shape[1]
    pred_val  = X_val[:, -n_targets:]
    pred_test = X_test[:, -n_targets:]

    from evaluation.metrics import rmse_per_target
    return BaselineResult(
        name        = "persistence",
        y_pred_val  = pred_val,
        y_pred_test = pred_test,
        val_rmse    = rmse_per_target(y_val, pred_val),
        test_rmse   = rmse_per_target(y_test, pred_test),
        meta        = {"type": "persistence"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. ARIMA (per-target)
# ─────────────────────────────────────────────────────────────────────────────

def run_arima(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
              target_names: list) -> Optional[BaselineResult]:
    """Auto-ARIMA per target variable. Returns None if pmdarima not installed."""
    if not ARIMA_AVAILABLE:
        logger.warning("Skipping ARIMA — pmdarima not available")
        return None

    n_targets = y_train.shape[1]
    pred_val  = np.zeros_like(y_val)
    pred_test = np.zeros_like(y_test)
    orders    = {}

    for t in range(n_targets):
        logger.info(f"  ARIMA fitting target: {target_names[t]}")
        try:
            model = auto_arima(y_train[:, t], stepwise=True, seasonal=False,
                               error_action="ignore", suppress_warnings=True,
                               max_p=5, max_q=5)
            orders[target_names[t]] = str(model.order)

            # Walk-forward forecast on val
            history = list(y_train[:, t])
            for i in range(len(y_val)):
                fc = model.predict(1)[0]
                pred_val[i, t] = fc
                history.append(y_val[i, t])
                model.update([y_val[i, t]])

            # Walk-forward on test
            for i in range(len(y_test)):
                fc = model.predict(1)[0]
                pred_test[i, t] = fc
                model.update([y_test[i, t]])

        except Exception as e:
            logger.error(f"  ARIMA failed for {target_names[t]}: {e}")
            pred_val[:, t]  = y_train[-1, t]
            pred_test[:, t] = y_train[-1, t]

    from evaluation.metrics import rmse_per_target
    return BaselineResult(
        name        = "arima",
        y_pred_val  = pred_val,
        y_pred_test = pred_test,
        val_rmse    = rmse_per_target(y_val, pred_val),
        test_rmse   = rmse_per_target(y_test, pred_test),
        meta        = {"type": "arima", "orders": orders},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Echo State Network (ESN) — Classical QRC analogue
# ─────────────────────────────────────────────────────────────────────────────

class EchoStateNetwork:
    """
    Classical Echo State Network — fixed random recurrent reservoir + linear readout.
    This is the CRITICAL baseline: outperforming ESN is what justifies QRC.

    Also produces warm-start weights for the QRC readout (get_readout_weights()).
    """

    def __init__(self, hidden_dim: int = None, spectral_radius: float = 0.95,
                 input_scaling: float = 1.0, leak_rate: float = 0.3,
                 ridge_lambda: float = 1e-4, seed: int = RANDOM_SEED):
        self.hidden_dim       = hidden_dim or ESN_HIDDEN_DIM
        self.spectral_radius  = spectral_radius
        self.input_scaling    = input_scaling
        self.leak_rate        = leak_rate
        self.ridge_lambda     = ridge_lambda
        self.seed             = seed
        self.W_res_   = None   # Reservoir weights
        self.W_in_    = None   # Input weights
        self.W_out_   = None   # Readout weights (trained)
        self._init_reservoir()

    def _init_reservoir(self):
        rng = np.random.default_rng(self.seed)
        # Random sparse reservoir matrix
        W = rng.standard_normal((self.hidden_dim, self.hidden_dim))
        # Scale to desired spectral radius
        eigvals = np.linalg.eigvals(W)
        W = W / (np.max(np.abs(eigvals)) + 1e-8) * self.spectral_radius
        self.W_res_ = W.astype(np.float32)
        logger.debug(f"ESN reservoir: {self.hidden_dim}×{self.hidden_dim} "
                     f"ρ={np.max(np.abs(np.linalg.eigvals(self.W_res_))):.3f}")

    def _run_reservoir(self, X: np.ndarray, warmup: int = 50) -> np.ndarray:
        """Drive ESN with X (N, n_features). Returns states (N-warmup, hidden_dim)."""
        N, n_in = X.shape
        if self.W_in_ is None:
            rng = np.random.default_rng(self.seed)
            self.W_in_ = (rng.standard_normal((self.hidden_dim, n_in))
                          * self.input_scaling).astype(np.float32)
        h = np.zeros(self.hidden_dim, dtype=np.float32)
        states = []
        for t in range(N):
            pre = self.W_res_ @ h + self.W_in_ @ X[t]
            h = (1 - self.leak_rate) * h + self.leak_rate * np.tanh(pre)
            if t >= warmup:
                states.append(h.copy())
        return np.array(states, dtype=np.float32)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            warmup: int = 50) -> "EchoStateNetwork":
        """Train linear readout on reservoir states via ridge regression."""
        H = self._run_reservoir(X_train, warmup)
        y_trimmed = y_train[warmup:]
        if len(H) != len(y_trimmed):
            min_len = min(len(H), len(y_trimmed))
            H, y_trimmed = H[:min_len], y_trimmed[:min_len]
        lam = self.ridge_lambda
        A = H.T @ H + lam * np.eye(self.hidden_dim)
        b = H.T @ y_trimmed
        self.W_out_ = np.linalg.solve(A, b).astype(np.float32)
        logger.info(f"ESN fitted: hidden={self.hidden_dim} "
                    f"lambda={lam} W_out shape={self.W_out_.shape}")
        return self

    def predict(self, X: np.ndarray, warmup: int = 0) -> np.ndarray:
        H = self._run_reservoir(X, warmup)
        return H @ self.W_out_

    def get_reservoir_states(self, X: np.ndarray, warmup: int = 0) -> np.ndarray:
        """Return raw reservoir states — used for warm-start weight extraction."""
        return self._run_reservoir(X, warmup)

    def get_readout_weights(self) -> np.ndarray:
        """Return trained W_out for use as QRC warm-start."""
        if self.W_out_ is None:
            raise RuntimeError("ESN not fitted yet. Call fit() first.")
        return self.W_out_


def run_esn(X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray,
            hidden_dim: int = None) -> tuple[BaselineResult, EchoStateNetwork]:
    """Fit and evaluate ESN. Returns (result, fitted_esn) — esn reused for warm-start."""
    esn = EchoStateNetwork(hidden_dim=hidden_dim or ESN_HIDDEN_DIM)
    esn.fit(X_train, y_train)

    pred_val  = esn.predict(X_val)
    pred_test = esn.predict(X_test)

    from evaluation.metrics import rmse_per_target
    result = BaselineResult(
        name        = "esn",
        y_pred_val  = pred_val,
        y_pred_test = pred_test,
        val_rmse    = rmse_per_target(y_val, pred_val),
        test_rmse   = rmse_per_target(y_test, pred_test),
        meta        = {"type": "esn", "hidden_dim": esn.hidden_dim},
    )
    return result, esn


# ─────────────────────────────────────────────────────────────────────────────
# 4. LSTM / GRU — swappable via model_type flag
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class _RNNModel(nn.Module):
        def __init__(self, n_features: int, n_targets: int,
                     hidden: int, n_layers: int,
                     model_type: Literal["lstm", "gru"]):
            super().__init__()
            self.model_type = model_type.lower()
            rnn_cls = nn.LSTM if self.model_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(n_features, hidden, n_layers,
                               batch_first=True, dropout=0.1 if n_layers > 1 else 0.0)
            self.fc  = nn.Linear(hidden, n_targets)

        def forward(self, x):
            # x: (batch, seq, features)
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])   # Last timestep


def run_rnn(X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray,
            X_test:  np.ndarray, y_test:  np.ndarray,
            window: int = 20,
            model_type: Literal["lstm", "gru"] = "lstm") -> Optional[BaselineResult]:
    """Train and evaluate LSTM or GRU. model_type='gru' to switch."""
    if not TORCH_AVAILABLE:
        logger.warning(f"Skipping {model_type.upper()} — PyTorch not available")
        return None

    import torch
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training {model_type.upper()} on {device}")

    n_features = X_train.shape[1] // window
    n_targets  = y_train.shape[1]

    def to_3d(X):
        return X.reshape(len(X), window, n_features)

    # Datasets and loaders
    X_tr = torch.tensor(to_3d(X_train), dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_vl = torch.tensor(to_3d(X_val),   dtype=torch.float32)
    y_vl = torch.tensor(y_val,   dtype=torch.float32)
    X_ts = torch.tensor(to_3d(X_test),  dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader   = torch.utils.data.DataLoader(train_ds, batch_size=RNN_BATCH, shuffle=False)

    model = _RNNModel(n_features, n_targets, RNN_HIDDEN, RNN_LAYERS, model_type).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=RNN_LR)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    patience, no_improve = 10, 0

    for epoch in range(RNN_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_vl.to(device)), y_vl.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"{model_type.upper()} early stop at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            logger.debug(f"  Epoch {epoch+1}/{RNN_EPOCHS} val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val  = model(X_vl.to(device)).cpu().numpy()
        pred_test = model(X_ts.to(device)).cpu().numpy()

    from evaluation.metrics import rmse_per_target
    return BaselineResult(
        name        = model_type,
        y_pred_val  = pred_val,
        y_pred_test = pred_test,
        val_rmse    = rmse_per_target(y_val, pred_val),
        test_rmse   = rmse_per_target(y_test, pred_test),
        meta        = {"type": model_type, "hidden": RNN_HIDDEN, "layers": RNN_LAYERS},
    )