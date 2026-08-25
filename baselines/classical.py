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
from config import (ESN_HIDDEN_DIM, ESN_WARMUP, RNN_HIDDEN, RNN_LAYERS, RNN_EPOCHS,
                    RNN_LR, RNN_BATCH, RNN_WARMUP, RESULTS, RANDOM_SEED)
from utils import get_logger

logger = get_logger(__name__)
np.random.seed(RANDOM_SEED)

# Optional imports — guarded so the project loads even if not installed.
# ARIMA has TWO independent backends so a single fragile dependency (pmdarima
# frequently fails to build/import on fresh environments — Cython ABI
# mismatches, missing compiler on hosted notebooks like qBraid) can no longer
# silently remove the baseline from the benchmark:
#   1. pmdarima.auto_arima   — preferred, fast stepwise order search
#   2. statsmodels ARIMA     — fallback, small (p,d,q) grid search by AIC
# ARIMA is only truly skipped if NEITHER package is importable.
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA as _SM_ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

ARIMA_AVAILABLE = PMDARIMA_AVAILABLE or STATSMODELS_AVAILABLE

if PMDARIMA_AVAILABLE:
    logger.info("ARIMA backend: pmdarima (auto_arima, stepwise)")
elif STATSMODELS_AVAILABLE:
    logger.warning("ARIMA backend: statsmodels fallback (pmdarima not installed — "
                   "pip install pmdarima for faster stepwise search). "
                   "Using small (p,d,q) grid search by AIC instead.")
else:
    logger.error("ARIMA UNAVAILABLE — neither pmdarima nor statsmodels is installed. "
                "The ARIMA baseline will be SKIPPED and will NOT appear in "
                "results_h*.csv. Install: pip install statsmodels pmdarima")

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
    label_offset: int = 0   # rows to skip on y_true when scoring (reservoir warmup)

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

class _PmdarimaAdapter:
    """Wraps pmdarima's auto_arima result behind a minimal predict/update interface."""
    def __init__(self, model):
        self._model = model
        self.order  = model.order

    def predict(self, n: int = 1) -> np.ndarray:
        return np.asarray(self._model.predict(n))

    def update(self, obs: list):
        self._model.update(obs)


class _StatsmodelsAdapter:
    """
    Wraps a fitted statsmodels ARIMAResults behind the same predict/update
    interface as _PmdarimaAdapter, so run_arima's walk-forward loop is
    identical regardless of which backend fit the model.
    `.append(..., refit=False)` applies new observations without a full
    re-optimisation — the statsmodels equivalent of pmdarima's `.update()`.
    """
    def __init__(self, res, order: tuple):
        self._res = res
        self.order = order

    def predict(self, n: int = 1) -> np.ndarray:
        return np.asarray(self._res.forecast(steps=n))

    def update(self, obs: list):
        self._res = self._res.append(obs, refit=False)


def _fit_arima_statsmodels(train_clean: np.ndarray,
                            max_p: int = 3, max_d: int = 1,
                            max_q: int = 3) -> _StatsmodelsAdapter:
    """Small (p,d,q) grid search by AIC — no pmdarima dependency required."""
    best_aic, best_order, best_res = np.inf, None, None
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    res = _SM_ARIMA(train_clean, order=(p, d, q)).fit()
                    if res.aic < best_aic:
                        best_aic, best_order, best_res = res.aic, (p, d, q), res
                except Exception:
                    continue
    if best_res is None:
        raise RuntimeError("statsmodels ARIMA grid search found no converging (p,d,q)")
    return _StatsmodelsAdapter(best_res, best_order)


def run_arima(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
              target_names: list) -> Optional[BaselineResult]:
    """
    Per-target ARIMA. Prefers pmdarima (auto_arima) when installed; falls
    back to a small statsmodels grid search otherwise. Returns None only if
    NEITHER backend is available — this is logged at ERROR level (see
    ARIMA_AVAILABLE check above) precisely so a missing ARIMA row in the
    results table is never silent.
    """
    if not ARIMA_AVAILABLE:
        logger.error("Skipping ARIMA baseline — no backend available "
                     "(pip install statsmodels pmdarima). This baseline will "
                     "be ABSENT from results_h*.csv.")
        return None

    backend = "pmdarima" if PMDARIMA_AVAILABLE else "statsmodels"
    n_targets = y_train.shape[1]
    pred_val  = np.zeros_like(y_val)
    pred_test = np.zeros_like(y_test)
    orders    = {}
    failed_targets = []

    for t in range(n_targets):
        logger.info(f"  ARIMA[{backend}] fitting target: {target_names[t]}")
        try:
            # Drop NaNs from training series before fitting
            train_t = y_train[:, t]
            nan_mask = np.isnan(train_t)
            nan_pct = nan_mask.mean() * 100
            if nan_pct > 50:
                logger.warning(f"  Skipping ARIMA for {target_names[t]}: "
                                f"{nan_pct:.1f}% missing — too sparse to fit reliably")
                pred_val[:, t]  = np.nanmean(train_t)
                pred_test[:, t] = np.nanmean(train_t)
                continue
            train_clean = train_t[~nan_mask]

            if PMDARIMA_AVAILABLE:
                model = _PmdarimaAdapter(auto_arima(
                    train_clean, stepwise=True, seasonal=False,
                    error_action="ignore", suppress_warnings=True,
                    max_p=5, max_q=5))
            else:
                model = _fit_arima_statsmodels(train_clean)
            orders[target_names[t]] = str(model.order)

            # Walk-forward forecast on val — fill NaN targets with last prediction
            for i in range(len(y_val)):
                fc = model.predict(1)[0]
                pred_val[i, t] = fc
                obs = y_val[i, t]
                if not np.isnan(obs):
                    model.update([obs])
                if (i + 1) % 250 == 0 or i + 1 == len(y_val):
                    logger.info("  ARIMA[%s] validation progress: %d/%d",
                                target_names[t], i + 1, len(y_val))

            # Walk-forward on test
            for i in range(len(y_test)):
                fc = model.predict(1)[0]
                pred_test[i, t] = fc
                obs = y_test[i, t]
                if not np.isnan(obs):
                    model.update([obs])
                if (i + 1) % 250 == 0 or i + 1 == len(y_test):
                    logger.info("  ARIMA[%s] test progress: %d/%d",
                                target_names[t], i + 1, len(y_test))

        except Exception as e:
            logger.error(f"  ARIMA failed for {target_names[t]}: {e}")
            failed_targets.append(target_names[t])
            pred_val[:, t]  = np.nanmean(y_train[:, t])
            pred_test[:, t] = np.nanmean(y_train[:, t])

    from evaluation.metrics import rmse_per_target
    return BaselineResult(
        name        = "arima",
        y_pred_val  = pred_val,
        y_pred_test = pred_test,
        val_rmse    = rmse_per_target(y_val, pred_val),
        test_rmse   = rmse_per_target(y_test, pred_test),
        meta        = {"type": "arima", "backend": backend, "orders": orders,
                       "failed_targets": failed_targets},
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

    def _run_reservoir(self, X: np.ndarray, warmup: int = None) -> np.ndarray:
        """
        Drive ESN with X (N, n_features). Returns states (N-warmup, hidden_dim).

        warmup is clamped to N-1: predict()/get_reservoir_states() reuse the
        warmup fit() chose for X_train (ESN_WARMUP=50 by default), but val/test
        splits — or a --smoke_test truncated run — can be shorter than that.
        Without clamping, warmup >= N silently produces an EMPTY state array,
        and the caller's `H @ W_out_` then fails with an opaque matmul
        dimension-mismatch error instead of a clear one.
        """
        warmup = ESN_WARMUP if warmup is None else warmup
        N, n_in = X.shape
        if warmup >= N:
            clamped = max(0, N - 1)
            logger.warning(f"ESN warmup={warmup} >= sequence length {N} — "
                           f"clamping to {clamped} (short split or smoke test).")
            warmup = clamped
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
            warmup: int = None) -> "EchoStateNetwork":
        """Train linear readout on reservoir states via ridge regression."""
        warmup = ESN_WARMUP if warmup is None else warmup
        self._warmup = warmup
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

    def predict(self, X: np.ndarray, warmup: int = None) -> np.ndarray:
        warmup = getattr(self, "_warmup", ESN_WARMUP) if warmup is None else warmup
        H = self._run_reservoir(X, warmup)
        return H @ self.W_out_

    def get_reservoir_states(self, X: np.ndarray, warmup: int = None) -> np.ndarray:
        """Return raw reservoir states — aligned with fit() transient discard."""
        warmup = getattr(self, "_warmup", ESN_WARMUP) if warmup is None else warmup
        return self._run_reservoir(X, warmup)

    def get_readout_weights(self) -> np.ndarray:
        """Return trained W_out for use as QRC warm-start."""
        if self.W_out_ is None:
            raise RuntimeError("ESN not fitted yet. Call fit() first.")
        return self.W_out_


def _effective_warmup(n_samples: int, warmup: int) -> int:
    """Mirrors EchoStateNetwork._run_reservoir's internal clamp exactly, so
    callers can correctly trim y_true to match a (possibly clamped) predict()
    output length instead of assuming the nominal, unclamped warmup applies."""
    return warmup if warmup < n_samples else max(0, n_samples - 1)


def run_esn(X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray,
            hidden_dim: int = None,
            warmup: int = None) -> tuple[BaselineResult, EchoStateNetwork]:
    """Fit and evaluate ESN. Returns (result, fitted_esn) — esn reused for warm-start."""
    warmup = ESN_WARMUP if warmup is None else warmup
    esn = EchoStateNetwork(hidden_dim=hidden_dim or ESN_HIDDEN_DIM)
    esn.fit(X_train, y_train, warmup=warmup)

    pred_val  = esn.predict(X_val, warmup=warmup)
    pred_test = esn.predict(X_test, warmup=warmup)
    # NOT y_val[warmup:] / y_test[warmup:] — predict() clamps its internal
    # warmup to each split's own length (see EchoStateNetwork._run_reservoir),
    # so trimming y_true by the nominal, unclamped `warmup` silently produces
    # a length-mismatched (often EMPTY) array whenever a split is shorter
    # than ESN_WARMUP — exactly the kind of short split --smoke_test produces.
    w_val  = _effective_warmup(len(X_val), warmup)
    w_test = _effective_warmup(len(X_test), warmup)
    y_val_a   = y_val[w_val:]
    y_test_a  = y_test[w_test:]
    n_val  = min(len(y_val_a), len(pred_val))
    n_test = min(len(y_test_a), len(pred_test))
    y_val_a,  pred_val  = y_val_a[:n_val],   pred_val[:n_val]
    y_test_a, pred_test = y_test_a[:n_test], pred_test[:n_test]

    from evaluation.metrics import rmse_per_target
    result = BaselineResult(
        name        = "esn",
        y_pred_val  = pred_val,
        y_pred_test = pred_test,
        val_rmse    = rmse_per_target(y_val_a, pred_val),
        test_rmse   = rmse_per_target(y_test_a, pred_test),
        meta        = {"type": "esn", "hidden_dim": esn.hidden_dim, "warmup": warmup},
        label_offset= w_val,
    )
    return result, esn


# ─────────────────────────────────────────────────────────────────────────────
# 4. LSTM / GRU — genuine streaming sequence models, swappable via model_type
#
# Earlier versions treated every window/PCA row as an INDEPENDENT training
# example (hidden state reset to zero every sample, every batch). Under
# USE_SHARED_PCA (the project's own default) that collapsed the RNN into a
# length-1-sequence feedforward transform with ZERO cross-sample memory —
# see docs/PROJECT_CRITIQUE.md §3.1. This version instead walks the full
# ordered sequence of samples exactly like EchoStateNetwork._run_reservoir /
# IsingQRC.run_sequence: one continuous pass with a persistent hidden state,
# an initial warmup/washout discarded before scoring (RNN_WARMUP, same role
# as ESN_WARMUP), and — for training — truncated backprop-through-time
# (chunk length RNN_BATCH) so there's no single impossibly long backward
# graph. LSTM/GRU are now a true reservoir-style peer to ESN/QRC rather than
# a differently-shaped model being compared unfairly.
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

        def forward(self, x, h0=None):
            """x: (1, seq_len, n_features) — one continuous ordered stream (batch=1)."""
            out, h = self.rnn(x, h0)
            return self.fc(out), out, h

        @staticmethod
        def detach_state(h):
            """Truncate the BPTT graph at a chunk boundary without losing the state value."""
            if isinstance(h, tuple):
                return tuple(t.detach() for t in h)
            return h.detach()


class RNNWarmStartExtractor:
    """
    Wraps a fitted LSTM/GRU so its hidden-state representation can be reused
    to warm-start the QRC ridge readout, analogous to
    EchoStateNetwork.get_reservoir_states(): walks X as ONE continuous
    ordered sequence from h=0, discards the warmup washout, returns the
    remaining per-step hidden states. Carries genuine cross-sample memory in
    every mode (including under USE_SHARED_PCA), unlike the earlier
    per-sample-reset design.
    """
    def __init__(self, model, device, warmup: int):
        self.model  = model
        self.device = device
        self.warmup = warmup

    def get_hidden_states(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        warmup = min(self.warmup, max(0, len(X) - 1))
        with torch.no_grad():
            x = torch.tensor(X[np.newaxis, :, :], dtype=torch.float32).to(self.device)
            _, out, _ = self.model(x)
            return out[0, warmup:, :].cpu().numpy().astype(np.float32)


def run_rnn(X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray,
            X_test:  np.ndarray, y_test:  np.ndarray,
            window: int = 20,
            model_type: Literal["lstm", "gru"] = "lstm"
            ) -> tuple[Optional[BaselineResult], Optional["RNNWarmStartExtractor"]]:
    """
    Train and evaluate LSTM or GRU as a streaming sequence model — one
    continuous ordered pass over X_train (batch=1), truncated BPTT with
    chunk length RNN_BATCH, hidden state carried across chunks within an
    epoch and reset to zero at the start of each epoch (mirrors
    EchoStateNetwork's "always start at h=0, use warmup to washout"
    convention). `window` is accepted for interface parity with the prior
    signature but is no longer used — the model receives one raw feature
    vector per ordered sample and derives memory from its own recurrence
    across samples, exactly like ESN/QRC, not from unfolding a window.

    Returns (result, warm_start_extractor) — extractor is None if PyTorch
    is unavailable. Mirrors run_esn()'s (result, fitted_esn) contract so
    WARM_START_SOURCE can point at "esn", "lstm", or "gru" interchangeably.
    """
    if not TORCH_AVAILABLE:
        logger.warning(f"Skipping {model_type.upper()} — PyTorch not available")
        return None, None

    import torch
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_targets  = y_train.shape[1]
    n_features = X_train.shape[1]
    warmup = min(RNN_WARMUP, max(0, len(X_train) - 1))
    if warmup != RNN_WARMUP:
        logger.warning(f"{model_type.upper()} warmup={RNN_WARMUP} >= train length "
                       f"{len(X_train)} — clamped to {warmup}.")
    logger.info(f"Training {model_type.upper()} on {device} "
               f"(streaming, chunk={RNN_BATCH}, warmup={warmup})")

    X_tr = torch.tensor(X_train[np.newaxis, :, :], dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train[np.newaxis, :, :], dtype=torch.float32).to(device)
    X_vl = torch.tensor(X_val[np.newaxis, :, :],   dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val[np.newaxis, :, :],   dtype=torch.float32).to(device)
    X_ts = torch.tensor(X_test[np.newaxis, :, :],  dtype=torch.float32).to(device)

    model = _RNNModel(n_features, n_targets, RNN_HIDDEN, RNN_LAYERS, model_type).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=RNN_LR)
    loss_fn = nn.MSELoss()

    n_train = X_tr.shape[1]
    chunk = max(1, RNN_BATCH)

    def _eval_sequence(X_seq, y_seq, w):
        """Fresh forward pass from h=0 over a full sequence; loss past warmup w."""
        model.eval()
        with torch.no_grad():
            pred, _, _ = model(X_seq)
            w_eff = min(w, max(0, X_seq.shape[1] - 1))
            return loss_fn(pred[:, w_eff:, :], y_seq[:, w_eff:, :]).item()

    best_val_loss = float("inf")
    # Initialise immediately so early-stop on epoch 1 never leaves this as None
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience, no_improve = 10, 0

    for epoch in range(RNN_EPOCHS):
        model.train()
        h = None
        for start in range(0, n_train, chunk):
            end = min(start + chunk, n_train)
            x_chunk = X_tr[:, start:end, :]
            y_chunk = y_tr[:, start:end, :]

            opt.zero_grad()
            pred_chunk, _, h = model(x_chunk, h)
            h = model.detach_state(h)   # truncate BPTT at the chunk boundary

            # Skip loss entirely for steps still inside the washout period
            mask_from = max(0, warmup - start)
            if mask_from < (end - start):
                loss = loss_fn(pred_chunk[:, mask_from:, :], y_chunk[:, mask_from:, :])
                loss.backward()
                opt.step()

        val_loss = _eval_sequence(X_vl, y_vl, warmup)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"{model_type.upper()} early stop at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            logger.info(f"  {model_type.upper()} epoch {epoch+1}/{RNN_EPOCHS} "
                        f"val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val_full,  _, _ = model(X_vl)
        pred_test_full, _, _ = model(X_ts)
    w_val  = min(warmup, max(0, X_vl.shape[1] - 1))
    w_test = min(warmup, max(0, X_ts.shape[1] - 1))
    pred_val  = pred_val_full[0, w_val:, :].cpu().numpy()
    pred_test = pred_test_full[0, w_test:, :].cpu().numpy()
    y_val_a  = y_val[w_val:]
    y_test_a = y_test[w_test:]

    from evaluation.metrics import rmse_per_target
    result = BaselineResult(
        name        = model_type,
        y_pred_val  = pred_val,
        y_pred_test = pred_test,
        val_rmse    = rmse_per_target(y_val_a, pred_val),
        test_rmse   = rmse_per_target(y_test_a, pred_test),
        meta        = {"type": model_type, "hidden": RNN_HIDDEN, "layers": RNN_LAYERS,
                       "warmup": warmup, "streaming": True},
        label_offset= warmup,
    )
    extractor = RNNWarmStartExtractor(model, device, warmup)
    return result, extractor
