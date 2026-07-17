"""
reservoir/quantum_reservoir.py — Transverse-Field Ising QRC via PennyLane.

Two encoding strategies implemented and switchable via USE_DATA_REUPLOADING:

  Standard (USE_DATA_REUPLOADING=False):
      Encode input ONCE before all Trotter steps.
      Each qubit gets one Ry(pi*x_i) rotation, then H evolves freely.
      Expressivity limited by single injection.

  Data Reuploading (USE_DATA_REUPLOADING=True):
      Re-encode input before EACH Trotter step (Pérez-Salinas et al. 2020).
      Pattern: [Encode → ZZ+X evolution] × trotter_steps
      Increases effective expressivity without adding qubits.
      This is the primary Phase 3 intervention to close the QRC–ESN gap.
      Challenge design space doc explicitly lists "repeated encoding strategies".

Both strategies are benchmarked in the qubit scaling and encoding ablation studies.
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (J_DEFAULT, H_DEFAULT, QUBIT_PRIMARY, TOPOLOGY_PRIMARY,
                    TROTTER_STEPS, EVOLUTION_TIME, USE_FEEDBACK,
                    RANDOM_SEED)
from utils import get_logger

logger = get_logger(__name__)

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
    logger.info(f"PennyLane {qml.__version__} loaded")
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not installed. pip install pennylane pennylane-lightning")

# ── Device factory ────────────────────────────────────────────────────────────

def _get_device(n_qubits: int, noise_rate: float = 0.0, shots: Optional[int] = None):
    """
    Select PennyLane device based on noise and available backends.
    lightning.qubit is used when available for GPU acceleration at n >= 9.
    default.mixed is required for noise channels.
    """
    if not PENNYLANE_AVAILABLE:
        raise ImportError("PennyLane required.")
    if noise_rate > 0.0:
        return qml.device("default.mixed", wires=n_qubits, shots=shots)
    # Try GPU-accelerated simulator first (qBraid has lightning.qubit available)
    try:
        import pennylane_lightning  # noqa
        dev = qml.device("lightning.qubit", wires=n_qubits, shots=shots)
        logger.debug(f"Using lightning.qubit (GPU) for n={n_qubits}")
        return dev
    except Exception:
        return qml.device("default.qubit", wires=n_qubits, shots=shots)


# ── Coupling pairs ─────────────────────────────────────────────────────────────

def _get_pairs(n_qubits: int, topology: str) -> list:
    if topology == "chain":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    elif topology == "all_to_all":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    else:
        raise ValueError(f"Unknown topology: {topology}. Choose 'chain' or 'all_to_all'.")


# ── Core circuit builders ──────────────────────────────────────────────────────

def _encode_block(enc_angles: np.ndarray, n_qubits: int):
    """Single encoding block: Ry(pi*x_i) per qubit."""
    for i in range(n_qubits):
        qml.RY(enc_angles[i], wires=i)


def _evolution_block(pairs: list, n_qubits: int, J: float, h: float, dt: float,
                     noise_rate: float):
    """Single Trotter step: ZZ couplings then X rotations, then optional noise."""
    for (i, j) in pairs:
        qml.IsingZZ(-2 * J * dt, wires=[i, j])
    for i in range(n_qubits):
        qml.RX(-2 * h * dt, wires=i)
    if noise_rate > 0.0:
        for i in range(n_qubits):
            qml.DepolarizingChannel(noise_rate, wires=i)


def _build_circuit(n_qubits, enc_angles, pairs, J, h, trotter_steps,
                   evolution_time, noise_rate, use_data_reuploading):
    """
    Build circuit body — called inside @qml.qnode decorated function.

    Standard:          encode once → [ZZ+X] × trotter_steps
    Data reuploading:  [encode → ZZ+X] × trotter_steps
    """
    dt = evolution_time / trotter_steps

    if use_data_reuploading:
        # Re-encode before every Trotter step (Pérez-Salinas et al. 2020)
        for _ in range(trotter_steps):
            _encode_block(enc_angles, n_qubits)
            _evolution_block(pairs, n_qubits, J, h, dt, noise_rate)
    else:
        # Standard: single injection
        _encode_block(enc_angles, n_qubits)
        for _ in range(trotter_steps):
            _evolution_block(pairs, n_qubits, J, h, dt, noise_rate)

    # Measure Z and X expectation values on all qubits (inside circuit body — PL >=0.38)
    return (
        [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] +
        [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    )


def _run_circuit(n_qubits, input_vec, prev_zexp, J, h, topology,
                 trotter_steps, evolution_time, noise_rate, shots,
                 use_data_reuploading, use_feedback) -> np.ndarray:
    """
    Build device, prepare input (with NaN guard + feedback), run circuit.
    Returns feature vector of length 2*n_qubits.
    """
    dev = _get_device(n_qubits, noise_rate, shots)
    pairs = _get_pairs(n_qubits, topology)

    # Feedback: concatenate prior Z expectations with current input
    if prev_zexp is not None and use_feedback:
        enc_input = np.concatenate([input_vec, prev_zexp])
    else:
        enc_input = input_vec

    # NaN guard — missing station observations produce NaN angles otherwise
    if np.any(np.isnan(enc_input)):
        enc_input = np.where(np.isnan(enc_input), 0.0, enc_input)

    # Resize to n_qubits after shared PCA ensures fair comparison
    enc_angles = np.resize(enc_input, n_qubits) * np.pi

    @qml.qnode(dev, diff_method=None)   # No gradients — analytical readout
    def circuit():
        return _build_circuit(n_qubits, enc_angles, pairs, J, h,
                              trotter_steps, evolution_time, noise_rate,
                              use_data_reuploading)

    return np.array(circuit(), dtype=np.float32)


# ── Main reservoir class ───────────────────────────────────────────────────────

class IsingQRC:
    """
    Transverse-Field Ising Quantum Reservoir Computer.

    Key parameters
    --------------
    n_qubits          : Reservoir size. Challenge requires sweep over {5,7,9,12,16,20}.
    J, h              : Ising coupling and transverse field. Selected by Hamiltonian sweep.
    topology          : 'chain' (sparse, high memory) or 'all_to_all' (dense, high NL).
    noise_rate        : Depolarizing noise p ∈ [0, 0.05]. Sweep determines optimal p*.
    shots             : None = exact simulation; int = finite-shot (hardware-realistic).
    use_data_reuploading : True = re-encode at every Trotter step (Phase 3 primary fix).
    use_feedback      : True = inject prior <Z> into next input (Zhu et al. 2025).

    Usage
    -----
    qrc = IsingQRC(n_qubits=9, J=0.3, h=0.8, use_data_reuploading=True)
    H = qrc.run_sequence(X_train)              # full training run
    H = qrc.run_sequence(X_train, max_steps=300)   # sweep mode (fast)
    """

    def __init__(self,
                 n_qubits: int = None,
                 J: float = None,
                 h: float = None,
                 topology: str = None,
                 trotter_steps: int = None,
                 evolution_time: float = None,
                 noise_rate: float = 0.0,
                 shots: Optional[int] = None,
                 use_feedback: bool = None,
                 use_data_reuploading: bool = False):

        self.n_qubits            = n_qubits       or QUBIT_PRIMARY
        self.J                   = J              if J is not None else J_DEFAULT
        self.h                   = h              if h is not None else H_DEFAULT
        self.topology            = topology       or TOPOLOGY_PRIMARY
        self.trotter_steps       = trotter_steps  or TROTTER_STEPS
        self.evolution_time      = evolution_time or EVOLUTION_TIME
        self.noise_rate          = noise_rate
        self.shots               = shots
        self.use_feedback        = use_feedback if use_feedback is not None else USE_FEEDBACK
        self.use_data_reuploading = use_data_reuploading
        self.feature_dim         = 2 * self.n_qubits

        encoding_label = "data-reuploading" if use_data_reuploading else "standard"
        logger.info(
            f"IsingQRC | n={self.n_qubits} J={self.J} h={self.h} "
            f"topo={self.topology} noise={self.noise_rate} "
            f"shots={self.shots} encoding={encoding_label} feedback={self.use_feedback}"
        )

    def _step(self, input_vec: np.ndarray, prev_zexp: Optional[np.ndarray]) -> np.ndarray:
        return _run_circuit(
            n_qubits             = self.n_qubits,
            input_vec            = input_vec,
            prev_zexp            = prev_zexp,
            J                    = self.J,
            h                    = self.h,
            topology             = self.topology,
            trotter_steps        = self.trotter_steps,
            evolution_time       = self.evolution_time,
            noise_rate           = self.noise_rate,
            shots                = self.shots,
            use_data_reuploading = self.use_data_reuploading,
            use_feedback         = self.use_feedback,
        )

    def run_sequence(self,
                     X_seq: np.ndarray,
                     warmup: int = 20,
                     max_steps: Optional[int] = None,
                     verbose: bool = False) -> np.ndarray:
        """
        Drive reservoir with input sequence X_seq of shape (T, n_features).
        Returns state matrix of shape (min(T-warmup, max_steps), 2*n_qubits).

        max_steps : cap for sweep mode (~300 steps); None = full sequence.
                    Sweeps use max_steps to keep each config to ~minutes not hours.
        """
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane required.")

        T = len(X_seq)
        if T <= warmup:
            raise ValueError(f"Sequence length {T} <= warmup {warmup}. "
                             f"Increase data or reduce warmup.")

        cap = (warmup + max_steps) if max_steps is not None else T
        cap = min(cap, T)

        states, prev_zexp = [], None
        t0 = time.time()

        for t in range(cap):
            state = self._step(X_seq[t], prev_zexp)

            # Guard: if circuit returns NaN, log and substitute zeros
            if np.any(np.isnan(state)):
                logger.warning(f"NaN in reservoir state at step {t} — "
                               f"substituting zeros. Check input data.")
                state = np.zeros(self.feature_dim, dtype=np.float32)

            if self.use_feedback:
                prev_zexp = state[:self.n_qubits]
            if t >= warmup:
                states.append(state)
            if verbose and t % 50 == 0:
                elapsed = time.time() - t0
                eta = (elapsed / max(t, 1)) * (cap - t)
                logger.info(f"  Reservoir step {t}/{cap} "
                            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        result = np.array(states, dtype=np.float32)
        encoding = "reupload" if self.use_data_reuploading else "standard"
        logger.info(f"run_sequence complete: {X_seq.shape} → {result.shape} "
                    f"[{encoding}, n={self.n_qubits}, "
                    f"max_steps={'full' if max_steps is None else max_steps}]")
        return result

    def get_config(self) -> dict:
        """Serialisable config for logging and qBraid Skill reproducibility."""
        return {
            "n_qubits":             self.n_qubits,
            "J":                    self.J,
            "h":                    self.h,
            "topology":             self.topology,
            "trotter_steps":        self.trotter_steps,
            "evolution_time":       self.evolution_time,
            "noise_rate":           self.noise_rate,
            "shots":                self.shots,
            "use_feedback":         self.use_feedback,
            "use_data_reuploading": self.use_data_reuploading,
            "feature_dim":          self.feature_dim,
        }


# ── Convenience sweep runner ───────────────────────────────────────────────────

def run_qubit_sweep(X_seq: np.ndarray,
                    qubit_counts: list = None,
                    J: float = None,
                    h: float = None,
                    use_data_reuploading: bool = False,
                    **kwargs) -> dict:
    """
    Run reservoir across all configured qubit counts.
    Returns {n_qubits: state_matrix}.
    Challenge requires characterising performance over n = 5 → 20.
    """
    from config import QUBIT_COUNTS
    qubit_counts = qubit_counts or QUBIT_COUNTS
    J = J if J is not None else J_DEFAULT
    h = h if h is not None else H_DEFAULT

    results = {}
    for n in qubit_counts:
        logger.info(f"Qubit sweep: n={n}")
        qrc = IsingQRC(n_qubits=n, J=J, h=h,
                       use_data_reuploading=use_data_reuploading, **kwargs)
        results[n] = qrc.run_sequence(X_seq, **kwargs)
    return results


# ── Encoding ablation: standard vs data reuploading ───────────────────────────

def encoding_ablation(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       n_qubits: int = 9, J: float = None, h: float = None,
                       max_steps: int = 300,
                       out_dir: Path = None) -> dict:
    """
    Compare standard vs data-reuploading encoding on the same reservoir config.
    This is the primary Phase 3 diagnostic experiment.
    Returns dict with RMSE for each encoding type.
    """
    import json
    from readout.ridge_readout import _ridge_solve, _rmse_per_target

    J = J if J is not None else J_DEFAULT
    h = h if h is not None else H_DEFAULT
    results = {}

    for reupload in [False, True]:
        label = "data_reuploading" if reupload else "standard"
        logger.info(f"Encoding ablation: {label}")
        try:
            qrc = IsingQRC(n_qubits=n_qubits, J=J, h=h,
                           use_data_reuploading=reupload)
            H_tr = qrc.run_sequence(X_train, warmup=20, max_steps=max_steps)
            H_vl = qrc.run_sequence(X_val,   warmup=0,  max_steps=max_steps)

            n_tr = min(len(H_tr), len(y_train))
            n_vl = min(len(H_vl), len(y_val))
            W    = _ridge_solve(H_tr[:n_tr], y_train[:n_tr], lambda_=1e-3)
            pred = H_vl[:n_vl] @ W
            rmse = float(_rmse_per_target(y_val[:n_vl], pred).mean())
            results[label] = rmse
            logger.info(f"  {label}: val_rmse={rmse:.4f}")
        except Exception as e:
            logger.error(f"  {label} failed: {e}")
            results[label] = float("inf")

    # Log comparative result clearly
    if "standard" in results and "data_reuploading" in results:
        diff = results["standard"] - results["data_reuploading"]
        if diff > 0:
            logger.info(f"Data reuploading IMPROVES RMSE by {diff:.4f} "
                        f"(standard={results['standard']:.4f} → "
                        f"reupload={results['data_reuploading']:.4f})")
        else:
            logger.info(f"Standard encoding matches or beats reuploading "
                        f"(diff={diff:.4f}). Single injection sufficient at this scale.")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "encoding_ablation.json", "w") as f:
            json.dump(results, f, indent=2)

    return results
