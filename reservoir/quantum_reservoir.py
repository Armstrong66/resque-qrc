"""
reservoir/quantum_reservoir.py — Transverse-Field Ising QRC via PennyLane.

Design decisions documented in config.py:
  H = -J ΣZᵢZⱼ - h ΣXᵢ
  Justification: (1) maps directly to QuEra Aquila Rydberg blockade Hamiltonian
                 (2) J controls nonlinearity, h controls memory — matches
                     Čindrak et al. 2026 tradeoff framework
                 (3) Kornjača et al. 2024 ran 108-qubit QRC on this hardware

Outputs 2*n_qubits expectation values per step [⟨σᶻ⟩, ⟨σˣ⟩].
Feedback: ⟨σᶻ⟩ from t-1 concatenated with input before encoding (Zhu et al. 2025).
"""

import sys
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (J_DEFAULT, H_DEFAULT, QUBIT_PRIMARY, TOPOLOGY_PRIMARY,
                    TROTTER_STEPS, EVOLUTION_TIME, OBSERVABLES, USE_FEEDBACK,
                    NOISE_RATES, NOISE_TYPE, RANDOM_SEED)
from utils import get_logger

logger = get_logger(__name__)

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not installed. Install: pip install pennylane")


def _get_device(n_qubits: int, noise_rate: float = 0.0) -> "qml.Device":
    """
    Return a PennyLane device.
    - noise_rate == 0 → exact state-vector (default.qubit)
    - noise_rate  > 0 → density-matrix simulator (default.mixed)
    """
    if not PENNYLANE_AVAILABLE:
        raise ImportError("PennyLane required. pip install pennylane")
    if noise_rate > 0.0:
        return qml.device("default.mixed", wires=n_qubits)
    return qml.device("default.qubit", wires=n_qubits)


def _ising_evolution_circuit(n_qubits: int,
                              input_vec: np.ndarray,
                              prev_zexp: Optional[np.ndarray],
                              J: float,
                              h: float,
                              topology: str,
                              trotter_steps: int,
                              evolution_time: float,
                              noise_rate: float,
                              shots: Optional[int]):
    """
    Build and return a PennyLane QNode for one reservoir step.
    Angle encoding: Ry(π * xᵢ) per qubit.
    Feedback: if prev_zexp provided, concatenate with input before encoding.
    Trotter evolution of H = -J ΣZᵢZⱼ - h ΣXᵢ.
    Measure: ⟨σᶻᵢ⟩ and ⟨σˣᵢ⟩ for all qubits.
    """
    dev = _get_device(n_qubits, noise_rate)
    obs_list = (
        [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] +
        [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    )

    # Augment input with feedback
    if prev_zexp is not None and USE_FEEDBACK:
        enc_input = np.concatenate([input_vec, prev_zexp])
    else:
        enc_input = input_vec

    # Pad or truncate enc_input to n_qubits (angle encoding 1 angle per qubit)
    enc_angles = np.resize(enc_input, n_qubits) * np.pi

    # Get coupling pairs based on topology
    if topology == "chain":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    elif topology == "all_to_all":
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    else:
        raise ValueError(f"Unknown topology: {topology}. Choose 'chain' or 'all_to_all'.")

    @qml.qnode(dev, diff_method=None)    # No gradients needed — analytical readout
    def circuit():
        # ── Encoding ────────────────────────────────────────────────────────
        for i in range(n_qubits):
            qml.RY(enc_angles[i], wires=i)

        # ── Trotter evolution of Ising Hamiltonian ───────────────────────────
        dt = evolution_time / trotter_steps
        for _ in range(trotter_steps):
            # ZZ couplings
            for (i, j) in pairs:
                qml.IsingZZ(-2 * J * dt, wires=[i, j])
            # Transverse field (X rotations)
            for i in range(n_qubits):
                qml.RX(-2 * h * dt, wires=i)

        # ── Noise (depolarizing, applied after evolution) ────────────────────
        if noise_rate > 0.0:
            for i in range(n_qubits):
                qml.DepolarizingChannel(noise_rate, wires=i)

        return obs_list

    return circuit()


class IsingQRC:
    """
    Transverse-Field Ising Quantum Reservoir Computer.

    Usage:
        reservoir = IsingQRC(n_qubits=9, J=1.0, h=0.5)
        states = reservoir.run_sequence(X_seq)   # (T, 2*n_qubits)

    X_seq: (T, n_features) — time series of input vectors
    """

    def __init__(self,
                 n_qubits:      int   = None,
                 J:             float = None,
                 h:             float = None,
                 topology:      str   = None,
                 trotter_steps: int   = None,
                 evolution_time:float = None,
                 noise_rate:    float = 0.0,
                 shots:         Optional[int] = None,
                 use_feedback:  bool  = None):

        self.n_qubits       = n_qubits       or QUBIT_PRIMARY
        self.J              = J              or J_DEFAULT
        self.h              = h              or H_DEFAULT
        self.topology       = topology       or TOPOLOGY_PRIMARY
        self.trotter_steps  = trotter_steps  or TROTTER_STEPS
        self.evolution_time = evolution_time or EVOLUTION_TIME
        self.noise_rate     = noise_rate
        self.shots          = shots
        self.use_feedback   = use_feedback if use_feedback is not None else USE_FEEDBACK

        self.feature_dim = 2 * self.n_qubits   # Z + X expectations

        logger.info(
            f"IsingQRC | qubits={self.n_qubits} J={self.J} h={self.h} "
            f"topology={self.topology} noise={self.noise_rate} "
            f"feedback={self.use_feedback}"
        )

    def _step(self, input_vec: np.ndarray,
              prev_zexp: Optional[np.ndarray]) -> np.ndarray:
        """Run one reservoir step. Returns feature vector of length 2*n_qubits."""
        results = _ising_evolution_circuit(
            n_qubits       = self.n_qubits,
            input_vec      = input_vec,
            prev_zexp      = prev_zexp,
            J              = self.J,
            h              = self.h,
            topology       = self.topology,
            trotter_steps  = self.trotter_steps,
            evolution_time = self.evolution_time,
            noise_rate     = self.noise_rate,
            shots          = self.shots,
        )
        return np.array(results, dtype=np.float32)

    def run_sequence(self, X_seq: np.ndarray,
                     warmup: int = 20,
                     verbose: bool = False) -> np.ndarray:
        """
        Drive reservoir with time-series X_seq of shape (T, n_features).
        Returns reservoir state matrix of shape (T - warmup, 2*n_qubits).

        warmup: initial steps discarded to let reservoir reach attractor.
        """
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane required.")

        T = len(X_seq)
        if T <= warmup:
            raise ValueError(f"Sequence length {T} ≤ warmup {warmup}")

        states = []
        prev_zexp = None

        for t in range(T):
            state = self._step(X_seq[t], prev_zexp)
            if self.use_feedback:
                prev_zexp = state[:self.n_qubits]   # Only Z expectations for feedback
            if t >= warmup:
                states.append(state)
            if verbose and t % 100 == 0:
                logger.debug(f"  Reservoir step {t}/{T}")

        result = np.array(states, dtype=np.float32)
        logger.debug(f"run_sequence: input {X_seq.shape} → states {result.shape}")
        return result

    def get_config(self) -> dict:
        """Return serialisable config dict for logging/reproducibility."""
        return {
            "n_qubits":       self.n_qubits,
            "J":              self.J,
            "h":              self.h,
            "topology":       self.topology,
            "trotter_steps":  self.trotter_steps,
            "evolution_time": self.evolution_time,
            "noise_rate":     self.noise_rate,
            "shots":          self.shots,
            "use_feedback":   self.use_feedback,
            "feature_dim":    self.feature_dim,
        }


def run_qubit_sweep(X_seq: np.ndarray,
                    qubit_counts: list = None,
                    J: float = None,
                    h: float = None,
                    **kwargs) -> dict[int, np.ndarray]:
    """
    Run reservoir across all configured qubit counts.
    Returns dict: {n_qubits: state_matrix}.
    Used by the qubit scaling study required by the challenge.
    """
    from config import QUBIT_COUNTS
    qubit_counts = qubit_counts or QUBIT_COUNTS
    J = J or J_DEFAULT
    h = h or H_DEFAULT

    results = {}
    for n in qubit_counts:
        logger.info(f"Running qubit sweep: n={n}")
        qrc = IsingQRC(n_qubits=n, J=J, h=h, **kwargs)
        results[n] = qrc.run_sequence(X_seq)
    return results