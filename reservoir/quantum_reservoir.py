"""
reservoir/quantum_reservoir.py — Transverse-Field Ising QRC via PennyLane.

Design decisions documented in config.py:
  H = -J ΣZᵢZⱼ - h ΣXᵢ
  Outputs 2*n_qubits expectation values per step [⟨σᶻ⟩, ⟨σˣ⟩].
  Feedback: blend prior ⟨Z⟩ with PCA input when dimensions match (Zhu et al. 2025).
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (J_DEFAULT, H_DEFAULT, QUBIT_PRIMARY, TOPOLOGY_PRIMARY,
                    TROTTER_STEPS, EVOLUTION_TIME, USE_FEEDBACK, QRC_WARMUP,
                    RANDOM_SEED)
from utils import get_logger

logger = get_logger(__name__)

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not installed. Install: pip install pennylane")


def _get_device(n_qubits: int, noise_rate: float, shots: Optional[int],
                platform: str = "local"):
    """State-vector or mixed simulator; finite shots when shots is set."""
    if not PENNYLANE_AVAILABLE:
        raise ImportError("PennyLane required. pip install pennylane")
    if noise_rate > 0.0:
        return qml.device("default.mixed", wires=n_qubits)
    if shots is not None and shots > 0:
        return qml.device("default.qubit", wires=n_qubits, shots=int(shots))
    if platform == "qbraid":
        try:
            return qml.device("lightning.qubit", wires=n_qubits)
        except Exception:
            logger.debug("lightning.qubit unavailable on qBraid; using default.qubit")
    return qml.device("default.qubit", wires=n_qubits)


def _encoding_angles(input_vec: np.ndarray,
                     prev_zexp: Optional[np.ndarray],
                     n_qubits: int,
                     use_feedback: bool) -> np.ndarray:
    """
    Map input to n_qubits rotation angles in [-π, π].
    Expects len(input_vec) == n_qubits when using PCA (no cyclic resize).
    """
    x = np.asarray(input_vec, dtype=np.float64).ravel()
    if len(x) < n_qubits:
        x = np.resize(x, n_qubits)
    elif len(x) > n_qubits:
        x = x[:n_qubits]

    if prev_zexp is not None and use_feedback:
        z = np.asarray(prev_zexp, dtype=np.float64).ravel()[:n_qubits]
        if len(z) < n_qubits:
            z = np.resize(z, n_qubits)
        x = 0.5 * (x + z)

    return np.clip(x * np.pi, -np.pi, np.pi)


class IsingQRC:
    """
    Transverse-Field Ising Quantum Reservoir Computer.

    Usage:
        reservoir = IsingQRC(n_qubits=9, J=1.0, h=0.5)
        states = reservoir.run_sequence(X_seq)   # (T - warmup, 2*n_qubits)
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
                 use_feedback:  bool  = None,
                 platform:      str   = "local"):

        self.platform       = platform
        self.n_qubits       = n_qubits       or QUBIT_PRIMARY
        self.J              = J if J is not None else J_DEFAULT
        self.h              = h if h is not None else H_DEFAULT
        self.topology       = topology       or TOPOLOGY_PRIMARY
        self.trotter_steps  = trotter_steps  or TROTTER_STEPS
        self.evolution_time = evolution_time or EVOLUTION_TIME
        self.noise_rate     = noise_rate
        self.shots          = shots
        self.use_feedback   = use_feedback if use_feedback is not None else USE_FEEDBACK
        self.feature_dim    = 2 * self.n_qubits
        self._pairs         = self._coupling_pairs(self.topology, self.n_qubits)
        self._qnode         = self._build_qnode()

        logger.info(
            f"IsingQRC | qubits={self.n_qubits} J={self.J} h={self.h} "
            f"topology={self.topology} noise={self.noise_rate} "
            f"shots={self.shots if self.shots else 'exact'} "
            f"feedback={self.use_feedback}"
        )

    @staticmethod
    def _coupling_pairs(topology: str, n_qubits: int = None) -> list:
        n = n_qubits
        if topology == "chain":
            return [(i, i + 1) for i in range(n - 1)]
        if topology == "all_to_all":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        raise ValueError(f"Unknown topology: {topology}")

    def _build_qnode(self):
        n = self.n_qubits
        pairs = self._pairs
        J, h_field = self.J, self.h
        trotter_steps = self.trotter_steps
        evolution_time = self.evolution_time
        noise_rate = self.noise_rate
        dev = _get_device(n, noise_rate, self.shots, self.platform)

        @qml.qnode(dev, diff_method=None)
        def circuit(angles):
            for i in range(n):
                qml.RY(float(angles[i]), wires=i)

            dt = evolution_time / trotter_steps
            for _ in range(trotter_steps):
                for (i, j) in pairs:
                    qml.IsingZZ(-2 * J * dt, wires=[i, j])
                for i in range(n):
                    qml.RX(-2 * h_field * dt, wires=i)

            if noise_rate > 0.0:
                for i in range(n):
                    qml.DepolarizingChannel(noise_rate, wires=i)

            return (
                [qml.expval(qml.PauliZ(i)) for i in range(n)] +
                [qml.expval(qml.PauliX(i)) for i in range(n)]
            )

        return circuit

    def _step(self, input_vec: np.ndarray,
              prev_zexp: Optional[np.ndarray]) -> np.ndarray:
        angles = _encoding_angles(
            input_vec, prev_zexp, self.n_qubits, self.use_feedback
        )
        results = self._qnode(angles)
        return np.array(results, dtype=np.float32)

    def run_sequence(self, X_seq: np.ndarray,
                     warmup: int = None,
                     verbose: bool = False) -> np.ndarray:
        """
        Drive reservoir with X_seq of shape (T, n_features).
        Returns (T - warmup, 2*n_qubits).
        """
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane required.")

        warmup = QRC_WARMUP if warmup is None else warmup
        T = len(X_seq)
        if T <= warmup:
            raise ValueError(f"Sequence length {T} ≤ warmup {warmup}")

        states = []
        prev_zexp = None

        for t in range(T):
            state = self._step(X_seq[t], prev_zexp)
            if self.use_feedback:
                prev_zexp = state[:self.n_qubits]
            if t >= warmup:
                states.append(state)
            if verbose and t % 100 == 0:
                logger.debug(f"  Reservoir step {t}/{T}")

        result = np.array(states, dtype=np.float32)
        logger.debug(f"run_sequence: {X_seq.shape} → {result.shape}")
        return result

    def get_config(self) -> dict:
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
