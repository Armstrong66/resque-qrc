"""
reservoir/quantum_reservoir.py — Transverse-Field Ising QRC via PennyLane.

Design decisions documented in config.py:
  H = -J ΣZᵢZⱼ - h ΣXᵢ
  Outputs 2*n_qubits expectation values per step [⟨σᶻ⟩, ⟨σˣ⟩].
  Feedback: blend prior ⟨Z⟩ with PCA input when dimensions match (Zhu et al. 2025).

Encoding, switchable via config.USE_DATA_REUPLOADING / IsingQRC(use_data_reuploading=):
  Standard (False):     encode ONCE, then [ZZ+X] × trotter_steps.
  Data reuploading (True): [encode → ZZ+X] × trotter_steps (Pérez-Salinas et al. 2020).
    Re-injecting the input before every Trotter step increases effective
    expressivity without adding qubits — the primary lever for closing the
    QRC-vs-ESN gap. Ported and hardened from the standalone prototype
    (quantum_reservoir_with_data_reuploading_but_with_old_error.py, since
    removed): that script rebuilt the PennyLane device AND recompiled the
    qnode on every single timestep (catastrophic slowdown — device/qnode
    construction is O(circuit size), not O(1), and doing it per-step turns an
    O(T) reservoir drive into something far worse), and it fed the feedback
    term through `np.resize` on the concatenated [input, prev_Z] vector,
    silently wrapping values cyclically instead of aligning them — the same
    "no cyclic resize" bug already fixed once in this file (see
    _encoding_angles below). Neither issue is reproduced here: the device and
    qnode are built ONCE in __init__ and reused for every step, and feedback
    uses the existing truncate/average path.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (J_DEFAULT, H_DEFAULT, QUBIT_PRIMARY, TOPOLOGY_PRIMARY,
                    TROTTER_STEPS, EVOLUTION_TIME, USE_FEEDBACK, QRC_WARMUP,
                    USE_DATA_REUPLOADING, RANDOM_SEED)
from utils import get_logger

logger = get_logger(__name__)

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not installed. Install: pip install pennylane")

# Real-hardware target, independent of `platform` (which only ever chooses
# between two SIMULATORS — lightning.qubit vs default.qubit). Resolution
# order: explicit constructor arg > $QRC_BACKEND env var > "simulation".
#   "simulation" (default) — unchanged behaviour, see _get_device below.
#   "ibm"                  — real IBM Eagle/Heron via pennylane-qiskit +
#                             Qiskit Runtime. See _get_ibm_device. A genuine
#                             device swap — the gate circuit below runs
#                             unchanged.
#   "aquila"                — real QuEra Aquila via bloqade-analog. Aquila is
#                             analog (no gate set), so this is NOT a device
#                             swap — IsingQRC delegates entirely to
#                             reservoir/aquila_backend.py::AquilaBackend for
#                             this backend, bypassing PennyLane/_get_device
#                             below. See that module's docstring for the full
#                             physical mapping and its caveats.
VALID_HARDWARE_BACKENDS = ("simulation", "ibm", "aquila")


def resolve_hardware_backend(explicit: Optional[str] = None) -> str:
    backend = (explicit or os.environ.get("QRC_BACKEND") or "simulation").lower()
    if backend not in VALID_HARDWARE_BACKENDS:
        raise ValueError(f"Unknown hardware_backend={backend!r}; "
                         f"expected one of {VALID_HARDWARE_BACKENDS}")
    return backend


def _get_ibm_device(n_qubits: int, shots: Optional[int]):
    """
    Real IBM Eagle/Heron via the pennylane-qiskit plugin + Qiskit Runtime.
    This is a genuine drop-in device swap: the existing RY/IsingZZ/RX qnode
    body is gate-based and runs UNCHANGED against this device — no circuit
    rewrite needed, which is exactly the "clean hardware integration point"
    described in docs/AGENTIC_DESIGN_GUIDE.md.

    IMPORTANT — disclosed, not hidden: this path is implemented against the
    documented pennylane-qiskit / qiskit-ibm-runtime interface but has NOT
    been executed against live IBM hardware in this environment (no IBM
    Quantum account/token available here). Confirm end-to-end against a real
    backend (or at minimum `ibmq_qasm_simulator`) before trusting it for a
    submission-critical run.

    Requires: pip install qiskit-ibm-runtime pennylane-qiskit, and an IBM
    Quantum account token saved once via
    `QiskitRuntimeService.save_account(channel="ibm_quantum", token=...)`
    — see https://docs.quantum.ibm.com/guides/setup-channel
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as e:
        raise ImportError(
            "QRC_BACKEND='ibm' requires qiskit-ibm-runtime and pennylane-qiskit: "
            "pip install qiskit-ibm-runtime pennylane-qiskit. Also requires an "
            "IBM Quantum account token saved via QiskitRuntimeService.save_account(...) "
            "— see https://docs.quantum.ibm.com/guides/setup-channel"
        ) from e

    service = QiskitRuntimeService()
    ibm_backend_name = os.environ.get("QRC_IBM_BACKEND")
    if ibm_backend_name:
        ibm_backend = service.backend(ibm_backend_name)
    else:
        ibm_backend = service.least_busy(operational=True, min_num_qubits=n_qubits)
    logger.warning(f"QRC_BACKEND=ibm -> submitting to REAL HARDWARE "
                   f"{ibm_backend.name} ({ibm_backend.num_qubits} qubits). "
                   f"This consumes real QPU time/credits.")

    # Real hardware has no "exact simulation" mode — always finite shots.
    shots_arg = int(shots) if (shots is not None and shots > 0) else 1024
    return qml.device("qiskit.remote", wires=n_qubits, backend=ibm_backend, shots=shots_arg)


def _get_device(n_qubits: int, noise_rate: float, shots: Optional[int],
                platform: str = "local", hardware_backend: Optional[str] = None):
    """
    Device factory for PennyLane-backed backends only ("simulation" / "ibm")
    — "aquila" never reaches this function; IsingQRC branches to
    AquilaBackend before building a qnode at all (Aquila has no gate-based
    device to construct here). `hardware_backend` falls back to
    $QRC_BACKEND, then "simulation"; `platform` separately only ever chooses
    between simulators (lightning.qubit vs default.qubit) and is ignored
    once "ibm" is selected.

    For the "simulation" backend, `shots` is deliberately NOT passed to
    qml.device(...) here — PennyLane >=0.4x deprecates device-level shots in
    favour of the qml.set_shots QNode transform (applied once in
    IsingQRC._build_qnode, alongside the device built here — noise and
    shot-budget stay independently combinable either way).
    """
    if not PENNYLANE_AVAILABLE:
        raise ImportError("PennyLane required. pip install pennylane")

    backend = resolve_hardware_backend(hardware_backend)
    if backend == "ibm":
        return _get_ibm_device(n_qubits, shots)
    if backend == "aquila":
        raise RuntimeError(
            "_get_device() should never be called for hardware_backend='aquila' "
            "— IsingQRC.__init__ must branch to AquilaBackend before this point. "
            "This is a bug in the caller, not a missing feature."
        )

    # backend == "simulation" — original behaviour, minus device-level shots.
    if noise_rate > 0.0:
        return qml.device("default.mixed", wires=n_qubits)
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
    NaN-guarded: a stray NaN (e.g. a degenerate PCA component) would
    otherwise propagate into every downstream expectation value for the
    remainder of the sequence via the feedback term.
    """
    x = np.asarray(input_vec, dtype=np.float64).ravel()
    if np.any(np.isnan(x)):
        x = np.where(np.isnan(x), 0.0, x)
    if len(x) < n_qubits:
        x = np.resize(x, n_qubits)
    elif len(x) > n_qubits:
        x = x[:n_qubits]

    if prev_zexp is not None and use_feedback:
        z = np.asarray(prev_zexp, dtype=np.float64).ravel()[:n_qubits]
        if np.any(np.isnan(z)):
            z = np.where(np.isnan(z), 0.0, z)
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
                 use_data_reuploading: bool = None,
                 platform:      str   = "local",
                 hardware_backend: Optional[str] = None):
        """
        hardware_backend: "simulation" (default) | "ibm" | "aquila". Falls
        back to the $QRC_BACKEND env var, then "simulation", if not given —
        this is the hook described in docs/AGENTIC_DESIGN_GUIDE.md §4.2.
        "ibm" is a genuine PennyLane device swap (see _get_ibm_device).
        "aquila" is NOT a device swap — Aquila has no gate set, so this
        delegates entirely to reservoir/aquila_backend.py::AquilaBackend,
        which re-expresses J/h as a real analog Rydberg program. Read that
        module's docstring before trusting numbers from this backend.
        Independent of `platform`, which only ever chooses between
        simulators.
        """

        self.platform       = platform
        self.hardware_backend = resolve_hardware_backend(hardware_backend)
        self.n_qubits       = n_qubits       or QUBIT_PRIMARY
        self.J              = J if J is not None else J_DEFAULT
        self.h              = h if h is not None else H_DEFAULT
        self.topology       = topology       or TOPOLOGY_PRIMARY
        self.trotter_steps  = trotter_steps  or TROTTER_STEPS
        self.evolution_time = evolution_time or EVOLUTION_TIME
        self.noise_rate     = noise_rate
        self.shots          = shots
        self.use_feedback   = use_feedback if use_feedback is not None else USE_FEEDBACK
        self.use_data_reuploading = (use_data_reuploading if use_data_reuploading is not None
                                     else USE_DATA_REUPLOADING)
        self.feature_dim    = 2 * self.n_qubits
        self._pairs         = self._coupling_pairs(self.topology, self.n_qubits)
        self._qnode         = None
        self._aquila         = None
        if self.hardware_backend == "aquila":
            self._aquila = self._build_aquila_backend()   # built ONCE, reused every step
        else:
            self._qnode = self._build_qnode()              # built ONCE, reused every step

        logger.info(
            f"IsingQRC | qubits={self.n_qubits} J={self.J} h={self.h} "
            f"topology={self.topology} noise={self.noise_rate} "
            f"shots={self.shots if self.shots else 'exact'} "
            f"feedback={self.use_feedback} "
            f"encoding={'data-reuploading' if self.use_data_reuploading else 'standard'} "
            f"hardware_backend={self.hardware_backend}"
        )

    def _build_aquila_backend(self):
        from reservoir.aquila_backend import AquilaBackend
        return AquilaBackend(
            n_qubits=self.n_qubits, J=self.J, h=self.h, topology=self.topology,
            trotter_steps=self.trotter_steps,
            use_data_reuploading=self.use_data_reuploading,
            use_feedback=self.use_feedback, shots=self.shots,
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
        use_reupload = self.use_data_reuploading
        dev = _get_device(n, noise_rate, self.shots, self.platform, self.hardware_backend)

        def _encode(angles):
            for i in range(n):
                qml.RY(float(angles[i]), wires=i)

        def _evolve(dt):
            for (i, j) in pairs:
                qml.IsingZZ(-2 * J * dt, wires=[i, j])
            for i in range(n):
                qml.RX(-2 * h_field * dt, wires=i)

        @qml.qnode(dev, diff_method=None)
        def circuit(angles):
            dt = evolution_time / trotter_steps

            if use_reupload:
                # Re-encode before EVERY Trotter step (Pérez-Salinas et al. 2020)
                for _ in range(trotter_steps):
                    _encode(angles)
                    _evolve(dt)
            else:
                # Standard: single injection, then free evolution
                _encode(angles)
                for _ in range(trotter_steps):
                    _evolve(dt)

            if noise_rate > 0.0:
                for i in range(n):
                    qml.DepolarizingChannel(noise_rate, wires=i)

            return (
                [qml.expval(qml.PauliZ(i)) for i in range(n)] +
                [qml.expval(qml.PauliX(i)) for i in range(n)]
            )

        # Finite shots applied via the QNode transform, not device-level —
        # only meaningful for the simulation backend; "ibm"/"aquila" set
        # their own shot count at device construction (see _get_ibm_device).
        if self.hardware_backend == "simulation" and self.shots is not None and self.shots > 0:
            circuit = qml.set_shots(circuit, shots=int(self.shots))

        return circuit

    def _step(self, input_vec: np.ndarray,
              prev_zexp: Optional[np.ndarray]) -> np.ndarray:
        if self.hardware_backend == "aquila":
            # AquilaBackend computes its own encoding angles internally (it
            # needs them to derive non-negative local-detuning coefficients,
            # not gate rotation angles) — see AquilaBackend.step().
            state = self._aquila.step(input_vec, prev_zexp)
        else:
            angles = _encoding_angles(
                input_vec, prev_zexp, self.n_qubits, self.use_feedback
            )
            results = self._qnode(angles)
            state = np.array(results, dtype=np.float32)
        if np.any(np.isnan(state)):
            logger.warning("NaN in reservoir output state — substituting zeros. "
                           "Check input data / Hamiltonian parameters.")
            state = np.zeros(self.feature_dim, dtype=np.float32)
        return state

    def run_sequence(self, X_seq: np.ndarray,
                     warmup: int = None,
                     verbose: bool = False) -> np.ndarray:
        """
        Drive reservoir with X_seq of shape (T, n_features).
        Returns (T - warmup, 2*n_qubits).
        """
        if self.hardware_backend != "aquila" and not PENNYLANE_AVAILABLE:
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
            "hardware_backend":     self.hardware_backend,
        }


def run_qubit_sweep(X_seq: np.ndarray,
                    qubit_counts: list = None,
                    J: float = None,
                    h: float = None,
                    use_data_reuploading: bool = None,
                    **kwargs) -> dict[int, np.ndarray]:
    from config import QUBIT_COUNTS
    qubit_counts = qubit_counts or QUBIT_COUNTS
    J = J or J_DEFAULT
    h = h or H_DEFAULT

    results = {}
    for n in qubit_counts:
        logger.info(f"Running qubit sweep: n={n}")
        qrc = IsingQRC(n_qubits=n, J=J, h=h,
                       use_data_reuploading=use_data_reuploading, **kwargs)
        results[n] = qrc.run_sequence(X_seq)
    return results


# ── Encoding ablation: standard vs data reuploading ────────────────────────────

def encoding_ablation(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       n_qubits: int = 9, J: float = None, h: float = None,
                       max_steps: Optional[int] = None,
                       out_dir: Path = None) -> dict:
    """
    Compare standard vs data-reuploading encoding on the same reservoir config.
    Primary Phase 3 diagnostic experiment. Returns {label: val_rmse}.

    max_steps truncates run_sequence for a fast comparison (None = full data).
    Note: run_sequence has no max_steps parameter — the cap is applied by
    slicing the input sequence before driving the reservoir, so both encodings
    see identical data.
    """
    import json
    from readout.ridge_readout import _ridge_solve, _rmse_per_target

    J = J if J is not None else J_DEFAULT
    h = h if h is not None else H_DEFAULT
    results = {}

    X_tr = X_train if max_steps is None else X_train[:max_steps + 20]
    X_vl = X_val if max_steps is None else X_val[:max_steps + 20]

    for reupload in [False, True]:
        label = "data_reuploading" if reupload else "standard"
        logger.info(f"Encoding ablation: {label}")
        try:
            qrc = IsingQRC(n_qubits=n_qubits, J=J, h=h, use_data_reuploading=reupload)
            H_tr = qrc.run_sequence(X_tr, warmup=20)
            H_vl = qrc.run_sequence(X_vl, warmup=20)

            n_tr = min(len(H_tr), len(y_train) - 20)
            n_vl = min(len(H_vl), len(y_val) - 20)
            W    = _ridge_solve(H_tr[:n_tr], y_train[20:20 + n_tr], lambda_=1e-3)
            pred = H_vl[:n_vl] @ W
            rmse = float(_rmse_per_target(y_val[20:20 + n_vl], pred).mean())
            results[label] = rmse
            logger.info(f"  {label}: val_rmse={rmse:.4f}")
        except Exception as e:
            logger.error(f"  {label} failed: {e}")
            results[label] = float("inf")

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
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "encoding_ablation.json", "w") as f:
            json.dump(results, f, indent=2)

    return results
