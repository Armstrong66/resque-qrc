"""
reservoir/aquila_backend.py — QuEra Aquila (analog Rydberg) execution path.

Aquila is an ANALOG neutral-atom device: it has no gate set, so it cannot
run this project's RY/IsingZZ/RX circuit via a device swap the way the IBM
path does (see reservoir/quantum_reservoir.py::_get_ibm_device). This module
is a genuinely separate execution path that re-expresses the same
transverse-field Ising Hamiltonian (H = -J ΣZᵢZⱼ - h ΣXᵢ) as a real physical
Rydberg-atom program, using QuEra's own `bloqade-analog` SDK.

THE PHYSICAL MAPPING, PLAINLY STATED — read before trusting numbers from
this path:

  - Atom positions are fixed once per reservoir (chosen from J via the
    Rydberg blockade interaction V = C6/r^6), NOT re-set per timestep — this
    matches how J is already a construction-time constant for IsingQRC too.
  - "J" is realized via atom SPACING (interaction is always physically
    present, not a value you turn on/off), clamped to Aquila's minimum
    spacing / lattice-area limits when the ideal spacing doesn't fit — see
    _compute_geometry(). The REALIZED J can differ from the requested J if
    clamped; this is logged, not silently absorbed.
  - "h" (transverse field) is realized via the GLOBAL Rabi drive amplitude —
    real hardware only supports one shared, time-dependent Rabi
    waveform for the whole atom register, not independent per-atom control.
  - Per-timestep classical input (our "encoding angles") is realized via
    Aquila's LOCAL DETUNING addressing — the ONLY per-atom-addressable
    channel the hardware exposes. Critically, local detuning and its
    per-atom scale coefficients are constrained to be NON-NEGATIVE
    (capabilities: site_coefficient in [0, 1], local detuning in
    [0, 125] rad/us) — there is no direct way to encode a signed value.
    Encoding angles in [-π, π] are affine-remapped to coefficients in
    [0, 1] via (angle/π + 1) / 2.
  - Measurement is ALWAYS in the Rydberg-occupation (Z-like) basis — there
    is no native ⟨X⟩ measurement. This module gets an X-basis-like reading
    by appending a short resonant π/2 Rabi pulse immediately before
    measurement (a standard basis-rotation technique), and submits TWO
    programs per timestep (one for ⟨Z⟩, one for the rotated ⟨X⟩-like
    reading) — double the shots/cost of the gate-based path for the same
    feature_dim, which is the honest cost of getting both feature halves
    from a device that only measures one basis natively.
  - "all_to_all" topology is NOT literally realizable — 1/r^6 interactions
    fall off too fast for any real 2D placement to give uniform coupling
    beyond nearest neighbors. Atoms are placed on a compact ring so ADJACENT
    atoms meet the target spacing; more distant pairs interact more weakly.
    This is a documented approximation, not a claim of true all-to-all
    coupling — treat "all_to_all" results from this backend with that in
    mind. "chain" topology has no such caveat (a literal 1D chain).

VALIDATED, NOT LIVE-HARDWARE-VERIFIED: every code path in this module was
built against and exercised through bloqade-analog's own free local
emulator AND AWS Braket's stricter local AHS emulator (`AQUILA_SUBMIT_TARGET
= "local_emulator"` / `"braket_local_emulator"`, both free, no credentials
needed) — this confirms the programs are physically well-formed (correct
units, non-discontinuous waveforms, within all published hardware limits)
and produce bounded, non-degenerate, encoding-sensitive output. It has NOT
been run against real Aquila hardware. The physical calibration constants
(config.AQUILA_J_SCALE, AQUILA_H_SCALE, pulse durations) are a documented
design choice picked to land in the "blockade-dominant" regime (V >> Ω) that
makes the physics approximate an Ising-like model — not something verified
against a real device. Run a tiny job (n_steps=5-10) against
AQUILA_SUBMIT_TARGET="aquila" first and sanity-check the returned features
before trusting a larger run. See docs/PROJECT_CRITIQUE.md for the full
reasoning behind every choice above.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (AQUILA_J_SCALE, AQUILA_H_SCALE, AQUILA_ENCODE_US, AQUILA_DT_US,
                    AQUILA_ROTATION_US, AQUILA_MIN_SPACING_UM, AQUILA_MAX_WIDTH_UM,
                    AQUILA_MAX_HEIGHT_UM, AQUILA_SUBMIT_TARGET)
from utils import get_logger

logger = get_logger(__name__)

try:
    import bloqade.analog as bloqade
    BLOQADE_AVAILABLE = True
except ImportError:
    BLOQADE_AVAILABLE = False

MAX_SHOTS = 1000   # Aquila hardware limit (bloqade.get_capabilities().capabilities.task)


def _compute_geometry(n_qubits: int, J: float, topology: str) -> tuple[list, float]:
    """
    Choose atom positions (um) realizing interaction strength J via Rydberg
    blockade, clamped to Aquila's minimum spacing / lattice-area limits.
    Returns (positions, J_realized) — J_realized may differ from J if the
    ideal spacing didn't fit and had to be clamped (logged when it does).

    Clamping is bidirectional and this matters in practice: the "ideal"
    spacing for the project's own default AQUILA_J_SCALE/J_SWEEP values is
    LARGER than the lattice area allows for the larger end of
    config.QUBIT_COUNTS (e.g. n=12+) — an earlier version of this function
    only clamped spacing UP to the minimum and then simply rejected anything
    that didn't fit, which incorrectly rejected combinations that a SMALLER
    (but still hardware-valid) spacing would have accommodated. Spacing is
    now clamped to fit within whichever of [MIN_SPACING, area-limited max]
    is tightest, and only raises when even AQUILA_MIN_SPACING_UM (the
    hardware's own absolute minimum, not a value we chose) doesn't fit.
    """
    ideal_spacing = (bloqade.RB_C6 / max(J * AQUILA_J_SCALE, 1e-9)) ** (1 / 6)

    if topology == "chain":
        gaps = max(n_qubits - 1, 1)
        max_dim = max(AQUILA_MAX_WIDTH_UM, AQUILA_MAX_HEIGHT_UM)
        max_spacing_for_fit = max_dim / gaps
        if AQUILA_MIN_SPACING_UM > max_spacing_for_fit:
            max_n = int(max_dim / AQUILA_MIN_SPACING_UM) + 1
            raise ValueError(
                f"n_qubits={n_qubits} does not fit on Aquila's lattice even at "
                f"minimum atom spacing ({AQUILA_MIN_SPACING_UM} um): a chain "
                f"needs {(gaps * AQUILA_MIN_SPACING_UM):.1f} um at minimum "
                f"spacing, hardware allows at most {max_dim} um (~{max_n} "
                f"atoms in a straight chain). Reduce n_qubits for the Aquila "
                f"backend."
            )
        spacing = min(max(ideal_spacing, AQUILA_MIN_SPACING_UM), max_spacing_for_fit)
        length = gaps * spacing
        if length <= AQUILA_MAX_WIDTH_UM:
            positions = [(i * spacing, 0.0) for i in range(n_qubits)]
        else:
            positions = [(0.0, i * spacing) for i in range(n_qubits)]
    elif topology == "all_to_all":
        logger.warning(
            "Aquila 'all_to_all' is an approximation: atoms are placed on a "
            "compact ring so ADJACENT atoms meet the target spacing, but true "
            "uniform all-to-all coupling is not physically realizable with "
            "1/r^6 interactions — see reservoir/aquila_backend.py module "
            "docstring."
        )
        max_dim = min(AQUILA_MAX_WIDTH_UM, AQUILA_MAX_HEIGHT_UM)
        ring_factor = 2 * np.sin(np.pi / n_qubits)
        max_spacing_for_fit = max_dim * ring_factor / 2   # diameter = 2r <= max_dim
        if AQUILA_MIN_SPACING_UM > max_spacing_for_fit:
            raise ValueError(
                f"n_qubits={n_qubits} does not fit on Aquila's lattice as a "
                f"ring even at minimum atom spacing ({AQUILA_MIN_SPACING_UM} "
                f"um). Reduce n_qubits or use topology='chain' for the "
                f"Aquila backend."
            )
        spacing = min(max(ideal_spacing, AQUILA_MIN_SPACING_UM), max_spacing_for_fit)
        r = spacing / ring_factor
        positions = [
            (r * np.cos(2 * np.pi * i / n_qubits), r * np.sin(2 * np.pi * i / n_qubits))
            for i in range(n_qubits)
        ]
    else:
        raise ValueError(f"Unknown topology for Aquila: {topology!r}")

    J_phys_realized = bloqade.RB_C6 / spacing**6
    J_realized = J_phys_realized / AQUILA_J_SCALE
    if abs(J_realized - J) > 0.05 * max(abs(J), 1e-6):
        logger.warning(
            f"Aquila geometry: requested J={J} but hardware spacing/area "
            f"limits only allow J~{J_realized:.4f} (spacing clamped to "
            f"{spacing:.2f} um, min allowed {AQUILA_MIN_SPACING_UM} um). "
            f"Using the realized value for this reservoir instance."
        )
    return positions, J_realized


class AquilaBackend:
    """
    Drives the transverse-field Ising QRC as a real analog Rydberg program.
    Mirrors IsingQRC's per-step contract (step(input_vec, prev_zexp) ->
    (2*n_qubits,) feature vector) so IsingQRC can delegate to it transparently
    — see IsingQRC._step. Geometry and Hamiltonian parameters are fixed once
    at construction, matching IsingQRC's own "build once, reuse every step"
    design (see the module-level warning in quantum_reservoir.py about the
    standalone prototype that rebuilt its device per-step).
    """

    def __init__(self, n_qubits: int, J: float, h: float, topology: str,
                 trotter_steps: int, use_data_reuploading: bool,
                 use_feedback: bool, shots: Optional[int] = None,
                 submit_target: Optional[str] = None):
        if not BLOQADE_AVAILABLE:
            raise ImportError(
                "QRC_BACKEND='aquila' requires bloqade-analog: "
                "pip install bloqade-analog"
            )
        self.n_qubits = n_qubits
        self.trotter_steps = trotter_steps
        self.use_data_reuploading = use_data_reuploading
        self.use_feedback = use_feedback
        # Resolution order matches resolve_hardware_backend()'s pattern:
        # explicit arg > $AQUILA_SUBMIT_TARGET env var > config default. The
        # env var override exists so a real-hardware run never depends on
        # having edited (and not forgotten to revert) config.py.
        self.submit_target = (submit_target or os.environ.get("AQUILA_SUBMIT_TARGET")
                              or AQUILA_SUBMIT_TARGET)
        if self.submit_target not in ("local_emulator", "braket_local_emulator", "aquila"):
            raise ValueError(f"Unknown AQUILA_SUBMIT_TARGET={self.submit_target!r}")

        requested_shots = shots if (shots is not None and shots > 0) else 200
        if requested_shots > MAX_SHOTS:
            logger.warning(f"Aquila shots={requested_shots} exceeds hardware max "
                           f"{MAX_SHOTS}; clamping.")
        self.shots = min(requested_shots, MAX_SHOTS)

        positions, J_realized = _compute_geometry(n_qubits, J, topology)
        self.h_phys = h * AQUILA_H_SCALE
        self.J = J_realized
        self._geometry = bloqade.start.add_position(positions)

        logger.info(
            f"AquilaBackend | qubits={n_qubits} J={J}->realized={J_realized:.4f} "
            f"h={h}->Omega={self.h_phys:.2f}rad/us topology={topology} "
            f"shots={self.shots} target={self.submit_target} "
            f"encoding={'data-reuploading' if use_data_reuploading else 'standard'}"
        )
        if self.submit_target == "aquila":
            logger.warning("AquilaBackend target='aquila' — every step() call "
                           "SUBMITS REAL CIRCUITS to real hardware and consumes "
                           "real QPU time/credits.")

    def _build_program(self, coeffs: np.ndarray, add_x_rotation: bool):
        n = self.n_qubits
        det_durations, det_values = [], [0.0]
        amp_durations, amp_values = [], [0.0]

        def add_encode_segment():
            det_durations.extend([AQUILA_ENCODE_US / 2, AQUILA_ENCODE_US / 2])
            det_values.extend([50.0, 0.0])
            amp_durations.append(AQUILA_ENCODE_US)
            amp_values.append(0.0)

        def add_evolve_segment():
            det_durations.append(AQUILA_DT_US)
            det_values.append(0.0)
            amp_durations.append(AQUILA_DT_US)
            amp_values.append(self.h_phys)

        if self.use_data_reuploading:
            for _ in range(self.trotter_steps):
                add_encode_segment()
                add_evolve_segment()
        else:
            add_encode_segment()
            for _ in range(self.trotter_steps):
                add_evolve_segment()

        if add_x_rotation:
            omega_rot = (np.pi / 2) / AQUILA_ROTATION_US
            det_durations.append(AQUILA_ROTATION_US)
            det_values.append(0.0)
            amp_durations.append(AQUILA_ROTATION_US)
            amp_values.append(omega_rot)

        prog = (
            self._geometry
            .rydberg.detuning.location(list(range(n)), scales=list(map(float, coeffs)))
            .piecewise_linear(det_durations, det_values)
            .rydberg.rabi.amplitude.uniform
            .piecewise_linear(amp_durations, amp_values)
        )
        return prog

    def _run_program(self, coeffs: np.ndarray, add_x_rotation: bool) -> np.ndarray:
        prog = self._build_program(coeffs, add_x_rotation)
        if self.submit_target == "local_emulator":
            job = prog.bloqade.python().run(self.shots)
        elif self.submit_target == "braket_local_emulator":
            job = prog.braket.local_emulator().run(self.shots)
        else:  # "aquila"
            job = prog.braket.aquila().run(self.shots)
        densities = job.report().rydberg_densities()
        return densities.iloc[0].to_numpy(dtype=np.float64)

    def step(self, input_vec: np.ndarray, prev_zexp: Optional[np.ndarray]) -> np.ndarray:
        """Matches IsingQRC._step's contract: returns a (2*n_qubits,) feature vector."""
        from reservoir.quantum_reservoir import _encoding_angles
        angles = _encoding_angles(input_vec, prev_zexp, self.n_qubits, self.use_feedback)
        coeffs = np.clip((angles / np.pi + 1.0) / 2.0, 0.0, 1.0)

        z_density = self._run_program(coeffs, add_x_rotation=False)
        x_density = self._run_program(coeffs, add_x_rotation=True)

        # |g> (not Rydberg) -> +1, |r> (Rydberg) -> -1 convention.
        z_exp = 1.0 - 2.0 * z_density
        x_exp = 1.0 - 2.0 * x_density
        return np.concatenate([z_exp, x_exp]).astype(np.float32)
