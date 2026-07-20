"""
scripts/hardware_validation.py — Small, deliberately-subsampled real-hardware
validation run for the FINAL selected QRC configuration.

Per docs/PROJECT_CRITIQUE.md §3.2: running the whole sweep/training pipeline
on real hardware is not feasible — hamiltonian_sweep alone is |J_SWEEP| x
|H_SWEEP| configs, each driving hundreds of timesteps, and every timestep is
one circuit execution. That is thousands of individually-queued hardware jobs
if pointed at real hardware as-is.

This script instead takes the config ALREADY selected by the simulator
sweeps and ALREADY fitted by main.py (read directly from
outputs/results/h{N}/{mode}_config.json and {mode}_readout.pkl — the same
files main.py itself writes, so there is exactly one source of truth) and
drives the reservoir over a small, fixed subsample of the test set on the
backend named by --backend / $QRC_BACKEND. Default backend is "simulation",
so running this script with no arguments is always safe and requires no
hardware access — it becomes a real hardware submission only when you
explicitly ask for one.

Usage:
    # Safe, simulator-only sanity check of the harness itself:
    python scripts/hardware_validation.py --horizon 6 --n_steps 50

    # QuEra Aquila (PRIMARY hardware backend — requires bloqade-analog):
    python scripts/hardware_validation.py --horizon 6 --n_steps 10 --backend aquila
    # Aquila is analog (no gate set) — see reservoir/aquila_backend.py for
    # the full physical mapping and its caveats before trusting numbers from
    # this path. config.AQUILA_SUBMIT_TARGET controls whether "aquila" here
    # actually reaches real hardware ("aquila") or one of the two FREE local
    # emulators ("local_emulator" default / "braket_local_emulator") — check
    # it's set to "aquila" before running this if you intend to spend real
    # qBraid/QuEra credits. Start with --n_steps 5-10, not 50.

    # IBM Eagle/Heron (FALLBACK hardware backend — requires
    # qiskit-ibm-runtime, pennylane-qiskit, and a saved IBM Quantum account
    # token):
    python scripts/hardware_validation.py --horizon 6 --n_steps 30 --backend ibm

Output: outputs/results/hardware_validation.json — records both the
simulator result AND the requested hardware backend's result (when not
"simulation") on the IDENTICAL subsampled input, so the delta between them
is a direct, honest sim-vs-hardware comparison rather than two different
runs on different data.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS, TARGETS, HORIZONS, QRC_WARMUP, HARDWARE_VALIDATION_STEPS
from utils import get_logger

logger = get_logger("hardware_validation")


def _require(path: Path, hint: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — {hint}")


def run_hardware_validation(horizon: int = None, n_steps: int = None,
                             backend: str = None,
                             mode: str = "warm_start_qrc") -> dict:
    from preprocessing.pipeline import WeatherPreprocessor
    from preprocessing.projection import PCAProjector
    from reservoir.quantum_reservoir import IsingQRC, resolve_hardware_backend
    from readout.ridge_readout import ReadoutResult
    from evaluation.metrics import rmse_per_target, valid_prediction_time

    horizon = horizon or HORIZONS[0]
    n_steps = n_steps or HARDWARE_VALIDATION_STEPS
    backend = resolve_hardware_backend(backend)

    out_h = RESULTS / f"h{horizon}"
    cfg_path = out_h / f"{mode}_config.json"
    _require(cfg_path, "run the full pipeline first (python main.py) so a "
                       "final config has been selected and written.")
    cfg = json.load(open(cfg_path))
    n_qubits = cfg["n_qubits"]

    readout_path = out_h / f"{mode}_readout.pkl"
    _require(readout_path, "run the full pipeline first (python main.py) so "
                           "a fitted readout exists.")
    readout = ReadoutResult.load(readout_path)

    if "use_data_reuploading" not in cfg:
        logger.warning(f"{cfg_path.name} predates the use_data_reuploading field "
                       f"— it was written before that flag was wired up. Re-run "
                       f"the full pipeline (python main.py) to regenerate it with "
                       f"a known, recorded encoding before trusting this "
                       f"validation run.")
    use_reupload = cfg.get("use_data_reuploading", False)

    ds = WeatherPreprocessor.load(horizon)
    pca_path = RESULTS / f"shared_pca_h{horizon}.pkl"
    X_test = PCAProjector.load(pca_path).transform(ds.X_test) if pca_path.exists() else ds.X_test

    w = min(QRC_WARMUP, max(0, n_steps // 5))
    n_total = n_steps + w
    if n_total > len(X_test):
        raise ValueError(f"Test set has only {len(X_test)} rows; need "
                         f"{n_total} (n_steps={n_steps} + warmup={w}). "
                         f"Reduce --n_steps.")
    X_sub = X_test[:n_total]
    y_sub = ds.y_test[w:n_total]

    logger.warning(
        f"Hardware validation: backend={backend} n_steps={n_steps} "
        f"horizon={horizon}h n_qubits={n_qubits} mode={mode}."
        + (" This SUBMITS REAL CIRCUITS to real hardware — consumes QPU "
           "time/credits." if backend != "simulation" else " Simulator "
           "only — safe, no hardware access needed.")
    )

    def _run(be: str) -> dict:
        qrc = IsingQRC(
            n_qubits=n_qubits, J=cfg["J"], h=cfg["h"], noise_rate=cfg["noise_rate"],
            topology=cfg["topology"], trotter_steps=cfg["trotter_steps"],
            evolution_time=cfg["evolution_time"], use_feedback=cfg["use_feedback"],
            use_data_reuploading=use_reupload, shots=cfg["shots"],
            hardware_backend=be,
        )
        t0 = time.time()
        H = qrc.run_sequence(X_sub, warmup=w)
        n = min(len(H), len(y_sub))
        H_a, y_a = H[:n], y_sub[:n]
        if H_a.shape[1] != readout.W.shape[0]:
            raise ValueError(
                f"Reservoir feature dim {H_a.shape[1]} != readout weight dim "
                f"{readout.W.shape[0]} — {cfg_path.name} and {readout_path.name} "
                f"are out of sync. Re-run the full pipeline (python main.py) "
                f"so both are regenerated together."
            )
        pred = H_a @ readout.W
        rmse = rmse_per_target(y_a, pred)
        vpt = valid_prediction_time(y_a, pred)
        wall = time.time() - t0
        logger.info(f"  [{be}] n={n} rmse_mean={rmse.mean():.4f} "
                    f"vpt={vpt:.3f} wall_clock={wall:.1f}s")
        return {
            "backend": be, "n_steps": n,
            "rmse_mean": float(rmse.mean()),
            "rmse_per_target": {t: float(v) for t, v in zip(TARGETS, rmse)},
            "vpt_lyapunov": float(vpt),
            "wall_clock_s": round(wall, 2),
        }

    result = {
        "task": "hardware_validation",
        "horizon_hours": horizon,
        "mode": mode,
        "config_source": str(cfg_path),
        "readout_source": str(readout_path),
        "n_qubits": n_qubits,
        "J": cfg["J"], "h": cfg["h"], "noise_rate": cfg["noise_rate"],
        "topology": cfg["topology"], "use_data_reuploading": use_reupload,
        "requested_backend": backend,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    result["simulation"] = _run("simulation")
    if backend != "simulation":
        result[backend] = _run(backend)
        result["sim_vs_hw_rmse_delta"] = (
            result[backend]["rmse_mean"] - result["simulation"]["rmse_mean"]
        )

    out_path = RESULTS / "hardware_validation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Hardware validation saved -> {out_path}")
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="Small, subsampled real-hardware QRC validation run "
                    "against the already-selected simulator config.")
    p.add_argument("--horizon", type=int, default=None,
                   help=f"Forecast horizon in hours (default: {HORIZONS[0]})")
    p.add_argument("--n_steps", type=int, default=None,
                   help=f"Number of post-warmup test steps to run "
                        f"(default: config.HARDWARE_VALIDATION_STEPS={HARDWARE_VALIDATION_STEPS})")
    p.add_argument("--backend", type=str, default=None,
                   help="simulation|ibm|aquila (default: $QRC_BACKEND env var, "
                        "else 'simulation')")
    p.add_argument("--mode", type=str, default="warm_start_qrc",
                   choices=["warm_start_qrc", "cold_start_qrc"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_hardware_validation(
        horizon=args.horizon, n_steps=args.n_steps,
        backend=args.backend, mode=args.mode,
    )
    print("AGENT_RESULT: " + json.dumps(result))
