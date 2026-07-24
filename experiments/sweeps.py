"""
experiments/sweeps.py — Automated sweeps for challenge requirements.

1. Memory–nonlinearity tradeoff sweep (J, h)         → select (J*, h*)
2. Noise rate sweep (depolarizing p)                  → optimal noise regime
3. Qubit scaling study (n = 5, 7, 9, 12, 16, 20)     → required by challenge
4. Shot budget ablation (N shots)                     → NISQ robustness

All sweeps save results to outputs/results/ as CSVs + JSON.
Logs all decisions for reproducibility and qBraid Skill compatibility.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (J_SWEEP, H_SWEEP, NOISE_RATES, QUBIT_COUNTS,
                    SHOT_COUNTS, TOPOLOGIES, RESULTS, RANDOM_SEED, QRC_WARMUP,
                    SWEEP_MAX_TRAIN_SAMPLES, NOISE_SWEEP_MAX_SAMPLES,
                    USE_DATA_REUPLOADING)
from utils import get_logger

logger = get_logger(__name__)
np.random.seed(RANDOM_SEED)


def _qrc_warmup(n_samples: int) -> int:
    return min(QRC_WARMUP, max(0, n_samples // 10))


def _subsample_sweep_data(X_train, y_train, X_val, y_val, max_n: int = None):
    """Use a fixed prefix of train/val for faster sweeps (temporal order preserved)."""
    if max_n is None:
        max_n = SWEEP_MAX_TRAIN_SAMPLES
    if max_n is None or len(X_train) <= max_n:
        return X_train, y_train, X_val, y_val
    logger.info(f"Sweep subsample: using first {max_n}/{len(X_train)} train windows")
    return X_train[:max_n], y_train[:max_n], X_val, y_val


def _quick_eval(qrc, X_train, y_train, X_val, y_val, lambda_=1e-3):
    """Run reservoir and evaluate with ridge regression. Returns val RMSE mean."""
    from readout.ridge_readout import _ridge_solve, _rmse_per_target

    try:
        w = _qrc_warmup(len(X_train))
        H_train = qrc.run_sequence(X_train, warmup=w)
        H_val   = qrc.run_sequence(X_val, warmup=w)

        y_tr = y_train[w:]
        y_vl = y_val[w:]
        min_len = min(len(H_train), len(y_tr))
        H_train, y_tr = H_train[:min_len], y_tr[:min_len]
        min_len = min(len(H_val), len(y_vl))
        H_val, y_vl = H_val[:min_len], y_vl[:min_len]

        W    = _ridge_solve(H_train, y_tr, lambda_)
        pred = H_val @ W
        return float(_rmse_per_target(y_vl, pred).mean())
    except Exception as e:
        logger.error(f"Quick eval failed: {e}")
        return float("inf")


# ── 1. Memory–nonlinearity tradeoff sweep ─────────────────────────────────────

def hamiltonian_sweep(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      n_qubits: int = 9,
                      topology: str = "chain",
                      use_data_reuploading: bool = None,
                      out_dir: Path = None) -> tuple[float, float, pd.DataFrame]:
    """
    Sweep J ∈ J_SWEEP × h ∈ H_SWEEP. Returns (J*, h*, results_df).
    Also saves a CSV heatmap for figures.

    use_data_reuploading: defaults to config.USE_DATA_REUPLOADING. J*/h* are
    encoding-dependent (reuploading changes the effective circuit), so the
    winning values and the encoding that produced them are both recorded in
    best_hamiltonian.json — re-running this sweep with reuploading enabled
    for the first time will overwrite that file with new J*/h*, as expected.
    """
    from reservoir.quantum_reservoir import IsingQRC
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    use_data_reuploading = (USE_DATA_REUPLOADING if use_data_reuploading is None
                            else use_data_reuploading)

    X_train, y_train, X_val, y_val = _subsample_sweep_data(X_train, y_train, X_val, y_val)

    logger.info(f"=== Hamiltonian sweep: {len(J_SWEEP)}x{len(H_SWEEP)} "
                f"= {len(J_SWEEP)*len(H_SWEEP)} configs, n_qubits={n_qubits}, "
                f"reuploading={use_data_reuploading} ===")

    rows = []
    best_rmse = float("inf")
    best_J, best_h = J_SWEEP[0], H_SWEEP[0]

    for J, h in product(J_SWEEP, H_SWEEP):
        qrc = IsingQRC(n_qubits=n_qubits, J=J, h=h, topology=topology, noise_rate=0.0,
                       use_data_reuploading=use_data_reuploading)
        rmse = _quick_eval(qrc, X_train, y_train, X_val, y_val)
        rows.append({"J": J, "h": h, "val_rmse": rmse})
        logger.debug(f"  J={J:.2f} h={h:.2f} → val_rmse={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_J, best_h = J, h

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hamiltonian_sweep.csv", index=False)

    logger.info(f"Hamiltonian sweep done. Best: J*={best_J} h*={best_h} "
                f"val_rmse={best_rmse:.4f}")
    with open(out_dir / "best_hamiltonian.json", "w") as f:
        json.dump({"J_star": best_J, "h_star": best_h,
                   "val_rmse": best_rmse, "n_qubits": n_qubits,
                   "use_data_reuploading": use_data_reuploading}, f, indent=2)
    return best_J, best_h, df


# ── 2. Noise rate sweep ────────────────────────────────────────────────────────

def noise_sweep(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                J: float, h: float,
                n_qubits: int = 9,
                use_data_reuploading: bool = None,
                out_dir: Path = None) -> tuple[float, pd.DataFrame]:
    """
    Sweep depolarizing noise rates. Returns (p*, results_df).
    Tests hypothesis: optimal p* > 0 (noise-assisted generalization).
    """
    from reservoir.quantum_reservoir import IsingQRC
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    use_data_reuploading = (USE_DATA_REUPLOADING if use_data_reuploading is None
                            else use_data_reuploading)

    # Use fewer samples for noise sweep because default.mixed (density matrix)
    # is O(4ⁿ) vs default.qubit's O(2ⁿ) — 50 steps is enough for p* selection
    X_train, y_train, X_val, y_val = _subsample_sweep_data(
        X_train, y_train, X_val, y_val, max_n=NOISE_SWEEP_MAX_SAMPLES
    )

    logger.info(f"=== Noise sweep: {len(NOISE_RATES)} rates, "
                f"J={J} h={h} n_qubits={n_qubits} reuploading={use_data_reuploading} ===")

    rows = []
    best_rmse = float("inf")
    best_p    = 0.0

    for p in NOISE_RATES:
        qrc  = IsingQRC(n_qubits=n_qubits, J=J, h=h, noise_rate=p,
                        use_data_reuploading=use_data_reuploading)
        rmse = _quick_eval(qrc, X_train, y_train, X_val, y_val)
        rows.append({"noise_rate": p, "val_rmse": rmse})
        logger.info(f"  p={p:.4f} → val_rmse={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_p    = p

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "noise_sweep.csv", index=False)

    if best_p > 0:
        logger.info(f"✓ Noise-assisted: optimal p*={best_p} outperforms noiseless "
                    f"(p=0 rmse={df[df.noise_rate==0.0].val_rmse.values[0]:.4f} "
                    f"vs p*={best_p} rmse={best_rmse:.4f})")
    else:
        logger.info(f"Noiseless is optimal (p*=0). No noise-assisted benefit observed.")

    with open(out_dir / "best_noise.json", "w") as f:
        json.dump({"p_star": best_p, "val_rmse": best_rmse,
                   "use_data_reuploading": use_data_reuploading}, f, indent=2)
    return best_p, df


# ── 3. Qubit scaling study ─────────────────────────────────────────────────────

def qubit_scaling_study(X_train: np.ndarray, y_train: np.ndarray,
                         X_val:   np.ndarray, y_val:   np.ndarray,
                         J: float, h: float, p: float,
                         qubit_counts: list = None,
                         use_data_reuploading: bool = None,
                         out_dir: Path = None) -> pd.DataFrame:
    """
    Run QRC across all configured qubit counts [5, 7, 9, 12, 16, 20].
    Challenge requires demonstrating performance across this range.
    Returns DataFrame with val_rmse per n_qubits.
    """
    from reservoir.quantum_reservoir import IsingQRC
    qubit_counts = qubit_counts or QUBIT_COUNTS
    out_dir      = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    use_data_reuploading = (USE_DATA_REUPLOADING if use_data_reuploading is None
                            else use_data_reuploading)

    logger.info(f"=== Qubit scaling study: n ∈ {qubit_counts} reuploading={use_data_reuploading} ===")

    rows = []
    for n in qubit_counts:
        qrc  = IsingQRC(n_qubits=n, J=J, h=h, noise_rate=p,
                        use_data_reuploading=use_data_reuploading)
        rmse = _quick_eval(qrc, X_train, y_train, X_val, y_val)
        feature_dim = 2 * n
        rows.append({"n_qubits": n, "feature_dim": feature_dim,
                     "val_rmse": rmse, "J": J, "h": h, "noise_rate": p,
                     "use_data_reuploading": use_data_reuploading})
        logger.info(f"  n={n:2d} features={feature_dim:3d} → val_rmse={rmse:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "qubit_scaling.csv", index=False)
    logger.info(f"Qubit scaling study saved → {out_dir / 'qubit_scaling.csv'}")
    return df


# ── 4. Shot budget ablation ────────────────────────────────────────────────────

def shot_ablation(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   J: float, h: float, p: float, n_qubits: int = 9,
                   use_data_reuploading: bool = None,
                   out_dir: Path = None) -> pd.DataFrame:
    """
    Compare performance at different shot budgets.
    shots=None → exact (infinite shots) simulation.
    Demonstrates NISQ resilience.
    """
    from reservoir.quantum_reservoir import IsingQRC
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    use_data_reuploading = (USE_DATA_REUPLOADING if use_data_reuploading is None
                            else use_data_reuploading)

    logger.info(f"=== Shot ablation: budgets={SHOT_COUNTS} noise_rate={p} "
                f"reuploading={use_data_reuploading} ===")

    rows = []
    for shots in SHOT_COUNTS:
        qrc  = IsingQRC(n_qubits=n_qubits, J=J, h=h, noise_rate=p, shots=shots,
                        use_data_reuploading=use_data_reuploading)
        rmse = _quick_eval(qrc, X_train, y_train, X_val, y_val)
        label = "∞ (exact)" if shots is None else str(shots)
        rows.append({"shots": label, "val_rmse": rmse})
        logger.info(f"  shots={label:<12} → val_rmse={rmse:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "shot_ablation.csv", index=False)
    return df


# ── 5. Topology comparison ─────────────────────────────────────────────────────

def topology_comparison(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          J: float, h: float, n_qubits: int = 9,
                          use_data_reuploading: bool = None,
                          out_dir: Path = None) -> tuple[str, pd.DataFrame]:
    """Compare chain vs all-to-all topology. Returns (best_topology, df)."""
    from reservoir.quantum_reservoir import IsingQRC
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    use_data_reuploading = (USE_DATA_REUPLOADING if use_data_reuploading is None
                            else use_data_reuploading)

    rows = []
    best_rmse = float("inf")
    best_topo = TOPOLOGIES[0]

    for topo in TOPOLOGIES:
        qrc  = IsingQRC(n_qubits=n_qubits, J=J, h=h, topology=topo,
                        use_data_reuploading=use_data_reuploading)
        rmse = _quick_eval(qrc, X_train, y_train, X_val, y_val)
        rows.append({"topology": topo, "val_rmse": rmse})
        logger.info(f"  topology={topo:<12} → val_rmse={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_topo = topo

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "topology_comparison.csv", index=False)
    with open(out_dir / "best_topology.json", "w") as f:
        json.dump({"topology_star": best_topo, "val_rmse": best_rmse,
                   "use_data_reuploading": use_data_reuploading}, f, indent=2)
    logger.info(f"Topology sweep done. Best: {best_topo} (val_rmse={best_rmse:.4f})")
    return best_topo, df