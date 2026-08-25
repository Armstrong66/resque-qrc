"""
evaluation/figures.py — Plots for sweep outputs (saved to outputs/figures/).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIGURES, RESULTS
from utils import get_logger

logger = get_logger(__name__)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping figures")
        return None


def plot_hamiltonian_heatmap(csv_path: Path = None, out_path: Path = None):
    plt = _require_matplotlib()
    if plt is None:
        return
    import pandas as pd

    if csv_path is None:
        candidates = sorted(RESULTS.glob("hamiltonian_sweep_h*_*.csv"))
        csv_path = candidates[0] if candidates else (RESULTS / "hamiltonian_sweep.csv")
    if not csv_path.exists():
        logger.debug(f"No hamiltonian sweep at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="J", columns="h", values="val_rmse")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2g}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:.2g}" for x in pivot.index])
    ax.set_xlabel("h")
    ax.set_ylabel("J")
    ax.set_title("Hamiltonian sweep (val RMSE)")
    fig.colorbar(im, ax=ax, label="val_rmse")
    out_path = out_path or (FIGURES / "hamiltonian_heatmap.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved -> {out_path}")


def plot_noise_sweep(csv_path: Path = None, out_path: Path = None):
    plt = _require_matplotlib()
    if plt is None:
        return
    import pandas as pd

    csv_path = csv_path or (RESULTS / "noise_sweep.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["noise_rate"], df["val_rmse"], "o-")
    ax.set_xlabel("Depolarizing rate p")
    ax.set_ylabel("Val RMSE (mean)")
    ax.set_title("Noise-assisted QRC sweep")
    out_path = out_path or (FIGURES / "noise_sweep.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved -> {out_path}")


def plot_qubit_scaling(csv_path: Path = None, out_path: Path = None):
    plt = _require_matplotlib()
    if plt is None:
        return
    import pandas as pd

    csv_path = csv_path or (RESULTS / "qubit_scaling.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["n_qubits"], df["val_rmse"], "s-")
    ax.set_xlabel("Qubit count")
    ax.set_ylabel("Val RMSE (mean)")
    ax.set_title("Qubit scaling study")
    out_path = out_path or (FIGURES / "qubit_scaling.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved -> {out_path}")


def plot_results_bar(csv_path: Path = None, out_path: Path = None):
    plt = _require_matplotlib()
    if plt is None:
        return
    import pandas as pd

    csv_path = csv_path or (RESULTS / "results_h6.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path).sort_values("test_rmse_mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["model"], df["test_rmse_mean"])
    ax.set_xlabel("Test RMSE (mean, normalized)")
    ax.set_title(f"Model comparison ({csv_path.stem})")
    out_path = out_path or (FIGURES / f"{csv_path.stem}_bar.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved -> {out_path}")


def generate_all_figures(results_dir: Path = None):
    """Generate all plots from existing sweep/result CSVs."""
    results_dir = results_dir or RESULTS
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Paired architecture selection writes one grid per horizon and encoding.
    # Plot each grid separately rather than mixing two encodings in one pivot.
    for csv_path in sorted(results_dir.glob("hamiltonian_sweep_h*_*.csv")):
        plot_hamiltonian_heatmap(
            csv_path, FIGURES / f"{csv_path.stem}_heatmap.png"
        )
    plot_noise_sweep(results_dir / "noise_sweep.csv")
    plot_qubit_scaling(results_dir / "qubit_scaling.csv")
    for p in sorted(results_dir.glob("results_h*.csv")):
        plot_results_bar(p, FIGURES / f"{p.stem}_bar.png")
