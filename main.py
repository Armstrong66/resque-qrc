"""
main.py — ResQue QRC Full Pipeline Orchestrator.

Runs the complete experiment sequence:
  1. Download NOAA ISD data
  2. Parse and preprocess
  3. Run all sweeps (Hamiltonian, noise, qubit scaling, shots)
  4. Train all baselines
  5. Train warm-start QRC with auto-selected readout
  6. Evaluate and produce results tables
  7. Save all outputs for download

Run locally (RTX workstation):
  python main.py

Run on qBraid (no changes needed — platform detected automatically):
  python main.py --platform qbraid

For quick smoke test (fast, small subset):
  python main.py --smoke_test
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (TARGETS, HORIZONS, QUBIT_PRIMARY, J_DEFAULT, H_DEFAULT,
                    RESULTS, FIGURES, LOGS, RANDOM_SEED, WINDOW_SIZE,
                    QUBIT_COUNTS)
from utils import get_logger

logger = get_logger("main")


def parse_args():
    p = argparse.ArgumentParser(description="ResQue QRC Pipeline")
    p.add_argument("--platform",    default="local",
                   choices=["local", "qbraid"],
                   help="Compute platform (affects device selection)")
    p.add_argument("--smoke_test",  action="store_true",
                   help="Quick test with small data subset")
    p.add_argument("--skip_download", action="store_true",
                   help="Skip download if raw files already exist")
    p.add_argument("--horizon",     type=int, default=None,
                   help="Run only one horizon (6 or 24). Default: both")
    p.add_argument("--n_qubits",    type=int, default=QUBIT_PRIMARY,
                   help=f"Primary qubit count (default: {QUBIT_PRIMARY})")
    p.add_argument("--skip_sweeps", action="store_true",
                   help="Skip parameter sweeps (use defaults from config)")
    p.add_argument("--skip_baselines", action="store_true",
                   help="Skip classical baselines (faster)")
    return p.parse_args()


def _load_sweep_results(out_dir: Path) -> tuple:
    """Load previously saved sweep results if available."""
    jh = out_dir / "best_hamiltonian.json"
    jn = out_dir / "best_noise.json"

    J, h, p = J_DEFAULT, H_DEFAULT, 0.0
    if jh.exists():
        d = json.load(open(jh))
        J, h = d["J_star"], d["h_star"]
        logger.info(f"Loaded J*={J} h*={h} from previous sweep")
    if jn.exists():
        d = json.load(open(jn))
        p = d["p_star"]
        logger.info(f"Loaded p*={p} from previous noise sweep")
    return J, h, p


def run_pipeline(args):
    start = time.time()
    logger.info("=" * 70)
    logger.info("ResQue QRC Pipeline — GIC 2026 Phase 2")
    logger.info(f"Platform: {args.platform}  Smoke test: {args.smoke_test}")
    logger.info("=" * 70)

    # ── 1. Data acquisition ────────────────────────────────────────────────
    logger.info("\n[1/6] Data acquisition")
    from data.downloader import download_all
    from data.parser import load_and_merge

    if not args.skip_download:
        raw_paths = download_all()
    else:
        from config import DATA_RAW, STATION_NAME
        raw_paths = sorted((DATA_RAW / STATION_NAME).glob("*.csv"))
        logger.info(f"Skipping download. Found {len(raw_paths)} cached files.")

    if not raw_paths:
        raise FileNotFoundError("No raw data files found. Remove --skip_download.")

    df = load_and_merge(raw_paths)
    logger.info(f"Loaded {len(df)} timesteps, columns: {list(df.columns)}")

    # ── 2. Preprocessing ───────────────────────────────────────────────────
    logger.info("\n[2/6] Preprocessing")
    from preprocessing.pipeline import WeatherPreprocessor

    if args.smoke_test:
        df = df.iloc[:500]
        logger.info("Smoke test: using first 500 timesteps")

    prep     = WeatherPreprocessor(df)
    horizons = [args.horizon] if args.horizon else HORIZONS
    datasets = {h: prep.build_dataset(h) for h in horizons}
    prep.save(datasets)

    # Use 6h horizon for sweeps (primary); 24h run afterwards
    ds_primary = datasets[horizons[0]]
    X_train, y_train = ds_primary.X_train, ds_primary.y_train
    X_val,   y_val   = ds_primary.X_val,   ds_primary.y_val
    X_test,  y_test  = ds_primary.X_test,  ds_primary.y_test

    logger.info(f"Primary dataset (h={horizons[0]}h): {ds_primary.summary()}")

    # ── 3. Parameter sweeps ────────────────────────────────────────────────
    logger.info("\n[3/6] Parameter sweeps")
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.skip_sweeps:
        logger.info("Skipping sweeps — loading from previous run or using defaults")
        J_star, h_star, p_star = _load_sweep_results(RESULTS)
    else:
        from experiments.sweeps import (hamiltonian_sweep, noise_sweep,
                                         qubit_scaling_study, shot_ablation,
                                         topology_comparison)
        try:
            J_star, h_star, _ = hamiltonian_sweep(
                X_train, y_train, X_val, y_val,
                n_qubits=args.n_qubits
            )
        except Exception as e:
            logger.error(f"Hamiltonian sweep failed: {e}\n{traceback.format_exc()}")
            J_star, h_star = J_DEFAULT, H_DEFAULT

        try:
            p_star, _ = noise_sweep(
                X_train, y_train, X_val, y_val,
                J=J_star, h=h_star, n_qubits=args.n_qubits
            )
        except Exception as e:
            logger.error(f"Noise sweep failed: {e}")
            p_star = 0.0

        try:
            qubit_scaling_study(X_train, y_train, X_val, y_val,
                                 J=J_star, h=h_star, p=p_star)
        except Exception as e:
            logger.error(f"Qubit scaling study failed: {e}")

        try:
            shot_ablation(X_train, y_train, X_val, y_val,
                          J=J_star, h=h_star, p=p_star, n_qubits=args.n_qubits)
        except Exception as e:
            logger.error(f"Shot ablation failed: {e}")

        try:
            topology_comparison(X_train, y_train, X_val, y_val,
                                 J=J_star, h=h_star, n_qubits=args.n_qubits)
        except Exception as e:
            logger.error(f"Topology comparison failed: {e}")

    logger.info(f"Using J*={J_star} h*={h_star} p*={p_star}")

    # ── 4. Classical baselines ─────────────────────────────────────────────
    logger.info("\n[4/6] Classical baselines")
    all_results = {}

    if not args.skip_baselines:
        from baselines.classical import (run_persistence, run_arima,
                                          run_esn, run_rnn)
        try:
            r = run_persistence(y_val, y_test, X_val, X_test, window=WINDOW_SIZE)
            all_results["persistence"] = r
            r.save(RESULTS)
        except Exception as e:
            logger.error(f"Persistence baseline failed: {e}")

        try:
            r = run_arima(y_train, y_val, y_test, TARGETS)
            if r:
                all_results["arima"] = r
                r.save(RESULTS)
        except Exception as e:
            logger.error(f"ARIMA baseline failed: {e}")

        # ESN — also provides warm-start weights
        try:
            r_esn, fitted_esn = run_esn(X_train, y_train, X_val, y_val,
                                         X_test, y_test)
            all_results["esn"] = r_esn
            r_esn.save(RESULTS)
            # Get ESN reservoir states aligned with training data (for warm-start)
            X_train_esn = fitted_esn.get_reservoir_states(X_train)
        except Exception as e:
            logger.error(f"ESN baseline failed: {e}")
            fitted_esn   = None
            X_train_esn  = None

        # LSTM
        try:
            r = run_rnn(X_train, y_train, X_val, y_val, X_test, y_test,
                        window=WINDOW_SIZE, model_type="lstm")
            if r:
                all_results["lstm"] = r
                r.save(RESULTS)
        except Exception as e:
            logger.error(f"LSTM baseline failed: {e}")

        # GRU (swappable — same config)
        try:
            r = run_rnn(X_train, y_train, X_val, y_val, X_test, y_test,
                        window=WINDOW_SIZE, model_type="gru")
            if r:
                all_results["gru"] = r
                r.save(RESULTS)
        except Exception as e:
            logger.error(f"GRU baseline failed: {e}")

    else:
        fitted_esn, X_train_esn = None, None
        logger.info("Baselines skipped.")

    # ── 5. QRC training ────────────────────────────────────────────────────
    logger.info("\n[5/6] QRC training — cold-start and warm-start")
    from reservoir.quantum_reservoir import IsingQRC
    from readout.ridge_readout import RidgeReadout

    for mode_label, use_warm in [("cold_start_qrc", False),
                                   ("warm_start_qrc",  True)]:
        logger.info(f"\n  Training: {mode_label}")
        try:
            qrc = IsingQRC(n_qubits=args.n_qubits, J=J_star, h=h_star,
                           noise_rate=p_star)

            H_train = qrc.run_sequence(X_train, verbose=not args.smoke_test)
            H_val   = qrc.run_sequence(X_val)
            H_test  = qrc.run_sequence(X_test)

            # Align shapes (warmup trimming)
            n_tr = min(len(H_train), len(y_train))
            n_vl = min(len(H_val),   len(y_val))
            n_ts = min(len(H_test),  len(y_test))

            readout = RidgeReadout(
                target_names = TARGETS,
                warm_start   = use_warm and (X_train_esn is not None),
            )
            best = readout.fit(
                H_train[:n_tr], y_train[:n_tr],
                H_val[:n_vl],   y_val[:n_vl],
                X_train_esn     = X_train_esn[:n_tr] if X_train_esn is not None else None,
            )
            readout.save_selection_log(RESULTS)

            # Predict on test set
            pred_test = best.predict(H_test[:n_ts])

            # Wrap as a simple object with y_pred attributes for metrics table
            class _QRCResult:
                def __init__(self, name, pred_val, pred_test):
                    self.name        = name
                    self.y_pred_val  = pred_val
                    self.y_pred_test = pred_test

            pred_val_qrc = best.predict(H_val[:n_vl])
            all_results[mode_label] = _QRCResult(
                mode_label, pred_val_qrc, pred_test
            )

            # Save weights
            best.save(RESULTS / f"{mode_label}_readout.pkl")
            qrc_cfg = qrc.get_config()
            qrc_cfg["readout_strategy"] = best.strategy
            qrc_cfg["warm_start"] = use_warm
            with open(RESULTS / f"{mode_label}_config.json", "w") as f:
                json.dump(qrc_cfg, f, indent=2)

        except Exception as e:
            logger.error(f"{mode_label} failed: {e}\n{traceback.format_exc()}")

    # ── 6. Evaluation ──────────────────────────────────────────────────────
    logger.info("\n[6/6] Evaluation and results tables")
    from evaluation.metrics import build_results_table

    for h_val, ds in datasets.items():
        try:
            # Pass full arrays — metrics.py now handles alignment internally
            build_results_table(
                results       = all_results,
                y_true_val    = ds.y_val,
                y_true_test   = ds.y_test,
                target_names  = TARGETS,
                horizon_hours = h_val,
                out_dir       = RESULTS,
            )
        except Exception as e:
            logger.error(f"Results table for h={h_val} failed: {e}")

    elapsed = time.time() - start
    logger.info(f"\n{'='*70}")
    logger.info(f"Pipeline complete in {elapsed/60:.1f} minutes.")
    logger.info(f"Outputs saved to: {RESULTS}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)