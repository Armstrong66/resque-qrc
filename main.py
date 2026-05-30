"""
main.py — ResQue QRC Full Pipeline Orchestrator.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (TARGETS, HORIZONS, QUBIT_PRIMARY, J_DEFAULT, H_DEFAULT,
                    TOPOLOGY_PRIMARY, RESULTS, WINDOW_SIZE,
                    USE_SHARED_PCA, ESN_WARMUP, QRC_WARMUP)
from utils import get_logger

logger = get_logger("main")


def parse_args():
    p = argparse.ArgumentParser(description="ResQue QRC Pipeline")
    p.add_argument("--platform", default="local", choices=["local", "qbraid"],
                   help="Compute platform (qbraid tries lightning.qubit)")
    p.add_argument("--smoke_test", action="store_true",
                   help="Quick test: 500 steps, auto --skip_sweeps")
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--n_qubits", type=int, default=QUBIT_PRIMARY)
    p.add_argument("--skip_sweeps", action="store_true")
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--force_rebuild_data", action="store_true")
    p.add_argument("--no_figures", action="store_true",
                   help="Skip generating plots after run")
    return p.parse_args()


def _load_sweep_results(out_dir: Path) -> tuple:
    jh, jn, jt = out_dir / "best_hamiltonian.json", out_dir / "best_noise.json", out_dir / "best_topology.json"
    J, h, p, topology = J_DEFAULT, H_DEFAULT, 0.0, TOPOLOGY_PRIMARY
    if jh.exists():
        d = json.load(open(jh))
        J, h = d["J_star"], d["h_star"]
    if jn.exists():
        p = json.load(open(jn))["p_star"]
    if jt.exists():
        topology = json.load(open(jt))["topology_star"]
    return J, h, p, topology


def _qrc_warmup(n: int) -> int:
    return min(QRC_WARMUP, max(0, n // 10))


def _align_after_warmup(H, y, warmup: int):
    y_a = y[warmup:]
    n = min(len(H), len(y_a))
    return H[:n], y_a[:n]


def _model_inputs(ds, n_qubits: int):
    from preprocessing.projection import maybe_project_dataset
    return maybe_project_dataset(ds, n_qubits)


def _train_horizon(h_val, ds, args, J_star, h_star, p_star, topology_star):
    from baselines.classical import run_persistence, run_arima, run_esn, run_rnn
    from reservoir.quantum_reservoir import IsingQRC
    from readout.ridge_readout import RidgeReadout

    logger.info(f"\n{'-'*70}")
    logger.info(f"Horizon {h_val}h — {ds.summary()}")
    logger.info(f"{'-'*70}")

    X_train, X_val, X_test, proj = _model_inputs(ds, args.n_qubits)
    y_train, y_val, y_test = ds.y_train, ds.y_val, ds.y_test

    if proj is not None:
        path = RESULTS / f"shared_pca_h{h_val}.pkl"
        proj.save(path)
        logger.info(f"Shared PCA saved -> {path} (USE_SHARED_PCA={USE_SHARED_PCA})")

    all_results = {}
    fitted_esn = None
    X_train_esn = None

    if not args.skip_baselines:
        try:
            r = run_persistence(y_val, y_test, ds.X_val, ds.X_test, window=WINDOW_SIZE)
            all_results["persistence"] = r
            r.save(RESULTS / f"h{h_val}")
        except Exception as e:
            logger.error(f"Persistence failed (h={h_val}h): {e}")

        try:
            r = run_arima(y_train, y_val, y_test, TARGETS)
            if r:
                all_results["arima"] = r
                r.save(RESULTS / f"h{h_val}")
        except Exception as e:
            logger.error(f"ARIMA failed (h={h_val}h): {e}")

        try:
            r_esn, fitted_esn = run_esn(X_train, y_train, X_val, y_val, X_test, y_test)
            all_results["esn"] = r_esn
            r_esn.save(RESULTS / f"h{h_val}")
            X_train_esn = fitted_esn.get_reservoir_states(X_train)
        except Exception as e:
            logger.error(f"ESN failed (h={h_val}h): {e}")

        for model_type in ("lstm", "gru"):
            try:
                r = run_rnn(X_train, y_train, X_val, y_val, X_test, y_test,
                            window=WINDOW_SIZE, model_type=model_type)
                if r:
                    all_results[model_type] = r
                    r.save(RESULTS / f"h{h_val}")
            except Exception as e:
                logger.error(f"{model_type.upper()} failed (h={h_val}h): {e}")

    w = _qrc_warmup(len(X_train))

    for mode_label, use_warm in [("cold_start_qrc", False), ("warm_start_qrc", True)]:
        logger.info(f"  QRC {mode_label} (h={h_val}h)")
        try:
            qrc = IsingQRC(
                n_qubits=args.n_qubits, J=J_star, h=h_star,
                noise_rate=p_star, topology=topology_star,
                platform=args.platform,
            )
            H_train = qrc.run_sequence(X_train, warmup=w)
            H_val = qrc.run_sequence(X_val, warmup=w)
            H_test = qrc.run_sequence(X_test, warmup=w)

            H_tr, y_tr = _align_after_warmup(H_train, y_train, w)
            H_vl, y_vl = _align_after_warmup(H_val, y_val, w)
            H_ts, y_ts = _align_after_warmup(H_test, y_test, w)

            X_esn = None
            if use_warm and X_train_esn is not None:
                n_align = min(len(H_tr), len(X_train_esn))
                H_tr, y_tr = H_tr[-n_align:], y_tr[-n_align:]
                X_esn = X_train_esn[-n_align:]

            readout = RidgeReadout(
                target_names=TARGETS,
                warm_start=use_warm and (X_esn is not None),
            )
            best = readout.fit(H_tr, y_tr, H_vl, y_vl, X_train_esn=X_esn)
            readout.save_selection_log(RESULTS / f"h{h_val}")

            class _QRCResult:
                def __init__(self, name, pv, pt, offset):
                    self.name = name
                    self.y_pred_val = pv
                    self.y_pred_test = pt
                    self.label_offset = offset

            all_results[mode_label] = _QRCResult(
                mode_label, best.predict(H_vl), best.predict(H_ts), w
            )

            out_h = RESULTS / f"h{h_val}"
            out_h.mkdir(parents=True, exist_ok=True)
            best.save(out_h / f"{mode_label}_readout.pkl")
            cfg = qrc.get_config()
            cfg.update(horizon_hours=h_val, readout_strategy=best.strategy,
                       warm_start=use_warm, shared_pca=USE_SHARED_PCA)
            with open(out_h / f"{mode_label}_config.json", "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            logger.error(f"{mode_label} failed: {e}\n{traceback.format_exc()}")

    return all_results, w


def run_pipeline(args):
    start = time.time()
    logger.info("=" * 70)
    logger.info("ResQue QRC Pipeline — GIC 2026")
    logger.info(f"Platform: {args.platform}  Smoke: {args.smoke_test}  Shared PCA: {USE_SHARED_PCA}")
    logger.info("=" * 70)

    from data.downloader import download_all
    from data.parser import load_and_merge
    from preprocessing.pipeline import WeatherPreprocessor
    from evaluation.metrics import build_results_table

    logger.info("\n[1/6] Data acquisition")
    if not args.skip_download:
        raw_paths = download_all()
    else:
        from config import DATA_RAW, STATION_NAME
        raw_paths = sorted((DATA_RAW / STATION_NAME).glob("*.csv"))
    if not raw_paths:
        raise FileNotFoundError("No raw data files found.")

    df = load_and_merge(raw_paths, force_rebuild=args.force_rebuild_data)
    logger.info(f"Loaded {len(df)} timesteps")

    logger.info("\n[2/6] Preprocessing")
    if args.smoke_test:
        df = df.iloc[:500]
    prep = WeatherPreprocessor(df)
    horizons = [args.horizon] if args.horizon else HORIZONS
    datasets = {h: prep.build_dataset(h) for h in horizons}
    prep.save(datasets)

    RESULTS.mkdir(parents=True, exist_ok=True)
    h_primary = horizons[0]
    ds_primary = datasets[h_primary]
    X_q_tr, X_q_vl, _, _ = _model_inputs(ds_primary, args.n_qubits)

    logger.info("\n[3/6] Parameter sweeps")
    if args.skip_sweeps:
        J_star, h_star, p_star, topology_star = _load_sweep_results(RESULTS)
    else:
        from experiments.sweeps import (hamiltonian_sweep, noise_sweep,
                                         qubit_scaling_study, shot_ablation,
                                         topology_comparison)
        J_star, h_star, p_star, topology_star = J_DEFAULT, H_DEFAULT, 0.0, TOPOLOGY_PRIMARY
        try:
            J_star, h_star, _ = hamiltonian_sweep(
                X_q_tr, ds_primary.y_train, X_q_vl, ds_primary.y_val, n_qubits=args.n_qubits)
        except Exception as e:
            logger.error(f"Hamiltonian sweep failed: {e}")
        try:
            p_star, _ = noise_sweep(
                X_q_tr, ds_primary.y_train, X_q_vl, ds_primary.y_val,
                J=J_star, h=h_star, n_qubits=args.n_qubits)
        except Exception as e:
            logger.error(f"Noise sweep failed: {e}")
        try:
            qubit_scaling_study(X_q_tr, ds_primary.y_train, X_q_vl, ds_primary.y_val,
                                J=J_star, h=h_star, p=p_star)
        except Exception as e:
            logger.error(f"Qubit scaling failed: {e}")
        try:
            shot_ablation(X_q_tr, ds_primary.y_train, X_q_vl, ds_primary.y_val,
                          J=J_star, h=h_star, p=p_star, n_qubits=args.n_qubits)
        except Exception as e:
            logger.error(f"Shot ablation failed: {e}")
        try:
            topology_star, _ = topology_comparison(
                X_q_tr, ds_primary.y_train, X_q_vl, ds_primary.y_val,
                J=J_star, h=h_star, n_qubits=args.n_qubits)
        except Exception as e:
            logger.error(f"Topology comparison failed: {e}")

    logger.info(f"Sweep: J*={J_star} h*={h_star} p*={p_star} topo={topology_star}")

    logger.info("\n[4-6/6] Per-horizon train and evaluate")
    for h_val in horizons:
        ds = datasets[h_val]
        try:
            all_results, _ = _train_horizon(
                h_val, ds, args, J_star, h_star, p_star, topology_star)
            build_results_table(
                results=all_results,
                y_true_val=ds.y_val,
                y_true_test=ds.y_test,
                target_names=TARGETS,
                horizon_hours=h_val,
                out_dir=RESULTS,
                dataset=ds,
            )
        except Exception as e:
            logger.error(f"Horizon {h_val}h failed: {e}\n{traceback.format_exc()}")

    if not args.no_figures:
        try:
            from evaluation.figures import generate_all_figures
            generate_all_figures(RESULTS)
        except Exception as e:
            logger.error(f"Figure generation failed: {e}")

    logger.info(f"\nPipeline complete in {(time.time()-start)/60:.1f} min. Outputs: {RESULTS}")


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        args.skip_sweeps = True
        logger.info("Smoke test: enabled --skip_sweeps automatically")
    run_pipeline(args)
