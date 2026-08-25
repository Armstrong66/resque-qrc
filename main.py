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
                    USE_SHARED_PCA, ESN_WARMUP, QRC_WARMUP, WARM_START_SOURCE,
                    USE_DATA_REUPLOADING, USE_SELECTED_NOISE_FOR_PRIMARY)
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
    p.add_argument("--use_selected_noise", action="store_true",
                   help="Apply calibrated p* to scaling and final QRC runs. "
                        "By default noise remains a separate robustness ablation.")
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
        _warn_if_stale_reuploading(jh.name, d)
    if jn.exists():
        d = json.load(open(jn))
        p = d["p_star"]
        _warn_if_stale_reuploading(jn.name, d)
    if jt.exists():
        d = json.load(open(jt))
        topology = d["topology_star"]
        _warn_if_stale_reuploading(jt.name, d)
    return J, h, p, topology


def _warn_if_stale_reuploading(filename: str, cached: dict):
    """
    Cached sweep JSON records the encoding it was produced under. J*/h*/p*
    found under standard encoding are not necessarily optimal under
    data-reuploading (and vice versa) — --skip_sweeps must not silently
    reuse stale winners after USE_DATA_REUPLOADING changes.
    """
    from config import USE_DATA_REUPLOADING
    cached_flag = cached.get("use_data_reuploading")
    if cached_flag is None:
        logger.warning(f"{filename} predates the reuploading flag being recorded — "
                       f"cannot verify it matches current USE_DATA_REUPLOADING="
                       f"{USE_DATA_REUPLOADING}. Re-run sweeps (drop --skip_sweeps) "
                       f"to regenerate it cleanly.")
    elif cached_flag != USE_DATA_REUPLOADING:
        logger.warning(f"{filename} was produced with use_data_reuploading="
                       f"{cached_flag}, but config.USE_DATA_REUPLOADING is now "
                       f"{USE_DATA_REUPLOADING}. These sweep winners (J*/h*/p*/topology*) "
                       f"are STALE for the current encoding — re-run sweeps "
                       f"(drop --skip_sweeps) before trusting this run's results.")


def _qrc_warmup(n: int) -> int:
    return min(QRC_WARMUP, max(0, n // 10))


def _align_after_warmup(H, y, warmup: int):
    y_a = y[warmup:]
    n = min(len(H), len(y_a))
    return H[:n], y_a[:n]


def _model_inputs(ds, n_qubits: int):
    from preprocessing.projection import maybe_project_dataset
    return maybe_project_dataset(ds, n_qubits)


def _resolve_warm_start_states(fitted_esn, fitted_rnn: dict, X_train):
    """
    Extract the warm-start hidden-state matrix from whichever classical model
    config.WARM_START_SOURCE points at. Source-agnostic by design so switching
    the warm-start layer (ESN/LSTM/GRU) is a one-line config.py edit — see
    config.WARM_START_SOURCE and docs/PROJECT_CRITIQUE.md.
    """
    from config import WARM_START_SOURCE
    source = WARM_START_SOURCE.lower()

    if source == "arima":
        raise ValueError(
            "WARM_START_SOURCE='arima' is invalid: ARIMA produces scalar "
            "per-target forecasts, not a reservoir-like hidden-state matrix, "
            "so it cannot warm-start the QRC ridge readout. Use 'esn', "
            "'lstm', or 'gru'."
        )
    if source == "esn":
        if fitted_esn is None:
            logger.warning("WARM_START_SOURCE='esn' but ESN did not fit — warm-start disabled")
            return None
        return fitted_esn.get_reservoir_states(X_train)
    if source in ("lstm", "gru"):
        extractor = fitted_rnn.get(source)
        if extractor is None:
            logger.warning(f"WARM_START_SOURCE='{source}' but {source.upper()} did not fit "
                           f"— warm-start disabled")
            return None
        return extractor.get_hidden_states(X_train)

    raise ValueError(f"Unknown WARM_START_SOURCE={WARM_START_SOURCE!r}; "
                     f"expected 'esn', 'lstm', or 'gru'.")


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
    fitted_rnn = {}
    baseline_status = {}   # model -> {"status": "ok"|"skipped"|"failed", "reason": str}

    if not args.skip_baselines:
        try:
            r = run_persistence(y_val, y_test, ds.X_val, ds.X_test, window=WINDOW_SIZE)
            all_results["persistence"] = r
            r.save(RESULTS / f"h{h_val}")
            baseline_status["persistence"] = {"status": "ok"}
        except Exception as e:
            logger.error(f"Persistence failed (h={h_val}h): {e}")
            baseline_status["persistence"] = {"status": "failed", "reason": str(e)}

        try:
            from baselines.classical import ARIMA_AVAILABLE
            r = run_arima(y_train, y_val, y_test, TARGETS)
            if r:
                all_results["arima"] = r
                r.save(RESULTS / f"h{h_val}")
                baseline_status["arima"] = {"status": "ok", "backend": r.meta.get("backend")}
            else:
                reason = ("no ARIMA backend installed (pip install statsmodels pmdarima)"
                          if not ARIMA_AVAILABLE else "run_arima returned no result")
                logger.error(f"ARIMA MISSING from results (h={h_val}h): {reason}")
                baseline_status["arima"] = {"status": "skipped", "reason": reason}
        except Exception as e:
            logger.error(f"ARIMA failed (h={h_val}h): {e}")
            baseline_status["arima"] = {"status": "failed", "reason": str(e)}

        try:
            r_esn, fitted_esn = run_esn(X_train, y_train, X_val, y_val, X_test, y_test)
            all_results["esn"] = r_esn
            r_esn.save(RESULTS / f"h{h_val}")
            baseline_status["esn"] = {"status": "ok"}
        except Exception as e:
            logger.error(f"ESN failed (h={h_val}h): {e}")
            baseline_status["esn"] = {"status": "failed", "reason": str(e)}

        for model_type in ("lstm", "gru"):
            try:
                r, wrapper = run_rnn(X_train, y_train, X_val, y_val, X_test, y_test,
                                      window=WINDOW_SIZE, model_type=model_type)
                if r:
                    all_results[model_type] = r
                    r.save(RESULTS / f"h{h_val}")
                    fitted_rnn[model_type] = wrapper
                    baseline_status[model_type] = {"status": "ok"}
                else:
                    baseline_status[model_type] = {"status": "skipped",
                                                   "reason": "PyTorch not installed"}
            except Exception as e:
                logger.error(f"{model_type.upper()} failed (h={h_val}h): {e}")
                baseline_status[model_type] = {"status": "failed", "reason": str(e)}

    out_h = RESULTS / f"h{h_val}"
    out_h.mkdir(parents=True, exist_ok=True)
    with open(out_h / "baseline_status.json", "w") as f:
        json.dump(baseline_status, f, indent=2)
    missing = [m for m, s in baseline_status.items() if s["status"] != "ok"]
    if missing:
        logger.warning(f"Baselines NOT in results_h{h_val}.csv: {missing} "
                       f"(see {out_h / 'baseline_status.json'} for reasons)")

    X_train_warm = _resolve_warm_start_states(fitted_esn, fitted_rnn, X_train)

    w = _qrc_warmup(len(X_train))

    try:
        # Cold and warm starts differ only in the readout; sharing these
        # deterministic trajectories preserves results while avoiding a
        # duplicate full reservoir execution.
        qrc = IsingQRC(n_qubits=args.n_qubits, J=J_star, h=h_star,
                       noise_rate=p_star, topology=topology_star,
                       use_data_reuploading=USE_DATA_REUPLOADING,
                       platform=args.platform)
        H_train = qrc.run_sequence(X_train, warmup=w, verbose=True)
        H_val = qrc.run_sequence(X_val, warmup=w, verbose=True)
        H_test = qrc.run_sequence(X_test, warmup=w, verbose=True)
        H_tr_base, y_tr_base = _align_after_warmup(H_train, y_train, w)
        H_vl, y_vl = _align_after_warmup(H_val, y_val, w)
        H_ts, y_ts = _align_after_warmup(H_test, y_test, w)
    except Exception as e:
        logger.error(f"QRC trajectory generation failed: {e}\n{traceback.format_exc()}")
        return all_results, w

    for mode_label, use_warm in [("cold_start_qrc", False), ("warm_start_qrc", True)]:
        logger.info(f"  QRC {mode_label} readout (h={h_val}h; shared trajectories)")
        try:
            H_tr, y_tr = H_tr_base.copy(), y_tr_base.copy()

            X_warm = None
            if use_warm and X_train_warm is not None:
                n_align = min(len(H_tr), len(X_train_warm))
                H_tr, y_tr = H_tr[-n_align:], y_tr[-n_align:]
                X_warm = X_train_warm[-n_align:]

            readout = RidgeReadout(
                target_names=TARGETS,
                warm_start=use_warm and (X_warm is not None),
            )
            best = readout.fit(H_tr, y_tr, H_vl, y_vl, X_train_warm_start=X_warm)
            readout.save_selection_log(RESULTS / f"h{h_val}")

            from baselines.classical import BaselineResult
            from evaluation.metrics import rmse_per_target
            pred_val, pred_test = best.predict(H_vl), best.predict(H_ts)
            qrc_result = BaselineResult(
                name=mode_label, y_pred_val=pred_val, y_pred_test=pred_test,
                val_rmse=rmse_per_target(y_vl, pred_val),
                test_rmse=rmse_per_target(y_ts, pred_test),
                meta={"type": mode_label, "readout_strategy": best.strategy},
                label_offset=w,
            )
            all_results[mode_label] = qrc_result

            out_h = RESULTS / f"h{h_val}"
            out_h.mkdir(parents=True, exist_ok=True)
            # Persisted (not just kept in-memory) so a separate later process —
            # e.g. agent_runner.py's benchmark_all task run standalone — can
            # reconstruct the full results table without re-driving the
            # reservoir. Matches how persistence/arima/esn/lstm/gru already save.
            qrc_result.save(out_h)
            best.save(out_h / f"{mode_label}_readout.pkl")
            cfg = qrc.get_config()
            cfg.update(horizon_hours=h_val, readout_strategy=best.strategy,
                       warm_start=use_warm, warm_start_source=WARM_START_SOURCE,
                       shared_pca=USE_SHARED_PCA)
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
    logger.info(f"Reuploading: {USE_DATA_REUPLOADING}  Warm-start source: {WARM_START_SOURCE}")
    logger.info("=" * 70)

    from baselines.classical import ARIMA_AVAILABLE, TORCH_AVAILABLE
    if not ARIMA_AVAILABLE:
        logger.error("ARIMA baseline will be ABSENT from all results tables this run "
                     "(no pmdarima or statsmodels installed). "
                     "pip install statsmodels pmdarima to include it.")
    if not TORCH_AVAILABLE:
        logger.error("LSTM/GRU baselines will be ABSENT from all results tables this run "
                     "(PyTorch not installed). pip install torch to include them.")

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
        p_primary = p_star if (args.use_selected_noise or USE_SELECTED_NOISE_FOR_PRIMARY) else 0.0
        if p_primary != p_star:
            logger.info("Recorded calibrated p*=%.4f as a noise-robustness ablation; "
                        "using p=0.0 for scaling and primary forecasts. "
                        "Pass --use_selected_noise for the separate noisy protocol.", p_star)
        try:
            qubit_scaling_study(X_q_tr, ds_primary.y_train, X_q_vl, ds_primary.y_val,
                                J=J_star, h=h_star, p=p_primary)
        except Exception as e:
            logger.error(f"Qubit scaling failed: {e}")
        try:
            shot_ablation(X_q_tr, ds_primary.y_train, X_q_vl, ds_primary.y_val,
                          J=J_star, h=h_star, p=p_primary, n_qubits=args.n_qubits)
        except Exception as e:
            logger.error(f"Shot ablation failed: {e}")
        try:
            topology_star, _ = topology_comparison(
                X_q_tr, ds_primary.y_train, X_q_vl, ds_primary.y_val,
                J=J_star, h=h_star, n_qubits=args.n_qubits)
        except Exception as e:
            logger.error(f"Topology comparison failed: {e}")

    p_primary = p_star if (args.use_selected_noise or USE_SELECTED_NOISE_FOR_PRIMARY) else 0.0
    logger.info(f"Sweep: J*={J_star} h*={h_star} p*_calibration={p_star} "
                f"p_primary={p_primary} topo={topology_star}")

    logger.info("\n[4-6/6] Per-horizon train and evaluate")
    for h_val in horizons:
        ds = datasets[h_val]
        try:
            all_results, _ = _train_horizon(
                h_val, ds, args, J_star, h_star, p_primary, topology_star)
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
