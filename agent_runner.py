"""
agent_runner.py — Agentic interface for the ResQue QRC pipeline.

Single point of entry for any automated executor (qBraid AI agent, judge's
reproduction script, CI). Reads a task name and an optional JSON config
override, runs the corresponding pipeline step using the SAME underlying
modules main.py uses (this file adds no new model/data logic of its own —
see docs/AGENTIC_DESIGN_GUIDE.md §8, "one pipeline, multiple interfaces"),
and writes a structured, machine-parseable result.

Usage:
    python agent_runner.py --task <task_name> [--config '<json_string>']

Tasks:
    setup              Verify environment (imports, optional-dep availability)
    download_data      Download NOAA ISD CSVs
    preprocess         Parse, clean, window, split (all HORIZONS)
    eda                Run data quality audit -> outputs/results/eda_report.json
    hamiltonian_sweep  Sweep J x h -> best_hamiltonian.json
    noise_sweep        Sweep depolarizing rate -> best_noise.json
    topology_sweep     Chain vs all-to-all -> best_topology.json
    encode_ablation    Standard vs. data-reuploading comparison
    qubit_scaling      Scale n=5..20 -> qubit_scaling.csv
    shot_ablation      Vary shot budget -> shot_ablation.csv
    train_baselines    Persistence + ARIMA + ESN + LSTM + GRU, all horizons
    train_qrc          Cold-start and warm-start QRC, all horizons
    benchmark_all      Full results tables, all horizons
    hardware_validation  Small subsampled real/simulator QRC comparison
                          (see scripts/hardware_validation.py)
    verify_results     Check all expected outputs exist and are NaN-free
    full_run           Runs the tasks above in the order documented in
                        docs/AGENTIC_DESIGN_GUIDE.md §4.1

Every task is wrapped in try/except: failure writes {"status": "failed",
"error": "..."} rather than raising past this script, so a calling agent can
always read a result. Every task prints `AGENT_RESULT: <json>` as its last
line of stdout — that is the contract a calling agent parses, not the log
lines above it. Each task is idempotent: re-running it just overwrites its
own output files with a fresh result, matching the already-existing
behaviour of the underlying pipeline functions (main.py works the same way).
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (TARGETS, HORIZONS, QUBIT_PRIMARY, J_DEFAULT, H_DEFAULT,
                    TOPOLOGY_PRIMARY, RESULTS, WINDOW_SIZE, WARM_START_SOURCE)
from utils import get_logger

logger = get_logger("agent_runner")

AGENT_TASK_DIR = RESULTS / "agent_tasks"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_task_result(task: str, result: dict):
    AGENT_TASK_DIR.mkdir(parents=True, exist_ok=True)
    with open(AGENT_TASK_DIR / f"{task}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


# ── Individual tasks ────────────────────────────────────────────────────────
# Every _task_* function returns a plain dict — no side-channel state.
# Config overrides (from --config) are passed through as explicit kwargs to
# the underlying pipeline function, never by mutating config.py globals.

def _task_setup(cfg: dict) -> dict:
    import numpy
    checks = {}
    try:
        import pennylane as qml
        checks["pennylane"] = qml.__version__
    except ImportError:
        checks["pennylane"] = None
    try:
        import torch
        checks["torch"] = torch.__version__
        checks["torch_cuda"] = bool(torch.cuda.is_available())
        # torch built against the NumPy 1.x C-API raises at the FIRST real
        # tensor<->numpy conversion, not at import time — import succeeding
        # does not mean LSTM/GRU baselines will actually work. Catch this
        # here, once, cheaply, rather than mid-pipeline hours into a run.
        try:
            torch.from_numpy(numpy.zeros(1, dtype=numpy.float32)).numpy()
            checks["torch_numpy_interop"] = True
        except Exception as e:
            checks["torch_numpy_interop"] = False
            checks["torch_numpy_interop_error"] = str(e)
    except ImportError:
        checks["torch"] = None
    try:
        from baselines.classical import ARIMA_AVAILABLE, PMDARIMA_AVAILABLE, STATSMODELS_AVAILABLE
        checks["arima_available"] = ARIMA_AVAILABLE
        checks["pmdarima_available"] = PMDARIMA_AVAILABLE
        checks["statsmodels_available"] = STATSMODELS_AVAILABLE
    except Exception as e:
        checks["arima_check_error"] = str(e)
    import pandas
    checks["numpy"] = numpy.__version__
    checks["pandas"] = pandas.__version__

    missing = [k for k, v in checks.items()
              if v is None or (isinstance(v, bool) and not v and "available" in k)]
    ready = checks.get("pennylane") is not None
    warnings_ = []
    if checks.get("torch") is not None and checks.get("torch_numpy_interop") is False:
        warnings_.append(
            "torch is importable but torch<->numpy conversion is BROKEN "
            "(likely a torch build predating this numpy's C-API — e.g. torch "
            "compiled for NumPy 1.x under NumPy 2.x). LSTM/GRU baselines will "
            "fail at prediction time, not at import time. Fix: reinstall a "
            "numpy-2-compatible torch build (torch>=2.3 generally works), or "
            "pin numpy<2 to match an older torch. See docs/PROJECT_CRITIQUE.md."
        )
    return {"checks": checks, "missing": missing, "warnings": warnings_, "ready": ready}


def _task_download_data(cfg: dict) -> dict:
    from data.downloader import download_all
    paths = download_all()
    return {"n_files": len(paths), "paths": [str(p) for p in paths]}


def _task_preprocess(cfg: dict) -> dict:
    from data.downloader import download_all
    from data.parser import load_and_merge
    from preprocessing.pipeline import WeatherPreprocessor
    from config import DATA_RAW, STATION_NAME

    raw_paths = sorted((DATA_RAW / STATION_NAME).glob("*.csv"))
    if not raw_paths:
        raw_paths = download_all()
    df = load_and_merge(raw_paths, force_rebuild=cfg.get("force_rebuild_data", False))
    prep = WeatherPreprocessor(df)
    horizons = cfg.get("horizons", HORIZONS)
    datasets = {h: prep.build_dataset(h) for h in horizons}
    prep.save(datasets)
    return {"n_timesteps": len(df), "horizons": horizons,
           "datasets": {h: ds.summary() for h, ds in datasets.items()}}


def _task_eda(cfg: dict) -> dict:
    from config import DATA_RAW, DATA_PROC, STATION_NAME
    from eda.inspect_data import (inspect_raw, inspect_processed,
                                   inspect_datasets, inspect_encoding_bottleneck)
    report = {
        "raw": inspect_raw(DATA_RAW / STATION_NAME),
        "processed": inspect_processed(DATA_PROC / STATION_NAME),
        "datasets": inspect_datasets(DATA_PROC / STATION_NAME),
        "encoding": inspect_encoding_bottleneck(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "eda_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def _primary_qrc_inputs(cfg: dict):
    """Shared setup for every sweep/ablation task: load the primary-horizon
    dataset (building it if not cached) and its shared-PCA projection."""
    from preprocessing.pipeline import WeatherPreprocessor
    from preprocessing.projection import maybe_project_dataset

    horizon = cfg.get("horizon", HORIZONS[0])
    try:
        ds = WeatherPreprocessor.load(horizon)
    except FileNotFoundError:
        _task_preprocess({"horizons": [horizon]})
        ds = WeatherPreprocessor.load(horizon)
    n_qubits = cfg.get("n_qubits", QUBIT_PRIMARY)
    X_tr, X_vl, X_ts, proj = maybe_project_dataset(ds, n_qubits)
    return ds, X_tr, X_vl, X_ts, n_qubits


def _task_hamiltonian_sweep(cfg: dict) -> dict:
    from experiments.sweeps import hamiltonian_sweep
    ds, X_tr, X_vl, _, n_qubits = _primary_qrc_inputs(cfg)
    J, h, df = hamiltonian_sweep(
        X_tr, ds.y_train, X_vl, ds.y_val, n_qubits=n_qubits,
        use_data_reuploading=cfg.get("use_data_reuploading"))
    return {"J_star": J, "h_star": h, "n_configs": len(df), "n_qubits": n_qubits}


def _task_noise_sweep(cfg: dict) -> dict:
    from experiments.sweeps import noise_sweep
    from main import _load_sweep_results
    ds, X_tr, X_vl, _, n_qubits = _primary_qrc_inputs(cfg)
    J_star, h_star, _, _ = _load_sweep_results(RESULTS)
    J = cfg.get("J", J_star)
    h = cfg.get("h", h_star)
    p, df = noise_sweep(X_tr, ds.y_train, X_vl, ds.y_val, J=J, h=h,
                        n_qubits=n_qubits,
                        use_data_reuploading=cfg.get("use_data_reuploading"))
    return {"p_star": p, "n_configs": len(df), "J": J, "h": h}


def _task_topology_sweep(cfg: dict) -> dict:
    from experiments.sweeps import topology_comparison
    from main import _load_sweep_results
    ds, X_tr, X_vl, _, n_qubits = _primary_qrc_inputs(cfg)
    J_star, h_star, _, _ = _load_sweep_results(RESULTS)
    J = cfg.get("J", J_star)
    h = cfg.get("h", h_star)
    topo, df = topology_comparison(X_tr, ds.y_train, X_vl, ds.y_val, J=J, h=h,
                                   n_qubits=n_qubits,
                                   use_data_reuploading=cfg.get("use_data_reuploading"))
    return {"topology_star": topo, "n_configs": len(df)}


def _task_encode_ablation(cfg: dict) -> dict:
    from reservoir.quantum_reservoir import encoding_ablation
    from main import _load_sweep_results
    ds, X_tr, X_vl, _, n_qubits = _primary_qrc_inputs(cfg)
    J_star, h_star, _, _ = _load_sweep_results(RESULTS)
    result = encoding_ablation(
        X_tr, ds.y_train, X_vl, ds.y_val, n_qubits=n_qubits,
        J=cfg.get("J", J_star), h=cfg.get("h", h_star),
        max_steps=cfg.get("max_steps", 300), out_dir=RESULTS)
    return result


def _task_qubit_scaling(cfg: dict) -> dict:
    from experiments.sweeps import qubit_scaling_study
    from main import _load_sweep_results
    ds, X_tr, X_vl, _, n_qubits = _primary_qrc_inputs(cfg)
    J_star, h_star, p_star, _ = _load_sweep_results(RESULTS)
    df = qubit_scaling_study(
        X_tr, ds.y_train, X_vl, ds.y_val,
        J=cfg.get("J", J_star), h=cfg.get("h", h_star), p=cfg.get("p", p_star),
        qubit_counts=cfg.get("qubit_counts"),
        use_data_reuploading=cfg.get("use_data_reuploading"))
    return {"n_configs": len(df), "qubit_counts": df["n_qubits"].tolist()}


def _task_shot_ablation(cfg: dict) -> dict:
    from experiments.sweeps import shot_ablation
    from main import _load_sweep_results
    ds, X_tr, X_vl, _, n_qubits = _primary_qrc_inputs(cfg)
    J_star, h_star, p_star, _ = _load_sweep_results(RESULTS)
    df = shot_ablation(
        X_tr, ds.y_train, X_vl, ds.y_val,
        J=cfg.get("J", J_star), h=cfg.get("h", h_star), p=cfg.get("p", p_star),
        n_qubits=n_qubits, use_data_reuploading=cfg.get("use_data_reuploading"))
    return {"n_configs": len(df)}


def _fit_baselines_and_warm_start(ds, X_train, X_val, X_test, horizon: int):
    """
    Shared by train_baselines and train_qrc so both can be run standalone
    without duplicating the "fit ESN/ARIMA/LSTM/GRU" logic — the fitting
    itself still happens in exactly one place: baselines/classical.py.
    Returns (all_results, baseline_status, X_train_warm, w).
    """
    from baselines.classical import run_persistence, run_arima, run_esn, run_rnn, ARIMA_AVAILABLE
    from main import _resolve_warm_start_states, _qrc_warmup

    y_train, y_val, y_test = ds.y_train, ds.y_val, ds.y_test
    all_results, fitted_esn, fitted_rnn, baseline_status = {}, None, {}, {}

    try:
        r = run_persistence(y_val, y_test, ds.X_val, ds.X_test, window=WINDOW_SIZE)
        all_results["persistence"] = r
        baseline_status["persistence"] = {"status": "ok"}
    except Exception as e:
        baseline_status["persistence"] = {"status": "failed", "reason": str(e)}

    try:
        r = run_arima(y_train, y_val, y_test, TARGETS)
        if r:
            all_results["arima"] = r
            baseline_status["arima"] = {"status": "ok", "backend": r.meta.get("backend")}
        else:
            baseline_status["arima"] = {"status": "skipped",
                                        "reason": "no backend" if not ARIMA_AVAILABLE else "unknown"}
    except Exception as e:
        baseline_status["arima"] = {"status": "failed", "reason": str(e)}

    try:
        r_esn, fitted_esn = run_esn(X_train, y_train, X_val, y_val, X_test, y_test)
        all_results["esn"] = r_esn
        baseline_status["esn"] = {"status": "ok"}
    except Exception as e:
        baseline_status["esn"] = {"status": "failed", "reason": str(e)}

    for model_type in ("lstm", "gru"):
        try:
            r, wrapper = run_rnn(X_train, y_train, X_val, y_val, X_test, y_test,
                                 window=WINDOW_SIZE, model_type=model_type)
            if r:
                all_results[model_type] = r
                fitted_rnn[model_type] = wrapper
                baseline_status[model_type] = {"status": "ok"}
            else:
                baseline_status[model_type] = {"status": "skipped", "reason": "PyTorch not installed"}
        except Exception as e:
            baseline_status[model_type] = {"status": "failed", "reason": str(e)}

    out_h = RESULTS / f"h{horizon}"
    out_h.mkdir(parents=True, exist_ok=True)
    for name, r in all_results.items():
        r.save(out_h)
    with open(out_h / "baseline_status.json", "w") as f:
        json.dump(baseline_status, f, indent=2)

    X_train_warm = _resolve_warm_start_states(fitted_esn, fitted_rnn, X_train)
    w = _qrc_warmup(len(X_train))
    return all_results, baseline_status, X_train_warm, w


def _task_train_baselines(cfg: dict) -> dict:
    from preprocessing.pipeline import WeatherPreprocessor
    from main import _model_inputs

    horizons = cfg.get("horizons", HORIZONS)
    per_horizon = {}
    for horizon in horizons:
        try:
            ds = WeatherPreprocessor.load(horizon)
        except FileNotFoundError:
            _task_preprocess({"horizons": [horizon]})
            ds = WeatherPreprocessor.load(horizon)
        X_tr, X_vl, X_ts, _ = _model_inputs(ds, cfg.get("n_qubits", QUBIT_PRIMARY))
        _, status, _, _ = _fit_baselines_and_warm_start(ds, X_tr, X_vl, X_ts, horizon)
        per_horizon[horizon] = status
    return {"horizons": horizons, "baseline_status": per_horizon}


def _task_train_qrc(cfg: dict) -> dict:
    from preprocessing.pipeline import WeatherPreprocessor
    from reservoir.quantum_reservoir import IsingQRC
    from readout.ridge_readout import RidgeReadout
    from main import _model_inputs, _align_after_warmup, _load_sweep_results
    from config import RESULTS as _RESULTS, USE_SHARED_PCA, USE_DATA_REUPLOADING

    horizons = cfg.get("horizons", HORIZONS)
    J_star, h_star, p_star, topology_star = _load_sweep_results(RESULTS)
    J_star = cfg.get("J", J_star)
    h_star = cfg.get("h", h_star)
    n_qubits = cfg.get("n_qubits", QUBIT_PRIMARY)

    per_horizon = {}
    for horizon in horizons:
        try:
            ds = WeatherPreprocessor.load(horizon)
        except FileNotFoundError:
            _task_preprocess({"horizons": [horizon]})
            ds = WeatherPreprocessor.load(horizon)
        X_train, X_val, X_test, proj = _model_inputs(ds, n_qubits)
        y_train, y_val, y_test = ds.y_train, ds.y_val, ds.y_test
        if proj is not None:
            proj.save(RESULTS / f"shared_pca_h{horizon}.pkl")

        _, _, X_train_warm, w = _fit_baselines_and_warm_start(ds, X_train, X_val, X_test, horizon)

        modes = {}
        for mode_label, use_warm in [("cold_start_qrc", False), ("warm_start_qrc", True)]:
            qrc = IsingQRC(n_qubits=n_qubits, J=J_star, h=h_star, noise_rate=p_star,
                          topology=topology_star,
                          use_data_reuploading=cfg.get("use_data_reuploading", USE_DATA_REUPLOADING))
            H_train = qrc.run_sequence(X_train, warmup=w)
            H_val = qrc.run_sequence(X_val, warmup=w)
            H_test = qrc.run_sequence(X_test, warmup=w)
            H_tr, y_tr = _align_after_warmup(H_train, y_train, w)
            H_vl, y_vl = _align_after_warmup(H_val, y_val, w)
            H_ts, y_ts = _align_after_warmup(H_test, y_test, w)

            X_warm = None
            if use_warm and X_train_warm is not None:
                n_align = min(len(H_tr), len(X_train_warm))
                H_tr, y_tr = H_tr[-n_align:], y_tr[-n_align:]
                X_warm = X_train_warm[-n_align:]

            readout = RidgeReadout(target_names=TARGETS, warm_start=use_warm and (X_warm is not None))
            best = readout.fit(H_tr, y_tr, H_vl, y_vl, X_train_warm_start=X_warm)
            readout.save_selection_log(RESULTS / f"h{horizon}")

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

            out_h = RESULTS / f"h{horizon}"
            out_h.mkdir(parents=True, exist_ok=True)
            qrc_result.save(out_h)   # so benchmark_all can reconstruct standalone
            best.save(out_h / f"{mode_label}_readout.pkl")
            qcfg = qrc.get_config()
            qcfg.update(horizon_hours=horizon, readout_strategy=best.strategy,
                       warm_start=use_warm, warm_start_source=WARM_START_SOURCE,
                       shared_pca=USE_SHARED_PCA)
            with open(out_h / f"{mode_label}_config.json", "w") as f:
                json.dump(qcfg, f, indent=2)
            modes[mode_label] = {"strategy": best.strategy, "val_rmse_mean": best.val_rmse_mean}

        per_horizon[horizon] = modes

    return {"horizons": horizons, "J": J_star, "h": h_star, "results": per_horizon}


def _task_benchmark_all(cfg: dict) -> dict:
    """
    Rebuilds the full results_h{N}.csv tables from whatever per-model .pkl
    files already exist under outputs/results/h{N}/ — run train_baselines
    and train_qrc first (or just use `full_run` / `python main.py`).
    """
    from preprocessing.pipeline import WeatherPreprocessor
    from baselines.classical import BaselineResult
    from evaluation.metrics import build_results_table

    horizons = cfg.get("horizons", HORIZONS)
    summaries = {}
    for horizon in horizons:
        ds = WeatherPreprocessor.load(horizon)
        out_h = RESULTS / f"h{horizon}"
        results = {}
        for pkl in out_h.glob("*.pkl"):
            if pkl.name.endswith("_readout.pkl"):
                continue
            try:
                with open(pkl, "rb") as f:
                    import pickle
                    r = pickle.load(f)
                if isinstance(r, BaselineResult) or hasattr(r, "y_pred_val"):
                    results[pkl.stem] = r
            except Exception as e:
                logger.warning(f"Could not load {pkl}: {e}")
        if not results:
            summaries[horizon] = {"status": "skipped", "reason": "no per-model results found"}
            continue
        df = build_results_table(results=results, y_true_val=ds.y_val, y_true_test=ds.y_test,
                                 target_names=TARGETS, horizon_hours=horizon,
                                 out_dir=RESULTS, dataset=ds)
        summaries[horizon] = {"n_models": len(df), "best_model": df.iloc[0]["model"],
                              "best_test_rmse_mean": float(df.iloc[0]["test_rmse_mean"])}
    return {"horizons": horizons, "summaries": summaries}


def _task_hardware_validation(cfg: dict) -> dict:
    from scripts.hardware_validation import run_hardware_validation
    return run_hardware_validation(
        horizon=cfg.get("horizon"), n_steps=cfg.get("n_steps"),
        backend=cfg.get("backend"), mode=cfg.get("mode", "warm_start_qrc"))


def _task_verify_results(cfg: dict) -> dict:
    from verify_results import verify_all
    return verify_all(horizons=cfg.get("horizons", HORIZONS))


TASKS = {
    "setup":               _task_setup,
    "download_data":       _task_download_data,
    "preprocess":          _task_preprocess,
    "eda":                 _task_eda,
    "hamiltonian_sweep":   _task_hamiltonian_sweep,
    "noise_sweep":         _task_noise_sweep,
    "topology_sweep":      _task_topology_sweep,
    "encode_ablation":     _task_encode_ablation,
    "qubit_scaling":       _task_qubit_scaling,
    "shot_ablation":       _task_shot_ablation,
    "train_baselines":     _task_train_baselines,
    "train_qrc":           _task_train_qrc,
    "benchmark_all":       _task_benchmark_all,
    "hardware_validation": _task_hardware_validation,
    "verify_results":      _task_verify_results,
}

FULL_RUN_ORDER = [
    "setup", "download_data", "preprocess", "eda",
    "hamiltonian_sweep", "noise_sweep", "topology_sweep", "encode_ablation",
    "train_baselines", "qubit_scaling", "shot_ablation",
    "train_qrc", "benchmark_all", "verify_results",
]


def run_task(task: str, cfg: dict) -> dict:
    if task == "full_run":
        return _run_full(cfg)
    if task not in TASKS:
        return {"task": task, "status": "failed",
               "error": f"Unknown task {task!r}. Known tasks: {sorted(TASKS)} + full_run",
               "timestamp": _now()}

    t0 = time.time()
    try:
        payload = TASKS[task](cfg)
        result = {"task": task, "status": "complete", "timestamp": _now(),
                 "wall_clock_s": round(time.time() - t0, 2), **payload}
    except Exception as e:
        result = {"task": task, "status": "failed", "error": str(e),
                 "traceback": traceback.format_exc(),
                 "timestamp": _now(), "wall_clock_s": round(time.time() - t0, 2)}
    _write_task_result(task, result)
    return result


def _run_full(cfg: dict) -> dict:
    t0 = time.time()
    steps = {}
    for task in FULL_RUN_ORDER:
        logger.info(f"[full_run] -> {task}")
        result = run_task(task, cfg)
        steps[task] = {"status": result["status"]}
        if result["status"] == "failed":
            logger.error(f"[full_run] {task} FAILED: {result.get('error')}")
            summary = {"task": "full_run", "status": "failed", "failed_at": task,
                      "error": result.get("error"), "steps": steps,
                      "timestamp": _now(), "wall_clock_s": round(time.time() - t0, 2)}
            _write_task_result("full_run", summary)
            return summary
    summary = {"task": "full_run", "status": "complete", "steps": steps,
              "timestamp": _now(), "wall_clock_s": round(time.time() - t0, 2)}
    _write_task_result("full_run", summary)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="ResQue QRC agentic task runner")
    p.add_argument("--task", required=True,
                   choices=sorted(TASKS) + ["full_run"])
    p.add_argument("--config", default="{}",
                   help="JSON string of task-specific overrides, e.g. "
                        "'{\"n_qubits\": 9, \"max_steps\": 300}'")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        cfg = json.loads(args.config)
    except json.JSONDecodeError as e:
        print(f"AGENT_RESULT: {json.dumps({'task': args.task, 'status': 'failed', 'error': f'Invalid --config JSON: {e}'})}")
        sys.exit(1)

    result = run_task(args.task, cfg)
    print("AGENT_RESULT: " + json.dumps(result, default=str))
    sys.exit(0 if result.get("status") == "complete" else 1)
