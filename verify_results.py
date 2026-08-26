"""
verify_results.py — Reproducibility checker for the ResQue QRC pipeline.

Per docs/AGENTIC_DESIGN_GUIDE.md §2.4, this is what a judge (or CI) runs
after re-executing the pipeline to confirm the run actually produced what it
claims to have produced:
  1. All expected output files exist for the horizons that were run.
  2. No NaN values in any results table.
  3. (If a reference run has been saved) key metrics match within tolerance.

There is no stored reference run in this repository yet — establish one
ONCE, after a trusted full run, via:
    python verify_results.py --save_reference
which snapshots outputs/results/results_h*.csv into
outputs/results/reference_run.json. Future runs are then compared against
it automatically. Until that snapshot exists, verify_results.py still
performs checks (1) and (2) in full; (3) is reported as "not available"
rather than silently skipped, so its absence is visible in the output.

Usage:
    python verify_results.py                      # verify against current outputs/
    python verify_results.py --horizons 6 24
    python verify_results.py --save_reference      # snapshot current results as reference
    python verify_results.py --tolerance 0.10      # 10% relative tolerance instead of 5%
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS, HORIZONS
from utils import get_logger

logger = get_logger("verify_results")

DEFAULT_TOLERANCE = 0.05   # 5% relative tolerance on test_rmse_mean vs reference
REFERENCE_PATH = RESULTS / "reference_run.json"


def _expected_files(horizons: list) -> dict:
    """Maps a human label to the Path expected to exist. Sweep-level files
    are expected once (not per horizon); per-horizon files repeat per h."""
    files = {
        "current_run_manifest.json": RESULTS / "current_run_manifest.json",
        "noise_sweep.csv":       RESULTS / "noise_sweep.csv",
        "best_noise.json":       RESULTS / "best_noise.json",
        "qubit_scaling.csv":     RESULTS / "qubit_scaling.csv",
        "shot_ablation.csv":     RESULTS / "shot_ablation.csv",
    }
    for h in horizons:
        out_h = RESULTS / f"h{h}"
        files[f"best_qrc_architecture_h{h}.json"] = RESULTS / f"best_qrc_architecture_h{h}.json"
        files[f"encoding_hamiltonian_comparison_h{h}.csv"] = (
            RESULTS / f"encoding_hamiltonian_comparison_h{h}.csv"
        )
        files[f"topology_comparison_h{h}.csv"] = RESULTS / f"topology_comparison_h{h}.csv"
        files[f"topology_comparison_h{h}_summary.json"] = (
            RESULTS / f"topology_comparison_h{h}_summary.json"
        )
        files[f"results_h{h}.csv"] = RESULTS / f"results_h{h}.csv"
        files[f"h{h}/baseline_status.json"] = out_h / "baseline_status.json"
        files[f"h{h}/warm_start_source_ablation.json"] = (
            out_h / "warm_start_source_ablation.json"
        )
        for mode in ("cold_start_qrc", "warm_start_qrc"):
            files[f"h{h}/{mode}_config.json"] = out_h / f"{mode}_config.json"
            files[f"h{h}/{mode}_readout.pkl"] = out_h / f"{mode}_readout.pkl"
            files[f"h{h}/{mode}_readout_selection.json"] = (
                out_h / f"{mode}_readout_selection.json"
            )
    return files


def check_files_exist(horizons: list) -> dict:
    expected = _expected_files(horizons)
    present = {label: path.exists() for label, path in expected.items()}
    missing = [label for label, ok in present.items() if not ok]
    return {"present": present, "missing": missing,
           "n_expected": len(expected), "n_present": sum(present.values())}


def check_legacy_artifacts() -> list[str]:
    """List old protocol files that are deliberately non-authoritative."""
    legacy = [
        RESULTS / "hamiltonian_sweep.csv", RESULTS / "best_hamiltonian.json",
        RESULTS / "best_topology.json",
    ]
    legacy.extend((RESULTS / f"h{h}" / "readout_selection.json") for h in HORIZONS)
    return [str(path.relative_to(RESULTS)) for path in legacy if path.exists()]


def check_no_nans(horizons: list) -> dict:
    import pandas as pd
    checks = {}
    for h in horizons:
        path = RESULTS / f"results_h{h}.csv"
        if not path.exists():
            checks[f"results_h{h}.csv"] = {"ok": False, "reason": "file missing"}
            continue
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include="number")
        nan_count = int(numeric.isna().sum().sum())
        nan_cols = numeric.columns[numeric.isna().any()].tolist()
        checks[f"results_h{h}.csv"] = {"ok": nan_count == 0, "nan_count": nan_count,
                                       "nan_columns": nan_cols}
    return checks


def _load_reference_metrics(horizons: list) -> dict:
    """{horizon: {model: test_rmse_mean}} from the current results tables."""
    import pandas as pd
    metrics = {}
    for h in horizons:
        path = RESULTS / f"results_h{h}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        metrics[str(h)] = dict(zip(df["model"], df["test_rmse_mean"]))
    return metrics


def save_reference(horizons: list) -> dict:
    metrics = _load_reference_metrics(horizons)
    ref = {"horizons": horizons, "metrics": metrics,
          "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_PATH, "w") as f:
        json.dump(ref, f, indent=2)
    logger.info(f"Reference run saved -> {REFERENCE_PATH}")
    return ref


def compare_to_reference(horizons: list, tolerance: float = DEFAULT_TOLERANCE) -> dict:
    if not REFERENCE_PATH.exists():
        return {"available": False,
               "note": f"No reference run saved yet — run with --save_reference "
                       f"after a trusted full run to enable this check. "
                       f"({REFERENCE_PATH} does not exist.)"}
    ref = json.load(open(REFERENCE_PATH))
    current = _load_reference_metrics(horizons)
    mismatches = []
    checked = 0
    for h in horizons:
        ref_h = ref.get("metrics", {}).get(str(h), {})
        cur_h = current.get(str(h), {})
        for model, ref_val in ref_h.items():
            if model not in cur_h:
                mismatches.append({"horizon": h, "model": model,
                                   "issue": "present in reference, missing now"})
                continue
            cur_val = cur_h[model]
            checked += 1
            if ref_val == 0:
                continue
            rel_diff = abs(cur_val - ref_val) / abs(ref_val)
            if rel_diff > tolerance:
                mismatches.append({"horizon": h, "model": model,
                                   "reference_test_rmse_mean": ref_val,
                                   "current_test_rmse_mean": cur_val,
                                   "relative_diff": round(rel_diff, 4)})
    return {"available": True, "reference_saved_at": ref.get("saved_at"),
           "tolerance": tolerance, "n_metrics_checked": checked,
           "n_mismatches": len(mismatches), "mismatches": mismatches}


def verify_all(horizons: list = None, tolerance: float = DEFAULT_TOLERANCE) -> dict:
    horizons = horizons or HORIZONS
    files = check_files_exist(horizons)
    nans = check_no_nans(horizons)
    reference = compare_to_reference(horizons, tolerance)

    nan_ok = all(v.get("ok", False) for v in nans.values())
    reference_ok = (not reference["available"]) or reference["n_mismatches"] == 0
    status = "pass" if (not files["missing"] and nan_ok and reference_ok) else "fail"

    result = {
        "task": "verify_results", "status": status, "horizons": horizons,
        "files": files, "nan_checks": nans, "reference_comparison": reference,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "legacy_artifacts_not_authoritative": check_legacy_artifacts(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "verify_results_report.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Verify ResQue QRC pipeline outputs")
    p.add_argument("--horizons", type=int, nargs="+", default=None)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    p.add_argument("--save_reference", action="store_true",
                   help="Snapshot current results_h*.csv as the reference run "
                        "for future comparisons, instead of verifying.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    horizons = args.horizons or HORIZONS
    if args.save_reference:
        ref = save_reference(horizons)
        print("AGENT_RESULT: " + json.dumps({"task": "save_reference", "status": "complete", **ref}))
        sys.exit(0)

    result = verify_all(horizons=horizons, tolerance=args.tolerance)
    print("AGENT_RESULT: " + json.dumps(result, default=str))
    sys.exit(0 if result["status"] == "pass" else 1)
