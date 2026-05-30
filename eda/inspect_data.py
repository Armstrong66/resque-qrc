"""
eda/inspect_data.py — Data quality audit for ResQue-QRC pipeline.

Run before any training to confirm data is clean at every stage.
Produces a printed report + saves eda_report.json to outputs/results/.

Usage:
    python eda/inspect_data.py                    # full audit
    python eda/inspect_data.py --stage raw        # raw CSVs only
    python eda/inspect_data.py --stage processed  # parquet only
    python eda/inspect_data.py --stage datasets   # windowed pkl files
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (DATA_RAW, DATA_PROC, RESULTS, STATION_NAME,
                    STATION_ID, YEARS, TARGETS, RESAMPLE_FREQ,
                    WINDOW_SIZE, HORIZONS)
from utils import get_logger

logger = get_logger(__name__)

SEP = "=" * 70


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(n, total):
    return f"{100 * n / total:.1f}%" if total > 0 else "N/A"


def _nan_summary(df, label=""):
    print(f"\n  NaN rates{' — ' + label if label else ''}:")
    for col in df.columns:
        n_nan = df[col].isna().sum()
        print(f"    {col:<20} {n_nan:>6} / {len(df)} "
              f"({_pct(n_nan, len(df))})")


# ── Stage 1: Raw CSVs ─────────────────────────────────────────────────────────

def inspect_raw(raw_dir: Path) -> dict:
    print(f"\n{SEP}")
    print("STAGE 1 — Raw NOAA ISD CSVs")
    print(SEP)

    raw_files = sorted(raw_dir.glob(f"{STATION_ID}_*.csv"))
    if not raw_files:
        print(f"  ✗ No raw files found in {raw_dir}")
        print(f"    Run: python data/downloader.py")
        return {}

    print(f"  Station : {STATION_NAME} ({STATION_ID})")
    print(f"  Files   : {len(raw_files)} found ({[f.name for f in raw_files]})")

    results = {}
    total_rows = 0

    for f in raw_files:
        year = f.stem.split("_")[-1]
        try:
            df = pd.read_csv(f, low_memory=False)
            n = len(df)
            total_rows += n

            # Check critical ISD columns exist
            present = {col: col in df.columns
                       for col in ["DATE", "TMP", "DEW", "SLP", "WND"]}
            missing_cols = [c for c, v in present.items() if not v]

            # Rough missing rate on TMP (sentinel = +/-9999)
            tmp_missing = 0
            if "TMP" in df.columns:
                def _is_missing_tmp(v):
                    try:
                        return abs(int(str(v).split(",")[0])) >= 9999
                    except Exception:
                        return True
                tmp_missing = df["TMP"].apply(_is_missing_tmp).sum()

            slp_missing = 0
            if "SLP" in df.columns:
                def _is_missing_slp(v):
                    try:
                        return int(str(v).split(",")[0]) >= 99999
                    except Exception:
                        return True
                slp_missing = df["SLP"].apply(_is_missing_slp).sum()

            status = "✓" if not missing_cols else f"✗ missing cols: {missing_cols}"
            print(f"\n  {year}: {n} rows  {status}")
            print(f"    TMP sentinel-missing : {_pct(tmp_missing, n)}")
            print(f"    SLP sentinel-missing : {_pct(slp_missing, n)}")

            results[year] = {
                "rows": n,
                "columns_present": present,
                "tmp_missing_pct": round(100 * tmp_missing / n, 1) if n else 0,
                "slp_missing_pct": round(100 * slp_missing / n, 1) if n else 0,
            }
        except Exception as e:
            print(f"  {year}: ✗ Failed to read — {e}")
            results[year] = {"error": str(e)}

    print(f"\n  Total raw rows across all years: {total_rows:,}")
    return results


# ── Stage 2: Processed Parquet ────────────────────────────────────────────────

def inspect_processed(proc_dir: Path) -> dict:
    print(f"\n{SEP}")
    print("STAGE 2 — Processed Parquet (parsed + resampled)")
    print(SEP)

    from data.parser import processed_parquet_path
    parquet_path = processed_parquet_path(proc_dir, RESAMPLE_FREQ)
    if not parquet_path.exists():
        print(f"  ✗ Parquet not found: {parquet_path}")
        print(f"    Run pipeline to generate, or run: python data/parser.py")
        return {}

    df = pd.read_parquet(parquet_path)
    print(f"\n  Path        : {parquet_path}")
    print(f"  Shape       : {df.shape}  ({len(df)} timesteps × {len(df.columns)} cols)")
    print(f"  Date range  : {df.index.min()} → {df.index.max()}")
    print(f"  Frequency   : {RESAMPLE_FREQ} (expected {len(df)} steps)")

    _nan_summary(df, "after resample + interpolation")

    # Check for any NaN in target columns — these will poison training
    target_nan = {col: int(df[col].isna().sum()) for col in TARGETS if col in df.columns}
    any_target_nan = any(v > 0 for v in target_nan.values())

    print(f"\n  Target NaN check:")
    for col, n_nan in target_nan.items():
        flag = "✗ WILL POISON TRAINING" if n_nan > 0 else "✓"
        print(f"    {col:<20} {n_nan:>6} NaN  {flag}")

    if any_target_nan:
        print("\n  *** ACTION REQUIRED ***")
        print("  Delete this parquet and re-run with updated parser.py:")
        print(f"    rm {parquet_path}")
        print("  Rebuild with: python main.py --force_rebuild_data")
    else:
        print("\n  ✓ No NaN in targets — parquet is clean")

    # Descriptive stats
    print(f"\n  Descriptive statistics (physical units before z-score):")
    try:
        print(df[TARGETS].describe().to_string())
    except Exception as e:
        print(f"    Could not compute stats: {e}")

    # Check temporal continuity
    if hasattr(df.index, 'freq') or len(df) > 1:
        expected_steps = pd.date_range(df.index.min(), df.index.max(),
                                        freq=RESAMPLE_FREQ)
        n_gaps = len(expected_steps) - len(df)
        if n_gaps > 0:
            print(f"\n  ✗ {n_gaps} missing timesteps in resampled grid "
                  f"(gaps after interpolation limit)")
        else:
            print(f"\n  ✓ Temporal grid complete (no missing steps)")

    return {
        "shape": list(df.shape),
        "date_range": [str(df.index.min()), str(df.index.max())],
        "target_nan_counts": target_nan,
        "any_target_nan": any_target_nan,
        "stats": df[TARGETS].describe().to_dict() if not any_target_nan else {},
    }


# ── Stage 3: Windowed datasets ────────────────────────────────────────────────

def inspect_datasets(proc_dir: Path) -> dict:
    print(f"\n{SEP}")
    print("STAGE 3 — Windowed Datasets (pkl files)")
    print(SEP)

    results = {}
    for h in HORIZONS:
        from config import PREPROCESS_VERSION
        pkl_path = proc_dir / f"dataset_h{h}_{PREPROCESS_VERSION}.pkl"
        if not pkl_path.exists():
            pkl_path = proc_dir / f"dataset_h{h}.pkl"
        print(f"\n  Horizon {h}h → {pkl_path.name}")

        if not pkl_path.exists():
            print(f"    ✗ Not found. Run preprocessing pipeline first.")
            continue

        try:
            with open(pkl_path, "rb") as f:
                ds = pickle.load(f)

            splits = {
                "train": (ds.X_train, ds.y_train),
                "val":   (ds.X_val,   ds.y_val),
                "test":  (ds.X_test,  ds.y_test),
            }

            for split_name, (X, y) in splits.items():
                x_nan = np.isnan(X).sum()
                y_nan = np.isnan(y).sum()
                x_flag = "✗ NaN in X!" if x_nan > 0 else "✓"
                y_flag = "✗ NaN in y!" if y_nan > 0 else "✓"
                print(f"    {split_name:<6}: X{X.shape} NaN={x_nan} {x_flag} | "
                      f"y{y.shape} NaN={y_nan} {y_flag}")

            # Check input dimension matches expectation
            expected_dim = WINDOW_SIZE * len(TARGETS)
            actual_dim = ds.X_train.shape[1]
            dim_flag = "✓" if actual_dim == expected_dim else \
                       f"✗ expected {expected_dim}"
            print(f"    input_dim = {actual_dim} {dim_flag}")
            print(f"    n_targets = {ds.y_train.shape[1]} "
                  f"({'✓' if ds.y_train.shape[1] == len(TARGETS) else '✗'})")
            print(f"    scaler_mean : {ds.scaler_mean}")
            print(f"    scaler_std  : {ds.scaler_std}")

            # Warn if scaler stats are NaN (means train data was NaN)
            if ds.scaler_mean is not None and np.any(np.isnan(ds.scaler_mean)):
                print(f"    ✗ scaler_mean contains NaN — "
                      f"training split had NaN values when stats were computed!")

            results[f"h{h}"] = {
                "train_shape": list(ds.X_train.shape),
                "val_shape":   list(ds.X_val.shape),
                "test_shape":  list(ds.X_test.shape),
                "X_train_nan": int(np.isnan(ds.X_train).sum()),
                "y_train_nan": int(np.isnan(ds.y_train).sum()),
                "scaler_mean_nan": bool(np.any(np.isnan(ds.scaler_mean)))
                                   if ds.scaler_mean is not None else True,
            }
        except Exception as e:
            print(f"    ✗ Failed to load: {e}")
            results[f"h{h}"] = {"error": str(e)}

    return results


# ── Stage 4: Encoding sanity check ───────────────────────────────────────────

def inspect_encoding_bottleneck() -> dict:
    """
    Check the input information bottleneck: 80-dim window → n_qubits angles.
    This is the P0 scientific fairness issue flagged in PROJECT_CRITIQUE.md.
    """
    print(f"\n{SEP}")
    print("STAGE 4 — Encoding bottleneck analysis (QRC vs classical)")
    print(SEP)

    from config import USE_SHARED_PCA, QUBIT_PRIMARY, PCA_COMPONENTS
    window_flat = WINDOW_SIZE * len(TARGETS)
    pca_dim = PCA_COMPONENTS or QUBIT_PRIMARY

    print(f"\n  Window size         : {WINDOW_SIZE} steps x {len(TARGETS)} vars "
          f"= {window_flat} features")
    print(f"  USE_SHARED_PCA      : {USE_SHARED_PCA}")
    if USE_SHARED_PCA:
        print(f"  All reservoir/RNN models: PCA -> {pca_dim}d (train-only, fair)")
        print(f"\n  OK Shared PCA enabled — no cyclic np.resize bottleneck.")
        bottleneck = False
    else:
        print(f"  QRC only: np.resize -> {pca_dim} angles")
        bottleneck = pca_dim < window_flat

    return {
        "window_flat_dim": window_flat,
        "pca_dim": pca_dim,
        "use_shared_pca": USE_SHARED_PCA,
        "bottleneck_active": bottleneck,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ResQue-QRC data quality audit")
    parser.add_argument("--stage", default="all",
                        choices=["all", "raw", "processed", "datasets", "encoding"])
    args = parser.parse_args()

    raw_dir  = DATA_RAW  / STATION_NAME
    proc_dir = DATA_PROC / STATION_NAME

    print(f"\n{'#'*70}")
    print(f"  ResQue-QRC — Data Quality Audit")
    print(f"  Station: {STATION_NAME} ({STATION_ID})")
    print(f"  Years  : {YEARS}")
    print(f"{'#'*70}")

    report = {"station": STATION_NAME, "station_id": STATION_ID}

    if args.stage in ("all", "raw"):
        report["raw"] = inspect_raw(raw_dir)

    if args.stage in ("all", "processed"):
        report["processed"] = inspect_processed(proc_dir)

    if args.stage in ("all", "datasets"):
        report["datasets"] = inspect_datasets(proc_dir)

    if args.stage in ("all", "encoding"):
        report["encoding"] = inspect_encoding_bottleneck()

    # Save report
    RESULTS.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS / "eda_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n{SEP}")
    print(f"EDA report saved → {report_path}")
    print(SEP)


if __name__ == "__main__":
    main()