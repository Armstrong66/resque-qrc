"""
data/parser.py — Parse NOAA ISD global-hourly CSV files into clean DataFrames.

ISD CSV columns we care about (ref: isd-format-document.pdf):
  DATE        — ISO timestamp (UTC)
  TMP         — Air temperature: "TTTT,Q"  value in tenths of °C; 9999 = missing
  DEW         — Dew point:       "TTTT,Q"  tenths of °C
  SLP         — Sea level pressure: "PPPPP,Q"  tenths of hPa; 99999 = missing
  WND         — Wind: "ddd,Q,T,ffff,Q"  speed in tenths of m/s; 9999 = missing
  AA1         — Precipitation (optional, not primary target)

Relative humidity is derived from temperature + dew point via Magnus formula.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (DATA_PROC, STATION_NAME, STATION_ID, RESAMPLE_FREQ,
                    MAX_INTERP_GAP, TARGETS, DATA_CACHE_VERSION)
from utils import get_logger

logger = get_logger(__name__)

# ISD missing-value sentinels
MISSING = {
    "TMP": 9999,
    "DEW": 9999,
    "SLP": 99999,
    "WND_SPEED": 9999,
}


def processed_parquet_path(out_dir: Path = None, resample: str = None) -> Path:
    """Versioned cache path so parser changes do not reuse stale Parquet."""
    out_dir = out_dir or (DATA_PROC / STATION_NAME)
    resample = resample or RESAMPLE_FREQ
    return out_dir / f"weather_{resample}_{STATION_ID}_{DATA_CACHE_VERSION}.parquet"


def _parse_tmp(series: pd.Series) -> pd.Series:
    """Parse TMP field 'TTTT,Q' → float °C. Returns NaN for missing."""
    def _p(v):
        try:
            val = int(str(v).split(",")[0])
            return np.nan if abs(val) >= MISSING["TMP"] else val / 10.0
        except Exception:
            return np.nan
    return series.apply(_p)


def _parse_dew(series: pd.Series) -> pd.Series:
    """Parse DEW field 'TTTT,Q' → float °C."""
    def _p(v):
        try:
            val = int(str(v).split(",")[0])
            return np.nan if abs(val) >= MISSING["DEW"] else val / 10.0
        except Exception:
            return np.nan
    return series.apply(_p)


def _parse_slp(series: pd.Series) -> pd.Series:
    """Parse SLP field 'PPPPP,Q' → float hPa."""
    def _p(v):
        try:
            val = int(str(v).split(",")[0])
            return np.nan if val >= MISSING["SLP"] else val / 10.0
        except Exception:
            return np.nan
    return series.apply(_p)


def _parse_wnd_speed(series: pd.Series) -> pd.Series:
    """Parse WND field 'ddd,Q,T,ffff,Q' → wind speed float m/s."""
    def _p(v):
        try:
            parts = str(v).split(",")
            speed = int(parts[3])
            return np.nan if speed >= MISSING["WND_SPEED"] else speed / 10.0
        except Exception:
            return np.nan
    return series.apply(_p)


def _relative_humidity(T: pd.Series, Td: pd.Series) -> pd.Series:
    """
    Magnus formula: RH = 100 * exp(17.625*Td/(243.04+Td)) / exp(17.625*T/(243.04+T))
    Returns % RH. NaN where either T or Td is NaN.
    """
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp(a * Td / (b + Td)) / np.exp(a * T / (b + T))
    return rh.clip(0, 100)


def parse_isd_csv(filepath: Path) -> pd.DataFrame:
    """
    Parse a single NOAA ISD CSV file.
    Returns DataFrame with columns: timestamp, temperature, humidity,
    pressure, wind_speed — all float, NaN where missing.
    """
    logger.debug(f"Parsing {filepath.name}")

    try:
        raw = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return pd.DataFrame()

    if "DATE" not in raw.columns:
        logger.error(f"No DATE column in {filepath.name}")
        return pd.DataFrame()

    df = pd.DataFrame()
    df["timestamp"] = pd.to_datetime(raw["DATE"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if "TMP" in raw.columns:
        df["temperature"] = _parse_tmp(raw["TMP"])
    else:
        df["temperature"] = np.nan

    if "DEW" in raw.columns:
        dew = _parse_dew(raw["DEW"])
        df["humidity"] = _relative_humidity(df["temperature"], dew)
    else:
        df["humidity"] = np.nan

    if "SLP" in raw.columns:
        df["pressure"] = _parse_slp(raw["SLP"])
    else:
        df["pressure"] = np.nan

    if "WND" in raw.columns:
        df["wind_speed"] = _parse_wnd_speed(raw["WND"])
    else:
        df["wind_speed"] = np.nan

    df = df.set_index("timestamp").sort_index()
    n_raw = len(df)
    df = df[~df.index.duplicated(keep="first")]
    logger.debug(f"  {n_raw} rows → {len(df)} after dedup")

    return df


def _clean_resampled(resampled: pd.DataFrame) -> pd.DataFrame:
    """
    Fill remaining gaps, then drop any row with a NaN in a target column.
    Ensures training arrays are never poisoned by partial missing rows.
    """
    n_before = len(resampled)
    resampled = resampled.interpolate(method="time", limit=MAX_INTERP_GAP)
    resampled = resampled.ffill().bfill()
    resampled = resampled.dropna(subset=TARGETS, how="any")
    n_after = len(resampled)

    for col in TARGETS:
        miss = resampled[col].isna().sum()
        if miss > 0:
            raise ValueError(
                f"Column {col} still has {miss} NaN after cleaning — check parser logic."
            )

    logger.info(
        f"Cleaned grid: {n_before} -> {n_after} steps "
        f"({100 * n_after / max(n_before, 1):.1f}% retained, zero NaN in targets)"
    )
    return resampled


def load_and_merge(raw_paths: list[Path],
                   out_dir: Path = None,
                   resample: str = None,
                   force_rebuild: bool = False) -> pd.DataFrame:
    """
    Parse all yearly CSVs, concatenate, resample to target frequency,
    interpolate short gaps, fill edges, drop partial-NaN rows, save Parquet.
    """
    out_dir = out_dir or (DATA_PROC / STATION_NAME)
    resample = resample or RESAMPLE_FREQ
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_parquet_path(out_dir, resample)

    if out_path.exists() and not force_rebuild:
        logger.info(f"Loading cached processed data: {out_path}")
        df = pd.read_parquet(out_path)
        nan_any = df[TARGETS].isna().any().any()
        if nan_any:
            logger.warning(
                f"Cached Parquet contains NaN targets — rebuilding. "
                f"Delete manually: {out_path}"
            )
        else:
            return df

    frames = []
    for p in sorted(raw_paths):
        df = parse_isd_csv(p)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("No data parsed — check raw files.")

    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]

    resampled = full.resample(resample).mean()
    resampled = _clean_resampled(resampled)

    resampled.to_parquet(out_path)
    logger.info(f"Saved processed data → {out_path}")
    return resampled


if __name__ == "__main__":
    from data.downloader import download_all
    paths = download_all()
    df = load_and_merge(paths, force_rebuild=True)
    print(df.describe())
    print(df.head())
