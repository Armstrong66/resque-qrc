"""
data/downloader.py — Download NOAA ISD global-hourly CSVs for configured station.

Direct URL pattern (no API key required):
  https://www.ncei.noaa.gov/data/global-hourly/access/{YEAR}/{STATION_ID}.csv

Run standalone:
  python data/downloader.py
"""

import sys
import time
import requests
from pathlib import Path

# Allow running from project root or this subdirectory
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_RAW, STATION_ID, STATION_NAME, YEARS, NOAA_URL
from utils import get_logger

logger = get_logger(__name__)


def download_station_year(station_id: str, year: int, out_dir: Path,
                          retries: int = 3, timeout: int = 60) -> Path:
    """
    Download one year of ISD data for a station. Returns local path.
    Skips download if file already exists and is non-empty.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / f"{station_id}_{year}.csv"

    if local_path.exists() and local_path.stat().st_size > 1000:
        logger.info(f"Cache hit: {local_path.name} ({local_path.stat().st_size // 1024} KB)")
        return local_path

    url = NOAA_URL.format(year=year, station=station_id)
    logger.info(f"Downloading {year} → {url}")

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            logger.info(f"  Saved: {local_path.name} ({len(resp.content) // 1024} KB)")
            return local_path
        except requests.HTTPError as e:
            logger.warning(f"  HTTP {e.response.status_code} on attempt {attempt}/{retries}")
            if e.response.status_code == 404:
                logger.error(f"  Station {station_id} not found for {year}. Skipping.")
                return None
        except requests.RequestException as e:
            logger.warning(f"  Network error attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)   # Exponential backoff

    logger.error(f"Failed to download {station_id} {year} after {retries} attempts.")
    return None


def download_all(station_id: str = STATION_ID,
                 years: list = None,
                 out_dir: Path = None) -> list[Path]:
    """
    Download all configured years for a station.
    Returns list of successfully downloaded local paths.
    """
    years   = years   or YEARS
    out_dir = out_dir or (DATA_RAW / STATION_NAME)

    logger.info(f"=== Downloading {len(years)} years for station {station_id} ===")
    paths = []
    for year in years:
        p = download_station_year(station_id, year, out_dir)
        if p is not None:
            paths.append(p)
        time.sleep(0.5)   # Be polite to NOAA servers

    logger.info(f"Download complete: {len(paths)}/{len(years)} files acquired.")
    return paths


if __name__ == "__main__":
    download_all()