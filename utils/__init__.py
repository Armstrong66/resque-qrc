"""
utils/__init__.py — Structured logging for ResQue QRC.
Use this when running on linux system


Design:
  - Every module calls get_logger(__name__) — no print() calls elsewhere.
  - All modules in a single run share ONE timestamped log file (set at
    first call, reused by all subsequent get_logger() calls in that process).
  - stdout is force-flushed after every record — safe for nohup + tail -f.
  - No changes needed in any other module; this is the only file to edit.

nohup usage:
  bash run.sh          <- recommended (handles nohup + log path announcement)
  -- or manually --
  PYTHONUNBUFFERED=1 nohup python main.py > outputs/logs/nohup_out.txt 2>&1 &

Tail the live log:
  tail -f outputs/logs/run_YYYYMMDD_HHMMSS.log
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# ── Single shared log file for the entire process ─────────────────────────────
# Set once at first import; all subsequent get_logger() calls reuse this path.
_LOG_FILE = None
_RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record — essential for nohup tailing."""
    def emit(self, record):
        super().emit(record)
        self.flush()


class _FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every record — essential for nohup stdout."""
    def emit(self, record):
        super().emit(record)
        self.flush()


def _get_log_file(log_dir):
    global _LOG_FILE
    if _LOG_FILE is None:
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_FILE = log_dir / f"run_{_RUN_STAMP}.log"
        _LOG_FILE.touch()
        # Announce immediately so you know where to tail
        print(f"\n[ResQue] Log file : {_LOG_FILE.resolve()}", flush=True)
        print(f"[ResQue] Monitor  : tail -f {_LOG_FILE.resolve()}\n", flush=True)
    return _LOG_FILE


def get_logger(name, log_dir=None):
    """
    Return a named logger writing to stdout + shared run log file.
    Call once per module: logger = get_logger(__name__)
    All modules in one run share the same log file automatically.
    Zero changes required in any other module.
    """
    from config import LOGS
    log_dir = log_dir or LOGS
    log_file = _get_log_file(log_dir)

    logger = logging.getLogger(name)
    if logger.handlers:      # Already configured — return immediately
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-32s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler (INFO+) — flushes after every line for nohup safety
    ch = _FlushingStreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG+) — same flush-after-emit behaviour
    fh = _FlushingFileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Prevent propagation to root logger (avoids duplicate lines)
    logger.propagate = False

    return logger
