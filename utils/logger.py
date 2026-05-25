"""
utils/logger.py — Structured logging for ResQue QRC.
Use this logger when running on windows system
Every module imports get_logger(__name__) — no print() calls elsewhere.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: Path = None) -> logging.Logger:
    """
    Return a logger that writes to both stdout and a timestamped file.
    Call once per module: logger = get_logger(__name__)
    """
    from config import LOGS
    log_dir = log_dir or LOGS
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # Avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG and above — catches everything including tracebacks)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"run_{stamp}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger