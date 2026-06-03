"""
utils/logger.py
===============
Centralised logging setup for the entire application.

Purpose:
    Every module calls `get_logger(__name__)` to get a consistently
    formatted logger instead of ad-hoc print statements.

Workflow:
    Module → get_logger(__name__) → writes to stdout + rotating file

Best Practices:
    - One call to setup_logging() at app startup (app.py)
    - Child loggers inherit the root handler — no duplication
    - RotatingFileHandler prevents unbounded log growth in production
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_INITIALIZED = False


def setup_logging(level: str = "INFO", log_file: str = "data/logs/app.log") -> None:
    """
    Configure the root logger.
    Call once from app.py before importing other modules.

    Args:
        level:    Logging level string ("DEBUG", "INFO", "WARNING", …).
        log_file: Path to the rotating log file.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler (5 MB × 3 backups)
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger. Call setup_logging() first."""
    return logging.getLogger(name)
