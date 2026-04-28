"""
File logging for the FPV Lap Counter app.

Call `setup()` once at the top of app.py.
Logs go to <project_root>/app.log only — stdout/stderr are NOT
redirected so Streamlit doesn't show log lines as UI notifications.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_LOG_FILE = Path(__file__).resolve().parent.parent / "app.log"
_initialized = False


def setup() -> None:
    """Initialize file logging. Safe to call multiple times (idempotent)."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(_LOG_FILE, encoding="utf-8", mode="a"),
        ],
        force=True,
    )

    logging.captureWarnings(True)
    logging.getLogger(__name__).info("Logging initialised → %s", _LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper: get a named logger (call after setup())."""
    return logging.getLogger(name)
