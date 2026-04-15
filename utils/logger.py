"""
File + console logging for the FPV Lap Counter app.

Call `setup()` once at the top of app.py.
The log file is written to <project_root>/app.log.
stdout and stderr are tee'd so every line also lands in the file.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_LOG_FILE = Path(__file__).resolve().parent.parent / "app.log"
_initialized = False


class _TeeStream:
    """Writes to multiple streams simultaneously (file + original stderr/stdout)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False

    # Make it behave like a proper text stream
    @property
    def encoding(self) -> str:
        for s in self._streams:
            enc = getattr(s, "encoding", None)
            if enc:
                return enc
        return "utf-8"

    @property
    def errors(self) -> str:
        return "replace"


def setup() -> None:
    """Initialize file logging. Safe to call multiple times (idempotent)."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Open log file (append mode, line-buffered)
    log_fh = open(_LOG_FILE, "a", encoding="utf-8", buffering=1)

    # Python logging → file
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(_LOG_FILE, encoding="utf-8", mode="a"),
            logging.StreamHandler(sys.__stderr__),
        ],
        force=True,
    )

    # Redirect Python warnings through logging
    logging.captureWarnings(True)

    # Tee stdout and stderr to the same log file
    sys.stdout = _TeeStream(sys.__stdout__, log_fh)
    sys.stderr = _TeeStream(sys.__stderr__, log_fh)

    logging.getLogger(__name__).info("Logging initialised → %s", _LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper: get a named logger (call after setup())."""
    return logging.getLogger(name)
