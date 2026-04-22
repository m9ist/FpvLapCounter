"""
Persistent usage counter for neural network models.
Counts how many videos each model has been used to analyze.
Stored in model_usage.json next to this file.
"""
from __future__ import annotations

import json
from pathlib import Path

_STATS_FILE = Path(__file__).resolve().parent / "model_usage.json"


def load() -> dict[str, int]:
    """Return {model_key: usage_count}. Missing keys default to 0."""
    try:
        return json.loads(_STATS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def increment(model_key: str) -> None:
    """Add 1 to the counter for model_key and persist."""
    stats = load()
    stats[model_key] = stats.get(model_key, 0) + 1
    _STATS_FILE.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
