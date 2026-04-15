"""
Lap time calculation from detected gate passes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Pass:
    """A single detected gate pass."""
    frame: int
    time_sec: float               # absolute video timestamp (seconds)
    osd_time: float | None        # timer value read from OSD (seconds), or None
    similarity: float             # cosine similarity score at this pass
    verified: bool | None = None  # True/False if manually verified; None = unknown


@dataclass
class Lap:
    """A lap defined by two consecutive gate passes."""
    number: int
    duration_sec: float
    start_pass: Pass
    end_pass: Pass

    @property
    def duration_str(self) -> str:
        """Format lap duration as 'MM:SS.ss'."""
        total = self.duration_sec
        if not math.isfinite(total) or total < 0:
            return "--:--.--"
        minutes = int(total // 60)
        seconds = total - minutes * 60
        # seconds can be up to 59.999…
        whole_sec = int(seconds)
        centiseconds = int(round((seconds - whole_sec) * 100))
        if centiseconds >= 100:
            centiseconds = 99
        return f"{minutes:02d}:{whole_sec:02d}.{centiseconds:02d}"


@dataclass
class LapResult:
    """Aggregated analysis result for a full flight."""
    passes: list[Pass]
    laps: list[Lap]
    best_lap: Lap | None
    best_n: dict[int, list[Lap]]   # n -> best consecutive n laps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass_time(p: Pass) -> float:
    """Return the best available absolute timestamp for a pass."""
    # Prefer OSD time because it is measured by the drone's own clock
    if p.osd_time is not None:
        return p.osd_time
    return p.time_sec


def _lap_duration(start: Pass, end: Pass) -> float:
    """Compute lap duration using OSD times when available, else video times."""
    t_start = _pass_time(start)
    t_end = _pass_time(end)
    return t_end - t_start


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_laps(passes: list[Pass]) -> list[Lap]:
    """
    Compute sequential laps from a list of gate passes.

    Only passes where ``verified`` is not ``False`` are used.
    Duration is computed using the OSD timestamp when available for both
    endpoints, otherwise the video timestamp is used.

    Parameters
    ----------
    passes : All detected (and optionally verified) gate passes.

    Returns
    -------
    List of Lap objects in chronological order.
    """
    valid = [p for p in passes if p.verified is not False]
    # Sort by the best available time to handle out-of-order input
    valid.sort(key=_pass_time)

    laps: list[Lap] = []
    for i in range(len(valid) - 1):
        start = valid[i]
        end = valid[i + 1]
        duration = _lap_duration(start, end)
        if duration <= 0:
            # Skip degenerate intervals (e.g. duplicate detections)
            continue
        laps.append(
            Lap(
                number=len(laps) + 1,
                duration_sec=duration,
                start_pass=start,
                end_pass=end,
            )
        )
    return laps


def best_consecutive(laps: list[Lap], n: int) -> list[Lap] | None:
    """
    Find the *n* consecutive laps with the lowest total duration.

    Parameters
    ----------
    laps : Ordered list of laps.
    n    : Number of consecutive laps to consider.

    Returns
    -------
    List of *n* consecutive Lap objects with the smallest combined time,
    or None if ``len(laps) < n``.
    """
    if n < 1 or len(laps) < n:
        return None

    best_start = 0
    best_total = sum(l.duration_sec for l in laps[:n])
    running = best_total

    for i in range(1, len(laps) - n + 1):
        running = running - laps[i - 1].duration_sec + laps[i + n - 1].duration_sec
        if running < best_total:
            best_total = running
            best_start = i

    return laps[best_start : best_start + n]


def analyze(
    passes: list[Pass],
    best_ns: list[int] | None = None,
) -> LapResult:
    """
    Full analysis: compute laps, find the best single lap and the best
    windows of *n* consecutive laps.

    Parameters
    ----------
    passes  : Detected gate passes (may be unordered, may have unverified entries).
    best_ns : List of window sizes for ``best_consecutive``. Defaults to [1, 3, 5].

    Returns
    -------
    LapResult with all computed fields populated.
    """
    if best_ns is None:
        best_ns = [1, 3, 5]

    laps = compute_laps(passes)

    best_lap: Lap | None = None
    if laps:
        best_lap = min(laps, key=lambda l: l.duration_sec)

    best_n: dict[int, list[Lap]] = {}
    for n in best_ns:
        result = best_consecutive(laps, n)
        if result is not None:
            best_n[n] = result

    return LapResult(
        passes=list(passes),
        laps=laps,
        best_lap=best_lap,
        best_n=best_n,
    )
