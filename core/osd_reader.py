"""
OSD timer reader — extracts elapsed time from FPV on-screen display text.
Uses EasyOCR for text recognition with optional GPU acceleration.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region descriptor
# ---------------------------------------------------------------------------

@dataclass
class OSDRegion:
    """
    Rectangular region expressed as fractions of frame dimensions (0..1).
    (x, y) is the top-left corner; (w, h) is width and height.
    """
    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, frame_h: int, frame_w: int) -> tuple[int, int, int, int]:
        """Return (x0, y0, x1, y1) in pixel coordinates (clamped)."""
        x0 = max(0, int(self.x * frame_w))
        y0 = max(0, int(self.y * frame_h))
        x1 = min(frame_w, int((self.x + self.w) * frame_w))
        y1 = min(frame_h, int((self.y + self.h) * frame_h))
        return x0, y0, x1, y1


DEFAULT_REGION = OSDRegion(0.03, 0.07, 0.22, 0.10)


# ---------------------------------------------------------------------------
# Timer text parser
# ---------------------------------------------------------------------------

# Patterns for common FPV OSD timer formats
_PATTERNS: list[re.Pattern] = [
    # MM:SS.s  or  M:SS.s  (with optional fractional seconds)
    re.compile(r"(\d{1,2}):(\d{2})[\.,](\d+)"),
    # MM:SS  (no fraction)
    re.compile(r"(\d{1,2}):(\d{2})"),
    # M SS s  (spaces instead of punctuation, e.g. "0 20 3")
    re.compile(r"(\d{1,2})\s+(\d{2})\s+(\d+)"),
    # SSS.s  (pure seconds with fraction, e.g. "125.3")
    re.compile(r"(\d+)[\.,](\d+)"),
]


def parse_timer(text: str) -> float | None:
    """
    Parse an OSD timer string into seconds.

    Accepts formats such as:
      "00:20.3"  -> 20.3
      "1:25"     -> 85.0
      "1:25.67"  -> 85.67
      "0 20 3"   -> 20.3
      "125.3"    -> 125.3

    Returns None when no recognisable pattern is found.
    """
    text = text.strip()

    # MM:SS.frac
    m = _PATTERNS[0].search(text)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        frac_str = m.group(3)
        frac = int(frac_str) / (10 ** len(frac_str))
        return minutes * 60.0 + seconds + frac

    # MM:SS
    m = _PATTERNS[1].search(text)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        return minutes * 60.0 + float(seconds)

    # M SS s  (space-separated)
    m = _PATTERNS[2].search(text)
    if m:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        frac_str = m.group(3)
        frac = int(frac_str) / (10 ** len(frac_str))
        return minutes * 60.0 + seconds + frac

    # SSS.frac  (no minutes field)
    m = _PATTERNS[3].search(text)
    if m:
        secs = int(m.group(1))
        frac_str = m.group(2)
        frac = int(frac_str) / (10 ** len(frac_str))
        return secs + frac

    return None


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def _preprocess_region(crop: np.ndarray) -> np.ndarray:
    """
    Prepare a BGR crop for OCR:
      1. Convert to grayscale.
      2. Upscale 3× for better OCR accuracy.
      3. Threshold to isolate bright (white) OSD text.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale 3×
    h, w = gray.shape[:2]
    upscaled = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold: keep pixels brighter than local neighbourhood
    # (handles varying brightness across the OSD area)
    binary = cv2.adaptiveThreshold(
        upscaled,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=-20,
    )

    # Convert back to BGR so EasyOCR receives a standard image
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# OSD Reader
# ---------------------------------------------------------------------------

class OSDReader:
    """
    Reads the elapsed-timer value from FPV video frames using EasyOCR.

    Parameters
    ----------
    use_gpu : bool
        Enable CUDA GPU acceleration for EasyOCR (if available).
    """

    def __init__(self, use_gpu: bool = True) -> None:
        self._use_gpu = use_gpu
        self._reader = None   # lazy-loaded easyocr.Reader

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Initialise EasyOCR (downloads model weights on first use)."""
        if self._reader is not None:
            return
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError(
                "easyocr is required for OSD reading. "
                "Install it with: pip install easyocr"
            ) from exc

        logger.info(
            "Loading EasyOCR (gpu=%s) …", self._use_gpu
        )
        self._reader = easyocr.Reader(
            ["en"],
            gpu=self._use_gpu,
            verbose=False,
        )
        logger.info("EasyOCR loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_frame(
        self,
        frame: np.ndarray,
        region: OSDRegion | None = None,
    ) -> float | None:
        """
        Extract the timer value from a single BGR frame.

        Parameters
        ----------
        frame  : BGR numpy array (H, W, 3).
        region : OSD region; uses DEFAULT_REGION when None.

        Returns
        -------
        Elapsed time in seconds, or None if OCR fails / no timer found.
        """
        self._load()
        rgn = region or DEFAULT_REGION

        h, w = frame.shape[:2]
        x0, y0, x1, y1 = rgn.to_pixels(h, w)

        if x1 <= x0 or y1 <= y0:
            logger.debug("OSD region has zero area for frame shape %s", frame.shape)
            return None

        crop = frame[y0:y1, x0:x1]
        processed = _preprocess_region(crop)

        results = self._reader.readtext(processed, detail=0)
        raw_text = " ".join(results)
        logger.debug("OCR raw text: %r", raw_text)

        return parse_timer(raw_text)

    def read_batch(
        self,
        frames: list[np.ndarray],
        region: OSDRegion | None = None,
    ) -> list[float | None]:
        """
        Extract timer values from a list of BGR frames.

        Parameters
        ----------
        frames : List of BGR numpy arrays.
        region : OSD region; uses DEFAULT_REGION when None.

        Returns
        -------
        List of elapsed times in seconds (or None for frames where OCR fails).
        """
        self._load()
        return [self.read_frame(f, region) for f in frames]
