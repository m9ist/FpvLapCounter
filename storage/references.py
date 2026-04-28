"""
Управление референсными кадрами ворот.
Три источника: файл, буфер обмена, текущий кадр видео.
"""
import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass

THUMB_SIZE = (224, 224)  # for display and storage

# Persistent history folder — sits next to the project root
REFS_DIR = Path(__file__).resolve().parent.parent / "refs"


def _imwrite(path: Path, bgr: np.ndarray) -> None:
    """cv2.imwrite that works with non-ASCII paths on Windows."""
    _, buf = cv2.imencode(path.suffix or ".jpg", bgr)
    path.write_bytes(buf.tobytes())


def _imread(path: Path) -> np.ndarray | None:
    """cv2.imread that works with non-ASCII paths on Windows."""
    try:
        arr = np.frombuffer(path.read_bytes(), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_ref_to_history(ref: "RefImage") -> Path:
    """Save a reference image as JPEG into REFS_DIR and return the path."""
    REFS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in ref.name)[:40]
    fpath = REFS_DIR / f"{ts}_{safe}.jpg"
    _imwrite(fpath, ref.bgr)
    return fpath


def load_ref_history() -> list["RefImage"]:
    """Load all JPEG files from REFS_DIR, newest first."""
    if not REFS_DIR.exists():
        return []
    result = []
    for fpath in sorted(REFS_DIR.glob("*.jpg"), reverse=True):
        bgr = _imread(fpath)
        if bgr is not None:
            # Strip timestamp prefix for display name
            stem = fpath.stem
            parts = stem.split("_", 2)
            display = parts[2] if len(parts) == 3 else stem
            result.append(RefImage(name=display, bgr=bgr))
    return result

@dataclass(eq=False)
class RefImage:
    name: str           # display label
    bgr: np.ndarray     # full image for embedding

    def to_b64(self) -> str:
        """JPEG base64 for JSON storage."""
        small = cv2.resize(self.bgr, THUMB_SIZE)
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf).decode()

    @staticmethod
    def from_b64(b64: str, name: str = "saved") -> "RefImage":
        buf = base64.b64decode(b64)
        arr = np.frombuffer(buf, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return RefImage(name=name, bgr=bgr)

    def thumbnail_rgb(self) -> np.ndarray:
        small = cv2.resize(self.bgr, THUMB_SIZE)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

def from_file(uploaded_file) -> RefImage:
    """From Streamlit UploadedFile."""
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return RefImage(name=uploaded_file.name, bgr=bgr)

def from_clipboard() -> RefImage | None:
    """Grab image from Windows clipboard via PIL.ImageGrab."""
    try:
        from PIL import ImageGrab
        img = ImageGrab.grabclipboard()
        if img is None:
            return None
        if not isinstance(img, Image.Image):
            return None
        bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        return RefImage(name="clipboard", bgr=bgr)
    except Exception:
        return None

def from_frame(frame_bgr: np.ndarray, label: str = "frame") -> RefImage:
    """From a video frame (BGR numpy array)."""
    return RefImage(name=label, bgr=frame_bgr.copy())
