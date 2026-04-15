"""
Сохранение и загрузка результатов анализа видео в .fpv.json
"""
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass
class PassData:
    frame: int
    time_sec: float
    osd_time: float | None
    similarity: float
    verified: bool | None = None  # None=not reviewed, True=ok, False=fake

@dataclass
class LapData:
    number: int
    duration_sec: float
    start_sec: float
    osd_start: float | None

@dataclass
class ProjectData:
    video: str                    # filename only
    model: str
    analyzed_at: str
    params: dict                  # threshold, min_lap_sec, sample_every, prominence
    osd_region: dict | None       # {x,y,w,h} fractions or None (use default)
    passes: list[PassData] = field(default_factory=list)
    laps: list[LapData] = field(default_factory=list)
    best_lap_idx: int | None = None
    best_3_indices: list[int] | None = None
    references_b64: list[str] = field(default_factory=list)  # base64 JPEG thumbnails stored here

def json_path(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.parent / (p.stem + ".fpv.json")

def save(data: ProjectData, video_path: str | Path) -> None:
    path = json_path(video_path)
    # convert dataclasses to dicts recursively
    raw = asdict(data)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

def load(video_path: str | Path) -> ProjectData | None:
    path = json_path(video_path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # reconstruct nested dataclasses
        raw["passes"] = [PassData(**p) for p in raw.get("passes", [])]
        raw["laps"] = [LapData(**l) for l in raw.get("laps", [])]
        return ProjectData(**raw)
    except Exception:
        return None

def exists(video_path: str | Path) -> bool:
    return json_path(video_path).exists()
