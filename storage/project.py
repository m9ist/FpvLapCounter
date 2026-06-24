"""
Сохранение и загрузка результатов анализа видео в .fpv.json
"""
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict, fields

logger = logging.getLogger(__name__)

@dataclass
class PassData:
    frame: int
    time_sec: float
    similarity: float
    verified: bool | None = None  # None=not reviewed, True=ok, False=fake

@dataclass
class LapData:
    number: int
    duration_sec: float
    start_sec: float

@dataclass
class ProjectData:
    video: str                    # filename only
    model: str
    analyzed_at: str
    params: dict                  # threshold, min_lap_sec, sample_every, prominence
    passes: list[PassData] = field(default_factory=list)
    laps: list[LapData] = field(default_factory=list)
    best_lap_idx: int | None = None
    best_3_indices: list[int] | None = None
    references_b64: list[str] = field(default_factory=list)  # base64 JPEG thumbnails stored here
    keep: bool | None = None          # user marked this file to keep in compare tab


def _only_known(cls, d: dict) -> dict:
    """Drop keys that aren't fields of cls — tolerates legacy JSON (e.g. osd_*)."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}

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
        # reconstruct nested dataclasses (legacy osd_* keys are dropped)
        raw["passes"] = [PassData(**_only_known(PassData, p)) for p in raw.get("passes", [])]
        raw["laps"] = [LapData(**_only_known(LapData, l)) for l in raw.get("laps", [])]
        return ProjectData(**_only_known(ProjectData, raw))
    except Exception:
        # Повреждённый/несовместимый JSON: не маскируем под «нет данных» молча —
        # видео повторно проанализируется, поэтому хотя бы фиксируем причину.
        logger.warning("Не удалось прочитать %s — будет переанализ", path, exc_info=True)
        return None

def exists(video_path: str | Path) -> bool:
    return json_path(video_path).exists()
