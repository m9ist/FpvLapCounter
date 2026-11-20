"""
«После клуба»: поиск Флешки HDZero и Выходной папки вечера.
Без Streamlit — чистые функции над файловой системой.
"""
from __future__ import annotations

import os
import re
import string
from datetime import date
from pathlib import Path

INPUT_SUBDIR = r"DCIM\100HDZRO"
VIDEO_EXTS = {".ts", ".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

_REC_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})-\d{6}")


def find_input_dirs() -> list[Path]:
    """Все Входные папки (DCIM\\100HDZRO) на всех дисках.

    Регистр Windows и так не различает. os.path.isdir, а не Path.is_dir:
    глотает любой OSError (пустой картридер, заблокированный диск).
    """
    paths = (f"{letter}:\\{INPUT_SUBDIR}" for letter in string.ascii_uppercase)
    return [Path(p) for p in paths if os.path.isdir(p)]


def output_dir(archive: str, today: date | None = None) -> Path:
    """Выходная папка вечера: <Архив>\\ГГГГ-ММ-ДД."""
    return Path(archive) / (today or date.today()).isoformat()


def recording_dates(files: list[Path]) -> set[date]:
    """Даты записей по именам вида 20261120-113831.ts (нераспознанные пропускаются)."""
    dates = set()
    for f in files:
        m = _REC_DATE_RE.match(f.name)
        if m:
            try:
                dates.add(date(int(m[1]), int(m[2]), int(m[3])))
            except ValueError:
                pass  # цифры похожи на дату, но не дата — просто не учитываем
    return dates
