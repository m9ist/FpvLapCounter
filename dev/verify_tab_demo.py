# -*- coding: utf-8 -*-
"""Демо-обёртка вкладки верификации для визуальной проверки дизайна.

Запуск: python -m streamlit run dev/verify_tab_demo.py
Рендерит render_verify_tab с первыми пролётами из реального проекта,
без сайдбара и остального приложения. Если реальных данных на машине нет —
подставляет синтетические пролёты (кадры не грузятся, но UI проверяем).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup as _setup_logging
_setup_logging()

import streamlit as st

from storage import project as proj
from storage.project import PassData
from ui.verify_tab import render_verify_tab

VIDEO = r"K:\disk d\fpv\!video\2026-06-30\20260630-202456.ts"
N = 999          # все пролёты — как в реальном приложении
DEMO_FILES = 3   # сколько «файлов» имитируем для кнопки «и следующий файл»

st.set_page_config(page_title="Verify tab demo", layout="wide")

if "demo_passes" not in st.session_state:
    data = proj.load(Path(VIDEO))
    if data is not None:
        passes = data.passes[:N]
        # в реальном JSON всё верифицировано — сбрасываем пару для демо «не проверен»
        for _idx in (1, 5):
            if _idx < len(passes):
                passes[_idx].verified = None
    else:
        # Нет реальных данных: синтетика, чтобы стенд всё равно поднимался.
        passes = [
            PassData(frame=100 + i * 300, time_sec=1.5 + i * 5.0,
                     similarity=0.70 + 0.01 * i,
                     verified=(True if i % 3 == 0 else None))
            for i in range(9)
        ]
    st.session_state["demo_passes"] = passes
    st.session_state["demo_cache"] = {}
    st.session_state["demo_file_idx"] = 0

_passes = st.session_state["demo_passes"]


def _on_verified_change(idx: int, value) -> None:
    _passes[idx].verified = value


def _on_next_video() -> bool:
    """Имитация переключения на следующий файл списка."""
    if st.session_state["demo_file_idx"] + 1 >= DEMO_FILES:
        return False
    st.session_state["demo_file_idx"] += 1
    return True


st.caption(
    f"demo: файл {st.session_state['demo_file_idx'] + 1} из {DEMO_FILES} · "
    f"не проверено: {sum(1 for p in _passes if p.verified is None)}"
)

render_verify_tab(
    video_path=VIDEO,
    passes=_passes,
    frames_cache=st.session_state["demo_cache"],
    on_verified_change=_on_verified_change,
    on_use_as_ref=lambda bgr: None,
    on_next_video=_on_next_video,
)
