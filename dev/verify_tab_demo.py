# -*- coding: utf-8 -*-
"""Демо-обёртка вкладки верификации для визуальной проверки дизайна.

Запуск: python -m streamlit run dev/verify_tab_demo.py
Рендерит render_verify_tab с первыми пролётами из реального проекта,
без сайдбара и остального приложения.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup as _setup_logging
_setup_logging()

import streamlit as st

from storage import project as proj
from ui.verify_tab import render_verify_tab

VIDEO = r"K:\disk d\fpv\!video\2026-06-30\20260630-202456.ts"
N = 999  # все пролёты — как в реальном приложении

st.set_page_config(page_title="Verify tab demo", layout="wide")

if "demo_passes" not in st.session_state:
    data = proj.load(Path(VIDEO))
    passes = data.passes[:N]
    # в реальном JSON всё верифицировано — сбрасываем пару для демо «не проверен»
    for _idx in (1, 5):
        if _idx < len(passes):
            passes[_idx].verified = None
    st.session_state["demo_passes"] = passes
    st.session_state["demo_cache"] = {}

_passes = st.session_state["demo_passes"]


def _on_verified_change(idx: int, value) -> None:
    _passes[idx].verified = value


render_verify_tab(
    video_path=VIDEO,
    passes=_passes,
    frames_cache=st.session_state["demo_cache"],
    on_verified_change=_on_verified_change,
    on_use_as_ref=lambda bgr: None,
)
