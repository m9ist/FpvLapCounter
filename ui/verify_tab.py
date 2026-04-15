"""
Verification grid tab for FPV lap timing app.
Lets the user confirm or reject each detected gate pass candidate.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import streamlit as st

from storage.project import PassData


_CACHE_LIMIT = 200

_CARD_VERIFIED_CSS = """
<style>
.card-ok   { border: 3px solid #28a745; background-color: rgba(40,167,69,0.12);
             border-radius: 8px; padding: 4px; margin-bottom: 4px; }
.card-bad  { border: 3px solid #dc3545; background-color: rgba(220,53,69,0.10);
             border-radius: 8px; padding: 4px; margin-bottom: 4px; }
.card-none { border: 3px solid #6c757d; background-color: transparent;
             border-radius: 8px; padding: 4px; margin-bottom: 4px; }
</style>
"""


def _load_frame(video_path: str, frame_idx: int, frames_cache: dict) -> np.ndarray | None:
    """Load a specific frame from video, using cache.

    HEVC (.ts) videos use B-frames with long GOPs.  A bare seek to frame_idx
    often lands mid-GOP without the required reference frames, producing
    corrupted output.  We seek back GOP_WARMUP frames and decode sequentially
    so the decoder can reconstruct all necessary reference frames first.
    """
    if frame_idx in frames_cache:
        return frames_cache[frame_idx]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        GOP_WARMUP = 60  # ~2 s at 30 fps; covers typical HEVC GOP size
        seek_to = max(0, frame_idx - GOP_WARMUP)
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
        frame = None
        for _ in range(frame_idx - seek_to + 1):
            ret, f = cap.read()
            if not ret:
                break
            frame = f
        if frame is None:
            return None
        if len(frames_cache) >= _CACHE_LIMIT:
            oldest = next(iter(frames_cache))
            del frames_cache[oldest]
        frames_cache[frame_idx] = frame
        return frame
    finally:
        cap.release()


def render_verify_tab(
    video_path: str,
    passes: list[PassData],
    frames_cache: dict,
    on_verified_change: Callable,  # (pass_idx: int, value: bool | None) -> None
    on_use_as_ref: Callable,       # (bgr_frame: np.ndarray) -> None
) -> None:
    """
    Render a verification grid for all detected gate pass candidates.

    Parameters
    ----------
    video_path         : Absolute path to the video file.
    passes             : List of PassData objects.
    frames_cache       : Mutable dict {frame_idx: bgr_frame} for caching.
    on_verified_change : Callback(pass_idx, True|False|None) when user changes status.
    on_use_as_ref      : Callback(bgr_frame) when user wants to use frame as reference.
    """
    st.markdown(_CARD_VERIFIED_CSS, unsafe_allow_html=True)

    if not passes:
        st.info("Нет обнаруженных пролётов для верификации. Запустите анализ.")
        return

    # ── Bulk action buttons ────────────────────────────────────────────
    col_all, col_reset = st.columns(2)
    with col_all:
        if st.button("✅ Подтвердить все", key="verify_all_btn", width='stretch'):
            for i in range(len(passes)):
                on_verified_change(i, True)
            st.rerun()
    with col_reset:
        if st.button("🔄 Сбросить верификацию", key="verify_reset_btn", width='stretch'):
            for i in range(len(passes)):
                on_verified_change(i, None)
            st.rerun()

    st.markdown(f"**Всего кандидатов: {len(passes)}**")
    st.markdown("---")

    # ── Pass grid (3 per row) ──────────────────────────────────────────
    cols_per_row = 3

    for row_start in range(0, len(passes), cols_per_row):
        row_passes = passes[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col_offset, (col, pass_data) in enumerate(zip(cols, row_passes)):
            pass_idx = row_start + col_offset
            _render_pass_card(
                col=col,
                pass_data=pass_data,
                pass_idx=pass_idx,
                video_path=video_path,
                frames_cache=frames_cache,
                on_verified_change=on_verified_change,
                on_use_as_ref=on_use_as_ref,
            )


def _render_pass_card(
    col,
    pass_data: PassData,
    pass_idx: int,
    video_path: str,
    frames_cache: dict,
    on_verified_change: Callable,
    on_use_as_ref: Callable,
) -> None:
    """Render a single verification card inside a Streamlit column."""
    with col:
        # Determine card border class based on verified state
        if pass_data.verified is True:
            css_class = "card-ok"
        elif pass_data.verified is False:
            css_class = "card-bad"
        else:
            css_class = "card-none"

        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)

        # Thumbnail
        frame_bgr = _load_frame(video_path, pass_data.frame, frames_cache)
        if frame_bgr is not None:
            thumb = cv2.resize(frame_bgr, (224, 126))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            st.image(thumb_rgb, width='stretch')
        else:
            st.warning("Кадр недоступен")

        # Pass info
        t = pass_data.time_sec
        minutes = int(t // 60)
        seconds = t - minutes * 60
        time_str = f"{minutes:02d}:{seconds:05.2f}"

        osd_str = f" | OSD: {pass_data.osd_time:.2f}с" if pass_data.osd_time is not None else ""
        st.caption(
            f"**#{pass_idx + 1}** · {time_str}{osd_str}\n\n"
            f"Сходство: {pass_data.similarity:.3f}"
        )

        # Status indicator
        if pass_data.verified is True:
            st.success("✅ Подтверждён", icon=None)
        elif pass_data.verified is False:
            st.error("❌ Фейк", icon=None)
        else:
            st.info("⬜ Не проверен", icon=None)

        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            ok_type = "primary" if pass_data.verified is True else "secondary"
            if st.button("✅", key=f"vok_{pass_idx}", width='stretch',
                         type=ok_type, help="Подтвердить пролёт"):
                new_val = None if pass_data.verified is True else True
                on_verified_change(pass_idx, new_val)
                st.rerun()

        with btn_col2:
            bad_type = "primary" if pass_data.verified is False else "secondary"
            if st.button("❌", key=f"vbad_{pass_idx}", width='stretch',
                         type=bad_type, help="Отметить как фейк"):
                new_val = None if pass_data.verified is False else False
                on_verified_change(pass_idx, new_val)
                st.rerun()

        with btn_col3:
            if st.button("🎯", key=f"vref_{pass_idx}", width='stretch',
                         help="Использовать как референс"):
                if frame_bgr is not None:
                    on_use_as_ref(frame_bgr)
                    st.success("Добавлен как референс")

        st.markdown("</div>", unsafe_allow_html=True)
