"""
Video list panel for FPV lap timing app.
Renders a scrollable list of video files with status badges and selection checkboxes.
"""
from __future__ import annotations

import streamlit as st


_STATUS_BADGE = {
    "new": "⬜ новый",
    "processing": "🔄 ...",
}

_ACTIVE_STYLE = (
    "background-color: #1e3a5f; border-radius: 6px; padding: 2px 4px;"
)
_INACTIVE_STYLE = (
    "background-color: transparent; border-radius: 6px; padding: 2px 4px;"
)

_CSS = """
<style>
div[data-testid="stVerticalBlock"] .video-list-btn button {
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 0.82rem !important;
    padding: 2px 6px !important;
}
</style>
"""


def _verify_badge(video: dict) -> str:
    """Return a short verification-status suffix for the video label.

    Returns:
      '· 🔍✓'        — all passes reviewed (verified True or False, none pending)
      '· ❓N'         — N passes still pending review
      ''              — no analysis data yet
    """
    data = video.get("data")
    if not data or not data.passes:
        return ""
    pending = sum(1 for p in data.passes if p.verified is None)
    if pending == 0:
        return " · 🔍✓"
    return f" · ❓{pending}"


def _status_label(video: dict) -> str:
    status = video.get("status", "new")
    if status == "done":
        n = video.get("lap_count", 0)
        return f"✅ {n} кругов{_verify_badge(video)}"
    if status == "no_laps":
        return f"⚠️ нет кругов{_verify_badge(video)}"
    return _STATUS_BADGE.get(status, status)


def render_video_list(videos: list[dict], active_idx: int) -> tuple[int, bool]:
    """
    Render a compact video list with checkboxes and status badges.

    Parameters
    ----------
    videos : list of dicts with keys:
        - path (str): full file path
        - status (str): "new" | "done" | "no_laps" | "processing"
        - selected (bool): current checkbox state
        - lap_count (int): number of detected laps (used when status=="done")
    active_idx : index of the currently active video

    Returns
    -------
    (new_active_idx, reprocess_requested)
        reprocess_requested=True when the user clicked "Обработать заново"
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    if not videos:
        st.info("Нет видеофайлов в выбранной папке.")
        return active_idx, False

    # ── Select all / deselect all ──────────────────────────────────────
    col_all, col_none = st.columns(2)
    with col_all:
        if st.button("Выбрать все", key="vlist_select_all", width='stretch'):
            for i, v in enumerate(videos):
                v["selected"] = True
                st.session_state[f"sel_{i}"] = True
    with col_none:
        if st.button("Снять все", key="vlist_deselect_all", width='stretch'):
            for i, v in enumerate(videos):
                v["selected"] = False
                st.session_state[f"sel_{i}"] = False

    # ── Reprocess button ───────────────────────────────────────────────
    selected_count = sum(1 for v in videos if v.get("selected", False))
    reprocess = st.button(
        f"♻️ Обработать заново ({selected_count})",
        key="vlist_reprocess",
        width='stretch',
        type="primary",
        disabled=selected_count == 0,
        help="Удалить сохранённые результаты и запустить анализ заново",
    )

    st.markdown("---")

    new_active = active_idx

    for i, video in enumerate(videos):
        path = video.get("path", "")
        filename = path.split("/")[-1].split("\\")[-1] if path else f"video_{i}"
        status_label = _status_label(video)
        is_active = i == active_idx

        # Sync checkbox state with session_state
        sel_key = f"sel_{i}"
        if sel_key not in st.session_state:
            st.session_state[sel_key] = video.get("selected", False)

        row_style = _ACTIVE_STYLE if is_active else _INACTIVE_STYLE
        st.markdown(f'<div style="{row_style}">', unsafe_allow_html=True)

        col_chk, col_btn = st.columns([1, 5])
        with col_chk:
            checked = st.checkbox(
                f"Выбрать видео {i}",
                key=sel_key,
                label_visibility="collapsed",
            )
            video["selected"] = checked

        with col_btn:
            btn_label = f"{filename}\n{status_label}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                btn_label,
                key=f"vlist_btn_{i}",
                width='stretch',
                type=btn_type,
                help=path,
            ):
                new_active = i

        st.markdown("</div>", unsafe_allow_html=True)

    return new_active, reprocess
