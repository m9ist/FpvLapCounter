"""
Graph + frame viewer tab for FPV lap timing app.
Shows the similarity signal, detected passes, and a navigable frame preview.
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

from core.osd_reader import OSDRegion, DEFAULT_REGION
from storage.project import PassData


# Maximum frames to keep in the cache dict
_CACHE_LIMIT = 200


# ---------------------------------------------------------------------------
# Frame loading helper
# ---------------------------------------------------------------------------

def _load_frame(video_path: str, frame_idx: int, frames_cache: dict) -> np.ndarray | None:
    """Load frame_idx from video, using / populating frames_cache.

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
        # Evict oldest entries if cache is full
        if len(frames_cache) >= _CACHE_LIMIT:
            oldest_key = next(iter(frames_cache))
            del frames_cache[oldest_key]
        frames_cache[frame_idx] = frame
        return frame
    finally:
        cap.release()


def _video_info(video_path: str) -> tuple[float, int]:
    """Return (fps, total_frames) for the video, or (30.0, 0) on error."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30.0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _sk(video_path: str, suffix: str) -> str:
    """Build a stable session-state key scoped to a video path."""
    safe = video_path.replace("\\", "/").replace(":", "_").replace("/", "_")
    return f"graph_{safe}_{suffix}"


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_graph_tab(
    video_path: str,
    similarities: np.ndarray | None,
    timestamps: np.ndarray | None,
    passes: list[PassData],
    params: dict,
    frames_cache: dict,
    on_use_as_ref: Callable,
    on_set_osd_region: Callable,
) -> None:
    """
    Render the similarity graph and frame viewer for one video.

    Parameters
    ----------
    video_path     : Absolute path to the video file.
    similarities   : 1-D float array of cosine similarities (or None).
    timestamps     : 1-D float array of timestamps in seconds (or None).
    passes         : List of PassData from storage.project.
    params         : Dict with 'threshold' key (float).
    frames_cache   : Mutable dict {frame_idx: bgr_frame}.
    on_use_as_ref  : Callable(bgr_frame) — user wants this frame as a reference.
    on_set_osd_region : Callable(OSDRegion) — user configured the OSD region.
    """
    fps, total_frames = _video_info(video_path)
    if total_frames == 0:
        st.error("Не удалось открыть видеофайл.")
        return

    # Session-state keys
    sk_frame = _sk(video_path, "frame_idx")
    if sk_frame not in st.session_state:
        st.session_state[sk_frame] = 0

    current_frame: int = int(st.session_state[sk_frame])
    current_frame = max(0, min(current_frame, total_frames - 1))

    threshold = params.get("threshold", 0.6)

    # ── Similarity graph ───────────────────────────────────────────────
    st.subheader("График сходства")

    if not _PLOTLY_OK:
        st.warning("plotly не установлен. Установите: pip install plotly")
    elif similarities is None or timestamps is None or len(similarities) == 0:
        st.info("Нет данных анализа. Запустите анализ видео.")
    else:
        _render_plotly_graph(
            video_path=video_path,
            similarities=similarities,
            timestamps=timestamps,
            passes=passes,
            threshold=threshold,
            fps=fps,
            sk_frame=sk_frame,
            frames_cache=frames_cache,
        )

    st.markdown("---")

    # ── Frame viewer ───────────────────────────────────────────────────
    st.subheader("Просмотр кадров")

    _render_frame_viewer(
        video_path=video_path,
        total_frames=total_frames,
        fps=fps,
        similarities=similarities,
        timestamps=timestamps,
        passes=passes,
        frames_cache=frames_cache,
        sk_frame=sk_frame,
        on_use_as_ref=on_use_as_ref,
        on_set_osd_region=on_set_osd_region,
    )


# ---------------------------------------------------------------------------
# Plotly graph
# ---------------------------------------------------------------------------

def _render_plotly_graph(
    video_path: str,
    similarities: np.ndarray,
    timestamps: np.ndarray,
    passes: list[PassData],
    threshold: float,
    fps: float,
    sk_frame: str,
    frames_cache: dict,
) -> None:
    from scipy.signal import savgol_filter

    fig = go.Figure()

    # Raw similarity (thin, semi-transparent)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=similarities,
        mode="lines",
        name="Сходство (сырое)",
        line=dict(color="rgba(255,165,0,0.35)", width=1),
        hovertemplate="t=%{x:.2f}s<br>sim=%{y:.3f}<extra></extra>",
    ))

    # Smoothed similarity
    window = max(3, int(len(similarities) * 0.02) | 1)
    window = min(window, len(similarities) - (1 if len(similarities) % 2 == 0 else 0))
    if window >= 3 and window < len(similarities):
        smoothed = savgol_filter(similarities, window_length=window, polyorder=2)
    else:
        smoothed = similarities.copy()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=smoothed,
        mode="lines",
        name="Сходство (сглаженное)",
        line=dict(color="orange", width=2),
        hovertemplate="t=%{x:.2f}s<br>sim=%{y:.3f}<extra></extra>",
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color="red", width=1, dash="dash"),
        annotation_text=f"порог={threshold:.2f}",
        annotation_position="bottom right",
    )

    # Pass markers
    confirmed_ts, confirmed_sim, confirmed_hover = [], [], []
    rejected_ts, rejected_sim, rejected_hover = [], [], []

    for p in passes:
        if p.verified is False:
            rejected_ts.append(p.time_sec)
            rejected_sim.append(p.similarity)
            rejected_hover.append(f"t={p.time_sec:.2f}s  sim={p.similarity:.3f}  [ФЕЙК]")
        else:
            confirmed_ts.append(p.time_sec)
            confirmed_sim.append(p.similarity)
            confirmed_hover.append(f"t={p.time_sec:.2f}s  sim={p.similarity:.3f}")

    if confirmed_ts:
        fig.add_trace(go.Scatter(
            x=confirmed_ts,
            y=confirmed_sim,
            mode="markers",
            name="Пролёт",
            marker=dict(symbol="star", color="lime", size=12, line=dict(color="green", width=1)),
            text=confirmed_hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    if rejected_ts:
        fig.add_trace(go.Scatter(
            x=rejected_ts,
            y=rejected_sim,
            mode="markers",
            name="Фейк",
            marker=dict(symbol="x", color="gray", size=10, line=dict(color="darkgray", width=1)),
            text=rejected_hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    # Layout
    fig.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(
            title="Время (с)",
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(title="Сходство", range=[-0.05, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )

    event = st.plotly_chart(fig, width='stretch', key=f"graph_{sk_frame}", on_select="rerun")

    # Handle click → jump to frame.
    # Plotly keeps the selection alive across reruns, so we track the last
    # processed click and ignore repeats — otherwise nav buttons (+1 / -1 …)
    # get overwritten by the stale selection on the very next render.
    sk_last_click = f"last_click_{sk_frame}"
    if event and hasattr(event, "selection") and event.selection:
        pts = event.selection.get("points", [])
        if pts:
            clicked_x = pts[0].get("x")
            if clicked_x is not None and clicked_x != st.session_state.get(sk_last_click):
                t = float(clicked_x)
                frame_idx = int(t * fps)
                st.session_state[sk_frame] = frame_idx
                st.session_state[f"time_slider_{sk_frame}"] = t
                st.session_state[sk_last_click] = clicked_x


# ---------------------------------------------------------------------------
# Frame viewer
# ---------------------------------------------------------------------------

def _nearest_pass(passes: list[PassData], time_sec: float) -> tuple[PassData | None, float]:
    """Return (nearest_pass, delta_seconds). delta is signed."""
    if not passes:
        return None, float("inf")
    best = min(passes, key=lambda p: abs(p.time_sec - time_sec))
    return best, best.time_sec - time_sec


def _render_frame_viewer(
    video_path: str,
    total_frames: int,
    fps: float,
    similarities: np.ndarray | None,
    timestamps: np.ndarray | None,
    passes: list[PassData],
    frames_cache: dict,
    sk_frame: str,
    on_use_as_ref: Callable,
    on_set_osd_region: Callable,
) -> None:
    current_frame: int = int(st.session_state.get(sk_frame, 0))
    current_frame = max(0, min(current_frame, total_frames - 1))

    max_sec = (total_frames - 1) / fps if fps > 0 else 0.0
    sk_slider = f"time_slider_{sk_frame}"

    # Initialise slider session-state key (only on first render)
    if sk_slider not in st.session_state:
        st.session_state[sk_slider] = current_frame / fps if fps > 0 else 0.0

    # ── Navigation buttons ─────────────────────────────────────────────
    nav_cols = st.columns(8)
    deltas = [-10, -5, -2, -1, 1, 2, 5, 10]
    for col, delta in zip(nav_cols, deltas):
        label = f"{delta:+d}"
        with col:
            if st.button(label, key=f"nav_{sk_frame}_{delta}", width='stretch'):
                new_f = max(0, min(current_frame + delta, total_frames - 1))
                current_frame = new_f
                st.session_state[sk_frame] = new_f
                # Sync slider — must update its own session-state key
                st.session_state[sk_slider] = new_f / fps if fps > 0 else 0.0

    # ── Time slider ────────────────────────────────────────────────────
    # No value= parameter: Streamlit reads from st.session_state[sk_slider]
    new_sec = st.slider(
        "Время (сек)",
        min_value=0.0,
        max_value=float(max_sec),
        step=1.0 / fps if fps > 0 else 0.033,
        format="%.2f",
        key=sk_slider,
    )
    # Sync slider → frame index
    # Use round() not int(): int(1910/30*30) = int(1909.9999…) = 1909, off by one!
    new_frame_from_slider = max(0, min(round(new_sec * fps), total_frames - 1))
    if new_frame_from_slider != current_frame:
        current_frame = new_frame_from_slider
        st.session_state[sk_frame] = current_frame

    # ── Load and display frame ─────────────────────────────────────────
    frame_bgr = _load_frame(video_path, current_frame, frames_cache)

    col_img, col_info = st.columns([3, 2])

    with col_img:
        if frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, width='stretch', caption=f"Кадр #{current_frame}")
        else:
            st.warning("Не удалось загрузить кадр.")

    with col_info:
        t_sec = current_frame / fps if fps > 0 else 0.0
        st.metric("Время", f"{t_sec:.3f} с")
        st.metric("Кадр", str(current_frame))

        # Similarity at current position
        if similarities is not None and timestamps is not None and len(similarities) > 0:
            idx = int(np.argmin(np.abs(timestamps - t_sec)))
            sim_val = float(similarities[idx])
            st.metric("Сходство", f"{sim_val:.4f}")
        else:
            sim_val = None
            st.metric("Сходство", "—")

        # Pass info
        nearest_pass, delta_sec = _nearest_pass(passes, t_sec)
        if nearest_pass is not None:
            if abs(delta_sec) < 0.5:
                pass_idx = passes.index(nearest_pass) + 1
                st.success(f"✅ Пролёт #{pass_idx}")
            else:
                sign = "+" if delta_sec > 0 else ""
                st.info(f"Ближайший пролёт: {sign}{delta_sec:.1f} с")

        st.markdown("---")

        # Use as reference button
        if st.button("🎯 Использовать как референс", key=f"use_ref_{sk_frame}", width='stretch'):
            if frame_bgr is not None:
                on_use_as_ref(frame_bgr)
                st.success("Кадр добавлен как референс")
            else:
                st.warning("Нет кадра для добавления")

        # OSD region configurator
        with st.expander("✂️ Настроить регион OSD"):
            st.caption("Координаты в долях от размера кадра (0..1)")
            osd_x = st.number_input("X (лево)", 0.0, 1.0, DEFAULT_REGION.x, step=0.01, key=f"osd_x_{sk_frame}")
            osd_y = st.number_input("Y (верх)", 0.0, 1.0, DEFAULT_REGION.y, step=0.01, key=f"osd_y_{sk_frame}")
            osd_w = st.number_input("Ширина", 0.01, 1.0, DEFAULT_REGION.w, step=0.01, key=f"osd_w_{sk_frame}")
            osd_h = st.number_input("Высота", 0.01, 1.0, DEFAULT_REGION.h, step=0.01, key=f"osd_h_{sk_frame}")
            if st.button("Применить регион OSD", key=f"osd_apply_{sk_frame}", width='stretch'):
                region = OSDRegion(x=osd_x, y=osd_y, w=osd_w, h=osd_h)
                on_set_osd_region(region)
                st.success("Регион OSD обновлён")

            # Preview: draw OSD rectangle on current frame
            if frame_bgr is not None:
                preview = frame_bgr.copy()
                h_px, w_px = preview.shape[:2]
                region_preview = OSDRegion(x=osd_x, y=osd_y, w=osd_w, h=osd_h)
                x0, y0, x1, y1 = region_preview.to_pixels(h_px, w_px)
                cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                st.image(preview_rgb, caption="Предпросмотр OSD-региона", width='stretch')
