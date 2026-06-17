"""
Laps table, comparison tab for FPV lap timing app.
Displays per-video lap results and a cross-video comparison table.
"""
from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from core.lap_analyzer import LapResult, Lap


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_sec(seconds: float) -> str:
    """Format seconds as MM:SS.ss string."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--.--"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    whole = int(secs)
    cs = int(round((secs - whole) * 100))
    if cs >= 100:
        cs = 99
    return f"{minutes:02d}:{whole:02d}.{cs:02d}"


def _total_sec(laps: list[Lap]) -> float:
    return sum(lap.duration_sec for lap in laps)


def _avg_sec(laps: list[Lap]) -> float:
    if not laps:
        return float("nan")
    return _total_sec(laps) / len(laps)


# ---------------------------------------------------------------------------
# Per-video laps tab
# ---------------------------------------------------------------------------

def render_laps_tab(
    result: LapResult,
    best_n: int,
    video_name: str,
) -> None:
    """
    Render the laps table and metrics for a single video.

    Parameters
    ----------
    result     : LapResult from core.lap_analyzer.analyze().
    best_n     : Window size for the best-consecutive query.
    video_name : Display name (filename) for the video.
    """
    laps = result.laps
    best_lap = result.best_lap
    best_n_laps: list[Lap] | None = result.best_n.get(best_n)

    st.subheader(f"Результаты: {video_name}")

    if not laps:
        st.warning("Нет засчитанных кругов.")
        return

    # ── Metrics row ────────────────────────────────────────────────────
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    with mcol1:
        if best_lap:
            st.metric("🥇 Лучший круг", _fmt_sec(best_lap.duration_sec))
        else:
            st.metric("🥇 Лучший круг", "—")

    with mcol2:
        if best_n_laps:
            total = _total_sec(best_n_laps)
            avg = _avg_sec(best_n_laps)
            st.metric(
                f"🏆 Лучшие {best_n} кругов",
                _fmt_sec(total),
                delta=f"avg {_fmt_sec(avg)}",
                delta_color="off",
            )
        else:
            st.metric(f"🏆 Лучшие {best_n} кругов", "—")

    with mcol3:
        avg_all = _avg_sec(laps)
        st.metric("📊 Средний круг", _fmt_sec(avg_all))

    with mcol4:
        st.metric("🔢 Всего кругов", str(len(laps)))

    st.markdown("---")

    # ── Build DataFrame ────────────────────────────────────────────────
    best_lap_num = best_lap.number if best_lap else None
    best_n_nums: set[int] = set()
    if best_n_laps:
        best_n_nums = {lap.number for lap in best_n_laps}

    rows = []
    for lap in laps:
        rows.append({
            "№": lap.number,
            "Время круга": _fmt_sec(lap.duration_sec),
            "Старт (с)": round(lap.start_pass.time_sec, 3),
            "Сходство": round(lap.start_pass.similarity, 4),
            "Сек.": round(lap.duration_sec, 3),
        })

    df = pd.DataFrame(rows)

    # ── Styling ────────────────────────────────────────────────────────
    def _highlight_row(row: pd.Series) -> list[str]:
        lap_num = int(row["№"])
        if lap_num == best_lap_num:
            return ["background-color: #d4f5e0; color: #1a4731; font-weight: bold"] * len(row)
        if lap_num in best_n_nums:
            return ["background-color: #fff8d6; color: #5a4000"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_highlight_row, axis=1)

    st.dataframe(
        styled,
        width='stretch',
        hide_index=True,
        column_config={
            "№": st.column_config.NumberColumn(width="small"),
            "Время круга": st.column_config.TextColumn(width="medium"),
            "Старт (с)": st.column_config.NumberColumn(format="%.3f", width="medium"),
            "Сходство": st.column_config.NumberColumn(format="%.4f", width="medium"),
            "Сек.": st.column_config.NumberColumn(format="%.3f", width="medium"),
        },
    )

    # ── Legend ─────────────────────────────────────────────────────────
    st.caption("🟢 Лучший круг  🟡 Лучшие N подряд")

    # ── CSV export ─────────────────────────────────────────────────────
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Экспорт CSV",
        data=csv_bytes,
        file_name=f"{video_name}_laps.csv",
        mime="text/csv",
        width='content',
    )


# ---------------------------------------------------------------------------
# Cross-video comparison tab
# ---------------------------------------------------------------------------

def render_compare_tab(
    all_results: dict[str, LapResult],
    best_n: int,
    on_delete_videos: Callable[[list[str]], None] | None = None,
    no_lap_paths: list[str] | None = None,
) -> None:
    """
    Render a comparison table across multiple analyzed videos.

    Parameters
    ----------
    all_results   : Mapping video_path -> LapResult (videos with laps).
    best_n        : Window size for best-consecutive-n query.
    no_lap_paths  : Paths of analyzed videos that have zero laps.
    """
    st.subheader("Сравнение видео")

    if not all_results and not no_lap_paths:
        st.info("Нет проанализированных видео для сравнения.")
        return

    rows = []
    for video_path, result in all_results.items():
        name = video_path.replace("\\", "/").split("/")[-1]
        laps = result.laps
        best_lap = result.best_lap
        best_n_laps = result.best_n.get(best_n)

        best_lap_sec = best_lap.duration_sec if best_lap else float("inf")
        best_lap_str = _fmt_sec(best_lap_sec) if best_lap else "—"

        if best_n_laps:
            best_n_total = _total_sec(best_n_laps)
            best_n_str = _fmt_sec(best_n_total)
            best_n_avg_str = _fmt_sec(_avg_sec(best_n_laps))
        else:
            best_n_total = float("inf")
            best_n_str = "—"
            best_n_avg_str = "—"

        avg_str = _fmt_sec(_avg_sec(laps)) if laps else "—"

        first_pass_str = _fmt_sec(laps[0].start_pass.time_sec) if laps else "—"

        rows.append({
            "Видео": name,
            "Кругов": len(laps),
            "Лучший круг": best_lap_str,
            f"Лучшие {best_n}": best_n_str,
            f"Avg {best_n}": best_n_avg_str,
            "1й пролёт": first_pass_str,
            "Средний": avg_str,
            "_path": video_path,
            "_best_lap_sec": best_lap_sec,
            "_best_n_total": best_n_total,
        })

    # Sort by best_n_total ascending (best = smallest)
    rows.sort(key=lambda r: (r["_best_n_total"], r["_best_lap_sec"]))

    # Append no-laps videos at the bottom (not sorted, just listed)
    for path in (no_lap_paths or []):
        name = path.replace("\\", "/").split("/")[-1]
        rows.append({
            "Видео": f"⚠️ {name}",
            "Кругов": 0,
            "Лучший круг": "—",
            f"Лучшие {best_n}": "—",
            f"Avg {best_n}": "—",
            "1й пролёт": "—",
            "Средний": "—",
            "_path": path,
            "_best_lap_sec": float("inf"),
            "_best_n_total": float("inf"),
        })

    df_display = pd.DataFrame([
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in rows
    ])

    # ── Global best metrics ────────────────────────────────────────────
    best_overall_lap = min(
        (r["_best_lap_sec"] for r in rows if math.isfinite(r["_best_lap_sec"])),
        default=None,
    )
    best_overall_n = min(
        (r["_best_n_total"] for r in rows if math.isfinite(r["_best_n_total"])),
        default=None,
    )

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        val = _fmt_sec(best_overall_lap) if best_overall_lap is not None else "—"
        st.metric("🌍 Лучший круг (все видео)", val)
    with gcol2:
        val = _fmt_sec(best_overall_n) if best_overall_n is not None else "—"
        st.metric(f"🌍 Лучшие {best_n} подряд (все видео)", val)

    st.markdown("---")

    # ── Medals ─────────────────────────────────────────────────────────
    _MEDALS = ["🥇", "🥈", "🥉"]

    def _highlight_compare(row: pd.Series) -> list[str]:
        rank = row.name  # index = rank after sort
        if rank == 0:
            return ["background-color: #fff3cd; color: #7a5000; font-weight: bold"] * len(row)  # gold
        if rank == 1:
            return ["background-color: #f0f0f0; color: #333333; font-weight: bold"] * len(row)  # silver
        if rank == 2:
            return ["background-color: #fde8d8; color: #7a3500; font-weight: bold"] * len(row)  # bronze
        return [""] * len(row)

    # Add medal prefix to top-3 rows that actually have laps
    n_with_laps = len(all_results)
    df_display = df_display.copy()
    for i, medal in enumerate(_MEDALS):
        if i < n_with_laps:
            df_display.at[i, "Видео"] = f"{medal} {df_display.at[i, 'Видео']}"

    # ── Editable table with "Оставить" checkbox ────────────────────────
    df_edit = df_display.copy()
    df_edit.insert(0, "Оставить", False)

    edited_df = st.data_editor(
        df_edit,
        key="compare_table",
        width='stretch',
        hide_index=True,
        column_config={
            "Оставить": st.column_config.CheckboxColumn("Оставить", width="small"),
            "Кругов":   st.column_config.NumberColumn(width="small"),
        },
        disabled=[c for c in df_edit.columns if c != "Оставить"],
    )

    # Paths whose "Оставить" is unchecked
    paths_to_delete = [
        rows[i]["_path"]
        for i in range(min(len(rows), len(edited_df)))
        if not edited_df.iloc[i]["Оставить"]
    ]

    # ── Open in player buttons ─────────────────────────────────────────
    st.markdown("**▶ Открыть в плеере:**")
    btn_cols = st.columns(max(len(rows), 1))
    for i, (row, col) in enumerate(zip(rows, btn_cols)):
        short = row["Видео"][:25]
        with col:
            if st.button(short, key=f"open_video_{i}", width='stretch',
                         help=row["_path"]):
                try:
                    os.startfile(row["_path"])
                except Exception as exc:
                    st.error(f"Не удалось открыть файл: {exc}")

    # ── Delete unchecked videos ────────────────────────────────────────
    _SK_DEL = "compare_delete_paths"

    if paths_to_delete and on_delete_videos:
        if not st.session_state.get(_SK_DEL):
            st.markdown("---")
            if st.button(
                f"🗑 Удалить не отмеченные ({len(paths_to_delete)})",
                key="compare_delete_btn",
                type="primary",
                width="content",
            ):
                st.session_state[_SK_DEL] = paths_to_delete
                st.rerun()

    if st.session_state.get(_SK_DEL):
        confirmed = st.session_state[_SK_DEL]
        with st.container(border=True):
            st.error(
                f"⚠️ Будет **безвозвратно удалено** {len(confirmed)} файлов:"
            )
            for p in confirmed:
                st.text(Path(p).name)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("❌ Отмена", key="compare_del_cancel", width='stretch'):
                    del st.session_state[_SK_DEL]
                    st.rerun()
            with c2:
                if st.button("🗑 Удалить", key="compare_del_confirm",
                             type="primary", width='stretch'):
                    on_delete_videos(confirmed)
                    del st.session_state[_SK_DEL]
                    st.rerun()
