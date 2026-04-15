"""
Streamlit sidebar component for FPV lap timing app.
Returns a config dict with all current settings.
"""
from __future__ import annotations

import subprocess
import streamlit as st

from core.models import MODELS, DEFAULT_MODEL
from storage.references import RefImage, from_file, from_clipboard


def _pick_folder_dialog() -> str | None:
    """Open a folder picker by spawning a child Python process with tkinter.

    Running in a subprocess avoids COM/STA threading issues inside Streamlit.
    wm_attributes topmost + focus_force ensure the dialog surfaces above the browser.
    """
    import sys
    import logging
    log = logging.getLogger(__name__)

    child_script = "\n".join([
        "import tkinter as tk",
        "from tkinter import filedialog",
        "root = tk.Tk()",
        "root.withdraw()",
        "root.wm_attributes('-topmost', True)",
        "root.lift()",
        "root.focus_force()",
        "path = filedialog.askdirectory(title='Выберите папку с видео')",
        "root.destroy()",
        "print(path, end='')",
    ])
    try:
        result = subprocess.run(
            [sys.executable, "-c", child_script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            log.warning("Folder picker stderr: %s", result.stderr.strip())
        path = result.stdout.strip()
        log.debug("Folder picker result: %r", path)
        return path if path else None
    except Exception as exc:
        log.exception("Folder picker failed: %s", exc)
        return None


def render_sidebar() -> dict:
    """
    Render the application sidebar and return the current configuration dict.

    Returns
    -------
    dict with keys:
        folder, model_key, threshold, min_lap_sec, sample_every,
        prominence, regularity_filter, regularity_tolerance,
        best_n, refs
    """
    # Initialise session state on first run
    if "sidebar_refs" not in st.session_state:
        st.session_state["sidebar_refs"] = []
    if "_sidebar_folder_input" not in st.session_state:
        st.session_state["_sidebar_folder_input"] = ""
    if "sidebar_model_key" not in st.session_state:
        st.session_state["sidebar_model_key"] = DEFAULT_MODEL
    if "sidebar_threshold" not in st.session_state:
        st.session_state["sidebar_threshold"] = 0.60
    if "sidebar_min_lap_sec" not in st.session_state:
        st.session_state["sidebar_min_lap_sec"] = 5.0
    if "sidebar_sample_every" not in st.session_state:
        st.session_state["sidebar_sample_every"] = 3
    if "sidebar_prominence" not in st.session_state:
        st.session_state["sidebar_prominence"] = 0.05
    if "sidebar_regularity_filter" not in st.session_state:
        st.session_state["sidebar_regularity_filter"] = False
    if "sidebar_regularity_tolerance" not in st.session_state:
        st.session_state["sidebar_regularity_tolerance"] = 0.30
    if "sidebar_best_n" not in st.session_state:
        st.session_state["sidebar_best_n"] = 3

    with st.sidebar:
        st.title("FPV Lap Timer")
        st.markdown("---")

        # ── Folder picker ──────────────────────────────────────────────
        st.subheader("📁 Папка с видео")
        col_btn, col_clear = st.columns([3, 1])
        with col_btn:
            if st.button("Выбрать папку…", width='stretch'):
                chosen = _pick_folder_dialog()
                if chosen:
                    st.session_state["_sidebar_folder_input"] = chosen
        with col_clear:
            if st.button("✕", width='stretch', help="Очистить путь"):
                st.session_state["_sidebar_folder_input"] = ""

        # value= не передаём — Streamlit сам берёт из st.session_state["_sidebar_folder_input"]
        st.text_input(
            "Путь к папке",
            key="_sidebar_folder_input",
            label_visibility="collapsed",
            placeholder="C:\\Videos\\FPV",
        )

        st.markdown("---")

        # ── Reference images ───────────────────────────────────────────
        st.subheader("🎯 Референсные кадры ворот")

        uploaded = st.file_uploader(
            "Загрузить изображения",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="sidebar_uploader",
            label_visibility="visible",
        )
        if uploaded:
            existing_names = {r.name for r in st.session_state["sidebar_refs"]}
            for uf in uploaded:
                if uf.name not in existing_names:
                    try:
                        ref = from_file(uf)
                        st.session_state["sidebar_refs"].append(ref)
                        existing_names.add(uf.name)
                    except Exception as exc:
                        st.warning(f"Не удалось загрузить {uf.name}: {exc}")

        if st.button("📋 Вставить из буфера", width='stretch'):
            ref = from_clipboard()
            if ref is None:
                st.warning("В буфере обмена нет изображения.")
            else:
                st.session_state["sidebar_refs"].append(ref)
                st.success(f"Добавлен кадр из буфера: {ref.name}")

        # Display current refs as thumbnails
        refs: list[RefImage] = st.session_state["sidebar_refs"]
        if refs:
            st.caption(f"Загружено: {len(refs)} реф.")
            to_remove = []
            # Show 3 thumbnails per row
            cols_per_row = 3
            indexed = list(enumerate(refs))  # [(global_idx, ref), ...]
            rows = [indexed[i : i + cols_per_row] for i in range(0, len(indexed), cols_per_row)]
            for row_refs in rows:
                thumb_cols = st.columns(cols_per_row)
                for col_idx, (ref_global_idx, ref) in enumerate(row_refs):
                    with thumb_cols[col_idx]:
                        try:
                            thumb = ref.thumbnail_rgb()
                            st.image(thumb, caption=ref.name[:10], width='stretch')
                        except Exception:
                            st.text(ref.name[:10])
                        if st.button("×", key=f"rm_ref_{ref_global_idx}", help=f"Удалить {ref.name}"):
                            to_remove.append(ref_global_idx)
            for idx in sorted(to_remove, reverse=True):
                st.session_state["sidebar_refs"].pop(idx)
                st.rerun()
        else:
            st.caption("Нет загруженных референсов")

        st.markdown("---")

        # ── Model selector ─────────────────────────────────────────────
        st.subheader("🧠 Модель")
        model_keys = list(MODELS.keys())
        model_labels = {k: f"{MODELS[k].name} — {MODELS[k].description}" for k in model_keys}

        try:
            current_model_idx = model_keys.index(st.session_state["sidebar_model_key"])
        except ValueError:
            current_model_idx = model_keys.index(DEFAULT_MODEL)

        selected_key = st.selectbox(
            "Выбор модели",
            options=model_keys,
            index=current_model_idx,
            format_func=lambda k: model_labels[k],
            key="sidebar_model_select",
            label_visibility="collapsed",
        )
        st.session_state["sidebar_model_key"] = selected_key

        info = MODELS[selected_key]
        st.caption(f"{info.speed_note} · {info.size_mb} МБ · {info.backend}")

        st.markdown("---")

        # ── Detection parameters ───────────────────────────────────────
        st.subheader("⚙️ Параметры детекции")

        threshold = st.slider(
            "Порог сходства",
            min_value=0.10,
            max_value=0.99,
            value=st.session_state["sidebar_threshold"],
            step=0.01,
            key="sidebar_threshold_slider",
            help="Минимальное косинусное сходство для засчитывания пролёта",
        )
        st.session_state["sidebar_threshold"] = threshold

        min_lap_sec = st.slider(
            "Мин. длина круга (сек)",
            min_value=1.0,
            max_value=60.0,
            value=st.session_state["sidebar_min_lap_sec"],
            step=0.5,
            key="sidebar_min_lap_slider",
            help="Минимальный промежуток между двумя пролётами",
        )
        st.session_state["sidebar_min_lap_sec"] = min_lap_sec

        sample_every = st.slider(
            "Анализировать каждый N-й кадр",
            min_value=1,
            max_value=30,
            value=st.session_state["sidebar_sample_every"],
            step=1,
            key="sidebar_sample_every_slider",
            help="1 = каждый кадр (медленнее), 5 = каждый 5-й (быстрее)",
        )
        st.session_state["sidebar_sample_every"] = sample_every

        prominence = st.slider(
            "Выраженность пика",
            min_value=0.01,
            max_value=0.50,
            value=st.session_state["sidebar_prominence"],
            step=0.01,
            key="sidebar_prominence_slider",
            help="Минимальная выраженность пика (scipy prominence)",
        )
        st.session_state["sidebar_prominence"] = prominence

        st.markdown("---")

        # ── Regularity filter ──────────────────────────────────────────
        st.subheader("📐 Фильтр регулярности")

        regularity_filter = st.toggle(
            "Фильтровать нерегулярные круги",
            value=st.session_state["sidebar_regularity_filter"],
            key="sidebar_regularity_toggle",
            help="Убрать выбросы по времени круга",
        )
        st.session_state["sidebar_regularity_filter"] = regularity_filter

        regularity_tolerance = st.slider(
            "Допуск отклонения",
            min_value=0.05,
            max_value=1.0,
            value=st.session_state["sidebar_regularity_tolerance"],
            step=0.05,
            key="sidebar_regularity_tol_slider",
            help="Макс. относительное отклонение от медианного круга (0.3 = 30%)",
            disabled=not regularity_filter,
        )
        st.session_state["sidebar_regularity_tolerance"] = regularity_tolerance

        st.markdown("---")

        # ── Best N consecutive laps ────────────────────────────────────
        st.subheader("🏆 Лучшие N подряд")

        best_n = st.slider(
            "Количество последовательных кругов",
            min_value=1,
            max_value=10,
            value=st.session_state["sidebar_best_n"],
            step=1,
            key="sidebar_best_n_slider",
            help="Для поиска лучшей серии кругов подряд",
        )
        st.session_state["sidebar_best_n"] = best_n

    return {
        "folder": st.session_state.get("_sidebar_folder_input") or None,
        "model_key": st.session_state["sidebar_model_key"],
        "threshold": st.session_state["sidebar_threshold"],
        "min_lap_sec": st.session_state["sidebar_min_lap_sec"],
        "sample_every": st.session_state["sidebar_sample_every"],
        "prominence": st.session_state["sidebar_prominence"],
        "regularity_filter": st.session_state["sidebar_regularity_filter"],
        "regularity_tolerance": st.session_state["sidebar_regularity_tolerance"],
        "best_n": st.session_state["sidebar_best_n"],
        "refs": list(st.session_state["sidebar_refs"]),
    }
