"""
FPV Lap Counter
Главный файл приложения Streamlit.
"""

# ── Logging must be set up before any other import ───────────────────────────
from utils.logger import setup as _setup_logging
_setup_logging()

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

from core.models import MODELS, DEFAULT_MODEL
from core.detector import GateDetector, find_passes
from core.osd_reader import OSDReader, OSDRegion, DEFAULT_REGION
from core.lap_analyzer import Pass, analyze
from storage import project as proj
from storage.project import ProjectData, PassData, LapData
from storage.references import RefImage, from_frame
from ui.sidebar import render_sidebar
from ui.video_list import render_video_list
from ui.graph_tab import render_graph_tab
from ui.verify_tab import render_verify_tab
from ui.laps_tab import render_laps_tab, render_compare_tab

# ── Конфигурация ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FPV Lap Counter",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stButton>button { font-size: 0.8rem; }
div[data-testid="column"] { padding: 0 4px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "videos": [],           # list of video dicts
        "active_idx": 0,
        "detector": None,       # loaded GateDetector
        "detector_model": None, # which model is loaded
        "osd_reader": None,
        "frames_cache": {},     # video_path -> {frame_idx: bgr}
        "similarities": {},     # video_path -> np.ndarray
        "timestamps": {},       # video_path -> np.ndarray
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def scan_folder(folder: str) -> list[dict]:
    """Сканирует папку, возвращает список видео."""
    p = Path(folder)
    extensions = {".ts", ".mp4", ".avi", ".mov", ".mkv"}
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in extensions)
    videos = []
    for f in files:
        data = proj.load(f)
        status = "done" if data else "new"
        lap_count = len(data.laps) if data else 0
        if data and lap_count == 0:
            status = "no_laps"
        videos.append({
            "path": str(f),
            "name": f.name,
            "status": status,
            "selected": True,
            "data": data,        # ProjectData | None
            "lap_count": lap_count,
        })
    return videos


def get_detector(model_key: str) -> GateDetector:
    """Возвращает загруженный детектор, перегружает если модель изменилась."""
    if (st.session_state["detector"] is None
            or st.session_state["detector_model"] != model_key):
        det = GateDetector(model_key)
        with st.spinner(f"Загрузка модели {MODELS[model_key].name}..."):
            det.load()
        st.session_state["detector"] = det
        st.session_state["detector_model"] = model_key
    return st.session_state["detector"]


def get_osd_reader() -> OSDReader:
    if st.session_state["osd_reader"] is None:
        with st.spinner("Загрузка EasyOCR..."):
            st.session_state["osd_reader"] = OSDReader(use_gpu=True)
    return st.session_state["osd_reader"]


def frames_cache_for(video_path: str) -> dict:
    if video_path not in st.session_state["frames_cache"]:
        st.session_state["frames_cache"][video_path] = {}
    return st.session_state["frames_cache"][video_path]


def project_data_to_passes(data: ProjectData) -> list[Pass]:
    return [
        Pass(
            frame=p.frame,
            time_sec=p.time_sec,
            osd_time=p.osd_time,
            similarity=p.similarity,
            verified=p.verified,
        )
        for p in data.passes
    ]


def run_analysis(video: dict, cfg: dict, refs: list[RefImage]) -> None:
    """Запускает полный анализ одного видео и сохраняет результат."""
    video_path = video["path"]
    det = get_detector(cfg["model_key"])

    # Устанавливаем референсы
    det.set_references([r.bgr for r in refs])

    # Прогресс
    progress = st.progress(0, text=f"Анализ {video['name']}...")

    def progress_cb(cur, total):
        progress.progress(min(cur / total, 1.0), text=f"{video['name']}: {int(cur/total*100)}%")

    # Similarity
    ts, sims, fps = det.compute_similarities(
        video_path,
        sample_every=cfg["sample_every"],
        progress_cb=progress_cb,
    )
    progress.empty()

    # Кешируем для графика
    st.session_state["similarities"][video_path] = sims
    st.session_state["timestamps"][video_path] = ts

    # Находим пики
    effective_fps = fps / cfg["sample_every"]
    peak_idxs = find_passes(
        sims, ts, effective_fps,
        threshold=cfg["threshold"],
        min_lap_sec=cfg["min_lap_sec"],
        prominence=cfg["prominence"],
    )

    # OCR на кадрах-кандидатах
    osd = get_osd_reader()
    osd_region = None  # TODO: per-video region
    passes_data = []

    cap = cv2.VideoCapture(video_path)
    for idx in peak_idxs:
        frame_no = int(ts[idx] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        osd_time = osd.read_frame(frame, osd_region) if ret else None
        passes_data.append(PassData(
            frame=frame_no,
            time_sec=float(ts[idx]),
            osd_time=osd_time,
            similarity=float(sims[idx]),
            verified=None,
        ))
    cap.release()

    # Расчёт кругов
    passes_for_lap = project_data_to_passes_from_data(passes_data)
    result = analyze(passes_for_lap, best_ns=[1, 3, 5])

    laps_data = [
        LapData(
            number=l.number,
            duration_sec=l.duration_sec,
            start_sec=l.start_pass.time_sec,
            osd_start=l.start_pass.osd_time,
        )
        for l in result.laps
    ]

    best_lap_idx = None
    if result.best_lap:
        for i, l in enumerate(result.laps):
            if l.number == result.best_lap.number:
                best_lap_idx = i
                break

    best_3 = result.best_n.get(3)
    best_3_indices = [l.number - 1 for l in best_3] if best_3 else None

    data = ProjectData(
        video=video["name"],
        model=cfg["model_key"],
        analyzed_at=datetime.now().isoformat(timespec="seconds"),
        params={
            "threshold": cfg["threshold"],
            "min_lap_sec": cfg["min_lap_sec"],
            "sample_every": cfg["sample_every"],
            "prominence": cfg["prominence"],
        },
        osd_region=None,
        passes=passes_data,
        laps=laps_data,
        best_lap_idx=best_lap_idx,
        best_3_indices=best_3_indices,
        references_b64=[r.to_b64() for r in refs],
    )

    proj.save(data, video_path)
    video["data"] = data
    video["status"] = "done" if laps_data else "no_laps"
    video["lap_count"] = len(laps_data)


def project_data_to_passes_from_data(passes_data: list[PassData]) -> list[Pass]:
    return [
        Pass(
            frame=p.frame,
            time_sec=p.time_sec,
            osd_time=p.osd_time,
            similarity=p.similarity,
            verified=p.verified,
        )
        for p in passes_data
    ]


# ── Sidebar ───────────────────────────────────────────────────────────────────

cfg = render_sidebar()

# Обновляем список видео если папка изменилась
if cfg["folder"] and cfg["folder"] != st.session_state.get("_last_folder"):
    st.session_state["videos"] = scan_folder(cfg["folder"])
    st.session_state["active_idx"] = 0
    st.session_state["_last_folder"] = cfg["folder"]

videos = st.session_state["videos"]
refs: list[RefImage] = cfg["refs"]


# ── Нет папки ─────────────────────────────────────────────────────────────────

if not videos:
    st.markdown("## 🏁 FPV Lap Counter")
    st.info("👈 Выбери папку с видео в боковой панели и добавь референсные кадры ворот.")
    st.stop()


# ── Кнопка "Обработать выбранные" ────────────────────────────────────────────

selected = [v for v in videos if v["selected"]]
unprocessed = [v for v in selected if v["status"] == "new"]

top_cols = st.columns([3, 1])
with top_cols[1]:
    if unprocessed:
        if not refs:
            st.warning("⚠️ Добавь референсные кадры ворот")
        elif st.button(f"🚀 Обработать ({len(unprocessed)})", type="primary", width='stretch'):
            for v in unprocessed:
                try:
                    run_analysis(v, cfg, refs)
                except Exception as e:
                    st.error(f"{v['name']}: {e}")
            st.rerun()


# ── Основной layout: список видео + контент ───────────────────────────────────

list_col, content_col = st.columns([1, 3])

with list_col:
    # Обновляем selected из виджетов
    for i, v in enumerate(videos):
        v["selected"] = st.session_state.get(f"sel_{i}", v["selected"])

    new_active, do_reprocess = render_video_list(videos, st.session_state["active_idx"])

    if do_reprocess:
        selected = [v for v in videos if v.get("selected", False)]
        if not refs:
            st.warning("⚠️ Добавь референсные кадры ворот перед запуском")
        else:
            for v in selected:
                # Удаляем сохранённый JSON
                json_file = proj.json_path(v["path"])
                if json_file.exists():
                    json_file.unlink()
                # Сбрасываем состояние видео
                v["data"] = None
                v["status"] = "new"
                v["lap_count"] = 0
                # Очищаем кеш сигналов
                st.session_state["similarities"].pop(v["path"], None)
                st.session_state["timestamps"].pop(v["path"], None)
                st.session_state["frames_cache"].pop(v["path"], None)
            # Запускаем анализ
            for v in selected:
                try:
                    run_analysis(v, cfg, refs)
                except Exception as e:
                    st.error(f"{v['name']}: {e}")
            st.rerun()

    if new_active != st.session_state["active_idx"]:
        st.session_state["active_idx"] = new_active
        st.rerun()

# Guard: session_state may contain a stale tuple if app reloaded mid-run
_raw_idx = st.session_state["active_idx"]
if not isinstance(_raw_idx, int):
    _raw_idx = 0
    st.session_state["active_idx"] = 0
active_idx = min(_raw_idx, len(videos) - 1)
active_video = videos[active_idx]
video_path = active_video["path"]
data: ProjectData | None = active_video["data"]


# ── Контент выбранного видео ──────────────────────────────────────────────────

with content_col:
    st.markdown(f"### {active_video['name']}")

    if data is None:
        status_map = {"new": "⬜ Не обработано", "no_laps": "⚠️ Нет кругов"}
        st.info(status_map.get(active_video["status"], "Выбери видео и нажми Обработать"))
        st.stop()

    # Получаем similarity данные (из кеша или пересчитываем)
    sims = st.session_state["similarities"].get(video_path)
    ts = st.session_state["timestamps"].get(video_path)
    frames_cache = frames_cache_for(video_path)

    # Callbacks
    def on_use_as_ref(bgr_frame: np.ndarray):
        ref = from_frame(bgr_frame, label=f"{active_video['name']} кадр")
        st.session_state.setdefault("sidebar_refs", []).append(ref)
        st.toast("✅ Кадр добавлен как референс")

    def on_set_osd_region(region: OSDRegion):
        if data:
            data.osd_region = {"x": region.x, "y": region.y, "w": region.w, "h": region.h}
            proj.save(data, video_path)

    def on_verified_change(pass_idx: int, value: bool | None):
        if data and 0 <= pass_idx < len(data.passes):
            data.passes[pass_idx].verified = value
            proj.save(data, video_path)
            # пересчитываем круги
            passes_for_lap = project_data_to_passes_from_data(data.passes)
            result = analyze(passes_for_lap, best_ns=[1, 3, 5])
            data.laps = [
                LapData(l.number, l.duration_sec, l.start_pass.time_sec, l.start_pass.osd_time)
                for l in result.laps
            ]
            active_video["lap_count"] = len(data.laps)
            active_video["status"] = "done" if data.laps else "no_laps"

    # Вкладки
    tab_graph, tab_verify, tab_laps, tab_compare = st.tabs(
        ["📈 График", "✅ Верификация", "🏆 Круги", "📊 Сравнение"]
    )

    with tab_graph:
        render_graph_tab(
            video_path=video_path,
            similarities=sims,
            timestamps=ts,
            passes=data.passes,
            params=cfg,
            frames_cache=frames_cache,
            on_use_as_ref=on_use_as_ref,
            on_set_osd_region=on_set_osd_region,
        )

    with tab_verify:
        render_verify_tab(
            video_path=video_path,
            passes=data.passes,
            frames_cache=frames_cache,
            on_verified_change=on_verified_change,
            on_use_as_ref=on_use_as_ref,
        )

    with tab_laps:
        passes_for_lap = project_data_to_passes_from_data(data.passes)
        result = analyze(passes_for_lap, best_ns=[1, cfg["best_n"], 5])
        render_laps_tab(result, cfg["best_n"], active_video["name"])

    with tab_compare:
        all_results = {}
        for v in videos:
            if v["data"] and v["data"].laps:
                passes = project_data_to_passes_from_data(v["data"].passes)
                all_results[v["path"]] = analyze(passes, best_ns=[1, cfg["best_n"], 5])
        render_compare_tab(all_results, cfg["best_n"])
