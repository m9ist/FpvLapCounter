"""
FPV Lap Counter
Главный файл приложения Streamlit.
"""

# ── Logging must be set up before any other import ───────────────────────────
from utils.logger import setup as _setup_logging
_setup_logging()

import ctypes
import shutil
import threading

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

from core.models import MODELS, DEFAULT_MODEL
from core.detector import GateDetector, find_passes
from core.lap_analyzer import Pass, analyze
from storage import project as proj
from storage.project import ProjectData, PassData, LapData
from storage.references import RefImage, from_frame, save_ref_to_history
from storage import model_stats
from ui.sidebar import render_sidebar, pick_folder_dialog
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

class _StopAnalysis(Exception):
    """Raised inside run_analysis when stop_event is set."""


# ── Windows sleep prevention ──────────────────────────────────────────────────

_ES_CONTINUOUS      = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001

def _prevent_sleep() -> None:
    """Запретить Windows засыпать (удерживает таймер системного бодрствования)."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        )
    except Exception:
        pass

def _restore_sleep() -> None:
    """Вернуть нормальный режим сна после завершения работы."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    except Exception:
        pass


_THREAD_PRIORITY_BELOW_NORMAL = -1

def _lower_current_thread_priority() -> None:
    """Понизить приоритет текущего потока (для фонового анализа).

    GetCurrentThread() возвращает псевдо-хэндл вызывающего потока, поэтому
    приоритет понижается только у рабочего потока, а UI остаётся отзывчивым.
    """
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentThread.restype = ctypes.c_void_p
        k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        k32.SetThreadPriority(k32.GetCurrentThread(), _THREAD_PRIORITY_BELOW_NORMAL)
    except Exception:
        pass


def _init_state():
    defaults = {
        "videos": [],           # list of video dicts
        "active_idx": 0,
        "detector": None,       # loaded GateDetector
        "detector_model": None, # which model is loaded
        "detector_device": None, # on which device it is loaded
        "pass_filter": None,    # loaded PassFilter (или False, если модели нет)
        "pass_filter_device": None, # на каком устройстве загружен фильтр
        "frames_cache": {},     # video_path -> {frame_idx: bgr}
        "similarities": {},     # video_path -> np.ndarray
        "timestamps": {},       # video_path -> np.ndarray
        # ── Analysis state ──────────────────────────────────────────────
        # shared is a plain Python dict written by the worker thread
        # (thread cannot access st.session_state without ScriptRunCtx).
        "analysis_shared": None,    # dict | None — set by run_batch
        "analysis_stop": None,      # threading.Event | None
        # ── Club post-session workflow ───────────────────────────────────
        "club_step": None,          # None | "preview" | "transfer"
        "club_source": None,        # source folder path (str)
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


def get_detector(model_key: str, device: str | None = None) -> GateDetector:
    """Возвращает загруженный детектор, перегружает если модель или устройство изменились."""
    if (st.session_state["detector"] is None
            or st.session_state["detector_model"] != model_key
            or st.session_state["detector_device"] != device):
        det = GateDetector(model_key)
        with st.spinner(f"Загрузка модели {MODELS[model_key].name} на {device}..."):
            det.load(device)
        st.session_state["detector"] = det
        st.session_state["detector_model"] = model_key
        st.session_state["detector_device"] = device
    return st.session_state["detector"]


def get_pass_filter(device: str | None = None):
    """Загруженный фильтр ложных пролётов, или None если модель не обучена."""
    from core import pass_filter as pf
    if not pf.model_available():
        return None
    if (not st.session_state.get("pass_filter")
            or st.session_state.get("pass_filter_device") != device):
        flt = pf.PassFilter()
        with st.spinner("Загрузка фильтра ложных пролётов…"):
            flt.load(device)
        st.session_state["pass_filter"] = flt
        st.session_state["pass_filter_device"] = device
    return st.session_state["pass_filter"]


def frames_cache_for(video_path: str) -> dict:
    if video_path not in st.session_state["frames_cache"]:
        st.session_state["frames_cache"][video_path] = {}
    return st.session_state["frames_cache"][video_path]


def _video_duration_sec(path: Path) -> float:
    """Длительность видео в секундах. -1.0 если не удалось определить."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return -1.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0 or frames <= 0:
            return -1.0
        return frames / fps
    finally:
        cap.release()


_MIN_CLUB_VIDEO_SEC = 30.0  # видео короче — мусор, удаляем при переносе из клуба


def run_club_transfer(source: str, dest: str, cfg: dict, refs: list[RefImage],
                      progress=None) -> None:
    """Переносит видео из source в dest, удаляет картинки, запускает анализ.

    Видео короче _MIN_CLUB_VIDEO_SEC секунд удаляются, а не переносятся.
    progress : callable(done:int, total:int, label:str) | None — отчёт о прогрессе.
    Вызывается из основного потока после подтверждения пользователем.
    Блокировка сна снимается автоматически по завершении фонового потока.
    """
    _prevent_sleep()

    src_path = Path(source)
    dst_path = Path(dest)
    dst_path.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png"}
    ts_exts  = {".ts", ".mp4", ".avi", ".mov", ".mkv"}

    all_files = [f for f in src_path.rglob("*") if f.is_file()]
    video_files = [f for f in all_files if f.suffix.lower() in ts_exts]
    image_files = [f for f in all_files if f.suffix.lower() in img_exts]
    total = len(video_files) + len(image_files)
    done = 0

    for f in video_files:
        if progress:
            progress(done, total, f"Видео: {f.name}")
        # Короткие ролики (< 30 с) — мусор, удаляем не перенося.
        dur = _video_duration_sec(f)
        if 0 <= dur < _MIN_CLUB_VIDEO_SEC:
            try:
                f.unlink()
            except Exception:
                pass
        else:
            target = dst_path / f.name
            # Не перезаписывать если уже есть
            if not target.exists():
                shutil.move(str(f), str(target))
        done += 1

    for f in image_files:
        if progress:
            progress(done, total, f"Удаляю картинку: {f.name}")
        try:
            f.unlink()
        except Exception:
            pass
        done += 1

    if progress:
        progress(total, total, "Готово")

    # Обновляем список видео
    st.session_state["videos"] = scan_folder(dest)
    st.session_state["active_idx"] = 0
    st.session_state["_last_folder"] = dest
    st.session_state["club_step"] = None
    st.session_state["club_source"] = None

    # Запускаем анализ (sleep вернётся когда поток закончит)
    unprocessed = [v for v in st.session_state["videos"] if v["status"] == "new"]
    if unprocessed and refs:
        run_batch(unprocessed, cfg, refs, restore_sleep=True)
    else:
        _restore_sleep()


def run_batch(
    videos_to_process: list[dict],
    cfg: dict,
    refs: list[RefImage],
    restore_sleep: bool = False,
) -> None:
    """Запускает анализ списка видео в фоновом потоке.

    Все обращения к session_state делаются в основном потоке до старта потока.
    Рабочий поток получает Python-объекты напрямую через замыкание.
    """
    total = len(videos_to_process)
    if total == 0:
        return

    # Прогрев тяжёлых объектов в основном потоке (со спиннерами),
    # захватываем прямые ссылки — поток не должен трогать session_state.
    det = get_detector(cfg["model_key"], cfg.get("device"))
    pass_filter = get_pass_filter(cfg.get("device")) if cfg.get("smart_filter", True) else None
    similarities_dict = st.session_state["similarities"]
    timestamps_dict = st.session_state["timestamps"]

    stop_event = threading.Event()
    st.session_state["analysis_stop"] = stop_event

    # Plain dict — рабочий поток пишет сюда; ScriptRunCtx не нужен.
    shared: dict = {
        "running": True,
        "status": f"Запуск анализа ({total} файлов)…",
        "progress": 0.0,
        "inner": 0.0,
        "errors": [],
    }
    st.session_state["analysis_shared"] = shared

    def _worker():
        _lower_current_thread_priority()
        for i, v in enumerate(videos_to_process):
            if stop_event.is_set():
                shared["status"] = f"⏹ Остановлено на файле {i + 1} из {total}"
                break
            shared["status"] = f"Файл {i + 1} из {total}: {v['name']}"
            shared["progress"] = i / total
            shared["inner"] = 0.0
            try:
                run_analysis(
                    v, cfg, refs, det,
                    similarities_dict, timestamps_dict,
                    stop_event=stop_event,
                    on_inner_progress=lambda p: shared.__setitem__("inner", p),
                    pass_filter=pass_filter,
                )
            except _StopAnalysis:
                shared["status"] = f"⏹ Остановлено: {v['name']}"
                break
            except Exception as exc:
                shared["errors"] = shared["errors"] + [f"{v['name']}: {exc}"]
        else:
            shared["progress"] = 1.0
            shared["status"] = f"✅ Готово: обработано {total} из {total} файлов"

        shared["running"] = False
        if restore_sleep:
            _restore_sleep()

    t = threading.Thread(target=_worker, daemon=True, name="fpv-analysis")
    t.start()


def run_analysis(
    video: dict,
    cfg: dict,
    refs: list[RefImage],
    det,                              # GateDetector — передаётся напрямую
    similarities_dict: dict,          # ссылка на st.session_state["similarities"]
    timestamps_dict: dict,            # ссылка на st.session_state["timestamps"]
    stop_event: threading.Event | None = None,
    on_inner_progress=None,           # callable(float 0‥1) | None
    pass_filter=None,                 # PassFilter | None — отсев ложных пролётов
) -> None:
    """Запускает полный анализ одного видео и сохраняет результат.

    Не обращается к st.session_state — безопасен для вызова из потока.
    """
    video_path = video["path"]

    # Устанавливаем референсы
    det.set_references([r.bgr for r in refs])

    def progress_cb(cur, total):
        if stop_event and stop_event.is_set():
            raise _StopAnalysis()
        pct = min(cur / total, 1.0)
        if on_inner_progress:
            on_inner_progress(pct)

    # Similarity
    ts, sims, fps = det.compute_similarities(
        video_path,
        sample_every=cfg["sample_every"],
        progress_cb=progress_cb,
    )

    # Кешируем для графика (мутируем dict по ссылке — thread-safe для CPython)
    similarities_dict[video_path] = sims
    timestamps_dict[video_path] = ts

    # Находим пики
    effective_fps = fps / cfg["sample_every"]
    peak_idxs = find_passes(
        sims, ts, effective_fps,
        threshold=cfg["threshold"],
        min_lap_sec=cfg["min_lap_sec"],
        prominence=cfg["prominence"],
    )

    # Отсев ложных пролётов обученным классификатором (если включён и доступен).
    if pass_filter is not None and len(peak_idxs):
        from core.frames import grab_frames
        peak_frames_no = [int(ts[i] * fps) for i in peak_idxs]
        cand = grab_frames(video_path, peak_frames_no)
        keep, _ = pass_filter.keep_mask(cand, detector=det)
        peak_idxs = peak_idxs[keep]

    # Кандидаты-пролёты (время круга считаем по видео-таймкоду)
    passes_data = [
        PassData(
            frame=int(ts[idx] * fps),
            time_sec=float(ts[idx]),
            similarity=float(sims[idx]),
            verified=None,
        )
        for idx in peak_idxs
    ]

    # Расчёт кругов
    passes_for_lap = project_data_to_passes_from_data(passes_data)
    result = analyze(passes_for_lap, best_ns=[1, 3, 5])

    laps_data = [
        LapData(
            number=l.number,
            duration_sec=l.duration_sec,
            start_sec=l.start_pass.time_sec,
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
            "smart_filter": bool(pass_filter is not None),
        },
        passes=passes_data,
        laps=laps_data,
        best_lap_idx=best_lap_idx,
        best_3_indices=best_3_indices,
        references_b64=[r.to_b64() for r in refs],
    )

    proj.save(data, video_path)
    model_stats.increment(cfg["model_key"])
    video["data"] = data
    video["status"] = "done" if laps_data else "no_laps"
    video["lap_count"] = len(laps_data)


def project_data_to_passes_from_data(passes_data: list[PassData]) -> list[Pass]:
    return [
        Pass(
            frame=p.frame,
            time_sec=p.time_sec,
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

# Идёт ли сейчас фоновый анализ (блокирует запуск новых батчей)
_shared: dict | None = st.session_state.get("analysis_shared")
_analysis_running = bool(_shared and _shared.get("running"))


# ── Кнопка «После клуба» (видна всегда, даже в пустой папке) ─────────────────

_club_cols = st.columns([4, 1])
with _club_cols[1]:
    if st.button("📦 После клуба", width='stretch',
                 help="Перенести файлы с флешки и запустить анализ"):
        chosen = pick_folder_dialog()
        if chosen:
            st.session_state["club_source"] = chosen
            st.session_state["club_step"] = "preview"
            st.rerun()

if st.session_state.get("club_step") == "preview":
    source   = st.session_state.get("club_source", "")
    dest     = cfg["folder"] or ""
    src_path = Path(source) if source else None

    video_exts = {".ts", ".mp4", ".avi", ".mov", ".mkv"}
    ts_files = sorted(
        (f for f in src_path.rglob("*") if f.is_file() and f.suffix.lower() in video_exts),
        key=lambda f: f.name,
    ) if src_path else []
    img_files = sorted(
        (f for f in src_path.rglob("*")
         if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda f: f.name,
    ) if src_path else []

    with st.container(border=True):
        st.markdown("### 📦 Перенос файлов после клуба")

        info_cols = st.columns(2)
        with info_cols[0]:
            st.markdown(f"**Откуда:** `{source or '—'}`")
            st.markdown(f"**Куда:** `{dest or '—'}`")
        with info_cols[1]:
            st.metric("Видеофайлов", len(ts_files))
            st.metric("Удалить картинок", len(img_files))

        if ts_files:
            with st.expander(f"Список файлов ({len(ts_files)})"):
                for f in ts_files:
                    st.text(f.name)

        if not dest:
            st.warning("⚠️ Укажи папку назначения в боковой панели")
        if not refs:
            st.warning("⚠️ Добавь референсные кадры ворот в боковой панели")
        if _analysis_running:
            st.warning("⏳ Идёт обработка — дождись её окончания")

        btn_cols = st.columns(2)
        with btn_cols[0]:
            if st.button("❌ Отмена", key="club_cancel", width='stretch'):
                st.session_state["club_step"] = None
                st.session_state["club_source"] = None
                st.rerun()
        with btn_cols[1]:
            ok = bool(dest and ts_files and refs) and not _analysis_running
            if st.button("✅ Подтвердить", key="club_confirm",
                         type="primary", width='stretch', disabled=not ok):
                st.session_state["club_step"] = "transfer"
                st.rerun()

# ── Перенос файлов после клуба (с прогрессом, затем запуск анализа) ──────────

if st.session_state.get("club_step") == "transfer":
    source = st.session_state.get("club_source", "")
    dest   = cfg["folder"] or ""
    with st.container(border=True):
        st.markdown("### 📦 Перенос файлов после клуба")
        bar = st.progress(0.0, text="Подготовка…")

        def _club_progress(done: int, total: int, label: str) -> None:
            frac = done / total if total else 1.0
            bar.progress(frac, text=f"{done}/{total} · {label}")

        run_club_transfer(source, dest, cfg, refs, progress=_club_progress)
    # Перенос завершён, анализ запущен в фоне — обновляем интерфейс
    st.rerun()

# ── Нет видео в папке ─────────────────────────────────────────────────────────

if not videos:
    st.markdown("## 🏁 FPV Lap Counter")
    if not cfg["folder"]:
        st.info("👈 Выбери папку с видео в боковой панели и добавь референсные кадры ворот.")
    else:
        st.info("Папка пуста. Добавь видеофайлы или воспользуйся кнопкой «📦 После клуба».")
    st.stop()


# ── Прогресс анализа (фоновый поток + авто-обновление через фрагмент) ────────
# Фрагмент сам перерисовывается каждые 0.5 с, не блокируя остальной интерфейс,
# поэтому можно валидировать кадры готовых видео пока идёт обработка.

@st.fragment(run_every=0.5)
def _analysis_progress_fragment():
    shared = st.session_state.get("analysis_shared")
    if not shared:
        return
    if not shared.get("running"):
        # Анализ завершился — сохраняем ошибки, обновляем весь интерфейс
        # (статусы видео) и прекращаем авто-обновление фрагмента.
        st.session_state["_analysis_errors"] = shared.get("errors", [])
        st.session_state["analysis_shared"] = None
        st.rerun(scope="app")
        return
    with st.container(border=True):
        col_info, col_stop = st.columns([5, 1])
        with col_info:
            st.markdown(f"**{shared.get('status', '')}**")
            st.progress(shared.get("progress", 0.0), text="Файлы")
            st.progress(shared.get("inner", 0.0), text="Кадры")
        with col_stop:
            if st.button("🛑 Стоп", key="stop_analysis_btn", width='stretch'):
                stop_ev = st.session_state.get("analysis_stop")
                if stop_ev:
                    stop_ev.set()
        for err in shared.get("errors", []):
            st.error(err)

if _analysis_running:
    _analysis_progress_fragment()

# Ошибки завершившегося анализа показываем один раз
for _err in st.session_state.pop("_analysis_errors", []):
    st.error(_err)


# ── Кнопка "Обработать выбранные" ────────────────────────────────────────────

selected = [v for v in videos if v["selected"]]
unprocessed = [v for v in selected if v["status"] == "new"]

top_cols = st.columns([3, 1])
with top_cols[1]:
    if _analysis_running:
        st.button("⏳ Идёт обработка…", disabled=True, width='stretch')
    elif unprocessed:
        if not refs:
            st.warning("⚠️ Добавь референсные кадры ворот")
        elif st.button(f"🚀 Обработать ({len(unprocessed)})", type="primary", width='stretch'):
            run_batch(unprocessed, cfg, refs)
            st.rerun()


# ── Основной layout: список видео + контент ───────────────────────────────────

_TABS = ["📈 График", "✅ Верификация", "🏆 Круги", "📊 Сравнение"]

list_col, content_col = st.columns([1, 3])


def _video_list_body():
    # Обновляем selected из виджетов
    for i, v in enumerate(videos):
        v["selected"] = st.session_state.get(f"sel_{i}", v["selected"])

    new_active, do_reprocess = render_video_list(videos, st.session_state["active_idx"])

    if do_reprocess and _analysis_running:
        st.warning("⏳ Дождись окончания текущей обработки")
    elif do_reprocess:
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
                st.session_state.get("_verify_thumb_cache", {}).pop(v["path"], None)
            # Запускаем анализ (поток стартует, polling loop подхватит)
            run_batch(selected, cfg, refs)
            st.rerun()  # войти в polling loop (полный rerun и из фрагмента)

    if new_active != st.session_state["active_idx"]:
        st.session_state["active_idx"] = new_active
        st.rerun()


with list_col:
    # На вкладке верификации клики ✅/❌ перерисовывают только фрагмент карточки,
    # список видео сам о них не узнаёт. Поэтому здесь список — фрагмент с опросом
    # раз в секунду (только на этой вкладке): бейдж активного видео (круги, ❓N)
    # обновляется по мере верификации. Данные in-memory, рендер копеечный.
    _poll = 1.0 if st.session_state.get("main_tab") == _TABS[1] else None
    st.fragment(_video_list_body, run_every=_poll)()

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
        save_ref_to_history(ref)
        st.toast("✅ Кадр добавлен как референс")

    def on_verified_change(pass_idx: int, value: bool | None):
        if data and 0 <= pass_idx < len(data.passes):
            data.passes[pass_idx].verified = value
            # Пересчитываем круги ДО сохранения, чтобы JSON содержал актуальный lap_count
            passes_for_lap = project_data_to_passes_from_data(data.passes)
            result = analyze(passes_for_lap, best_ns=[1, 3, 5])
            data.laps = [
                LapData(l.number, l.duration_sec, l.start_pass.time_sec)
                for l in result.laps
            ]
            proj.save(data, video_path)
            active_video["lap_count"] = len(data.laps)
            active_video["status"] = "done" if data.laps else "no_laps"

    def on_smart_filter():
        """Прогнать классификатор по текущим кандидатам и пометить ✅/❌."""
        from core.frames import grab_frames
        flt = get_pass_filter(cfg.get("device"))
        if flt is None or not data or not data.passes:
            return None
        frames = grab_frames(video_path, [p.frame for p in data.passes])
        mask, _ = flt.keep_mask(frames)  # свой DINOv2-эмбеддер, независимо от выбранной модели
        # Помечаем только фейки (❌); остальных не трогаем — подтверждаешь сам.
        for i, keep in enumerate(mask):
            if not keep:
                on_verified_change(i, False)
        return int((~mask).sum())

    # Вкладки: st.tabs выполняет тело ВСЕХ вкладок при каждом rerun (клик по
    # любому виджету пересобирал график Plotly и analyze() по всем видео).
    # segmented_control + if рендерит только активную; переключение вкладки —
    # обычный виджет-клик, т.е. полный rerun, поэтому данные всегда свежие.
    active_tab = st.segmented_control(
        "Вкладка", _TABS, default=_TABS[0],
        key="main_tab", label_visibility="collapsed",
    ) or _TABS[0]  # клик по выбранной вкладке снимает выбор → считаем «График»

    if active_tab == _TABS[0]:
        render_graph_tab(
            video_path=video_path,
            similarities=sims,
            timestamps=ts,
            passes=data.passes,
            params=cfg,
            frames_cache=frames_cache,
            on_use_as_ref=on_use_as_ref,
        )

    elif active_tab == _TABS[1]:
        from core.pass_filter import model_available as _pf_available
        render_verify_tab(
            video_path=video_path,
            passes=data.passes,
            frames_cache=frames_cache,
            on_verified_change=on_verified_change,
            on_use_as_ref=on_use_as_ref,
            on_smart_filter=on_smart_filter if _pf_available() else None,
        )

    elif active_tab == _TABS[2]:
        passes_for_lap = project_data_to_passes_from_data(data.passes)
        result = analyze(passes_for_lap, best_ns=[1, cfg["best_n"], 5])
        render_laps_tab(result, cfg["best_n"], active_video["name"])

    elif active_tab == _TABS[3]:
        all_results = {}
        no_lap_paths = []
        keep_map = {}
        for v in videos:
            if v["data"]:
                keep_map[v["path"]] = v["data"].keep
            if v["data"] and v["data"].laps:
                passes = project_data_to_passes_from_data(v["data"].passes)
                all_results[v["path"]] = analyze(passes, best_ns=[1, cfg["best_n"], 5])
            elif v["status"] == "no_laps":
                no_lap_paths.append(v["path"])

        def on_keep_change(path: str, keep: bool) -> None:
            for v in videos:
                if v["path"] == path and v["data"]:
                    v["data"].keep = keep
                    proj.save(v["data"], path)
                    break

        def on_delete_videos(paths: list[str]) -> None:
            for path in paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
                json_f = proj.json_path(path)
                if json_f.exists():
                    json_f.unlink()
                st.session_state["similarities"].pop(path, None)
                st.session_state["timestamps"].pop(path, None)
                st.session_state["frames_cache"].pop(path, None)
            if cfg["folder"]:
                st.session_state["videos"] = scan_folder(cfg["folder"])
                st.session_state["active_idx"] = 0

        render_compare_tab(all_results, cfg["best_n"],
                           on_delete_videos=on_delete_videos,
                           no_lap_paths=no_lap_paths,
                           keep_map=keep_map,
                           on_keep_change=on_keep_change)
