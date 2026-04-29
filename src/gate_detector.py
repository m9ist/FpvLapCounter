"""
LED Gate Detector
Детектирует пролёты через стартовые ворота по яркости LED-подсветки.

Принцип:
- LED ворота очень яркие и насыщенные (синий/зелёный/красный/белый)
- При пролёте ворота занимают большую часть центра кадра
- Момент пролёта = переход от "ворота спереди" к "ворота сзади"
  → пик яркости LED в центре, затем резкий спад

Алгоритм:
1. Для каждого кадра считаем "gate score" = яркость насыщенных пикселей в центре
2. Находим пики gate score (локальные максимумы)
3. Пики > порога = пролёты через ворота
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple
from scipy.signal import find_peaks, savgol_filter


@dataclass
class GatePass:
    """Один пролёт через ворота."""
    frame_idx: int
    timestamp_sec: float       # время по видео (frame / fps)
    osd_time: float | None     # время из OSD таймера (если прочитали)
    gate_score: float          # насколько уверены (0..1)


# HSV диапазоны для LED цветов ворот
# (hue_min, hue_max, sat_min, val_min)
LED_COLORS = {
    "blue":  (100, 130, 150, 150),
    "cyan":  (85,  100, 150, 150),
    "green": (45,  85,  150, 150),
    "red1":  (0,   10,  150, 150),   # красный (0-10)
    "red2":  (170, 180, 150, 150),   # красный (170-180)
    "white": (0,   180, 0,   220),   # белый = любой hue, низкая насыщенность, высокая яркость
}


def compute_led_mask(frame: np.ndarray) -> np.ndarray:
    """
    Создаёт маску ярких LED пикселей.
    Возвращает бинарную маску (255 = LED пиксель).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    combined = np.zeros(frame.shape[:2], dtype=np.uint8)

    for name, (h_min, h_max, s_min, v_min) in LED_COLORS.items():
        if name == "white":
            mask = (s < 60) & (v > v_min)
        else:
            mask = (h >= h_min) & (h <= h_max) & (s >= s_min) & (v >= v_min)
        combined = combined | mask.astype(np.uint8) * 255

    return combined


def compute_gate_score(frame: np.ndarray, center_weight: float = 0.7) -> float:
    """
    Считает "gate score" для одного кадра.
    Высокий score = ворота занимают большую часть поля зрения.

    center_weight: насколько важна центральная область (0.5 = равномерно, 1.0 = только центр)
    """
    h, w = frame.shape[:2]
    led_mask = compute_led_mask(frame)

    # Общее количество LED пикселей
    total_led = np.sum(led_mask > 0)

    if total_led == 0:
        return 0.0

    # Центральная область (40% по обеим осям)
    cy, cx = h // 2, w // 2
    margin_y, margin_x = int(h * 0.20), int(w * 0.20)
    center_roi = led_mask[
        cy - margin_y: cy + margin_y,
        cx - margin_x: cx + margin_x
    ]
    center_led = np.sum(center_roi > 0)

    # Score = взвешенная комбинация центра и общего
    frame_area = h * w
    center_area = center_roi.size

    center_density = center_led / center_area if center_area > 0 else 0
    total_density = total_led / frame_area

    score = center_weight * center_density + (1 - center_weight) * total_density
    return float(score)


def detect_gate_passes(
    scores: np.ndarray,
    fps: float,
    min_lap_sec: float = 3.0,
    peak_prominence: float = 0.05,
    peak_height: float = 0.08,
) -> List[Tuple[int, float]]:
    """
    Находит моменты пролётов через ворота по временному ряду gate scores.

    Args:
        scores: массив gate score для каждого кадра
        fps: кадров в секунду
        min_lap_sec: минимальное время круга (для фильтрации ложных срабатываний)
        peak_prominence: минимальная выраженность пика
        peak_height: минимальная высота пика

    Returns:
        Список (frame_idx, time_sec) для каждого пролёта
    """
    if len(scores) < 10:
        return []

    # Сглаживаем для удаления шума (окно ~0.3 секунды)
    window = max(5, int(fps * 0.3) | 1)  # нечётное число
    if window % 2 == 0:
        window += 1

    smoothed = savgol_filter(scores, window_length=min(window, len(scores) - 2), polyorder=2)
    smoothed = np.clip(smoothed, 0, None)

    # Минимальное расстояние между пиками = min_lap_sec
    min_distance = int(fps * min_lap_sec)

    peaks, properties = find_peaks(
        smoothed,
        height=peak_height,
        prominence=peak_prominence,
        distance=min_distance,
    )

    result = []
    for peak in peaks:
        time_sec = peak / fps
        result.append((int(peak), float(time_sec)))

    return result


class GateDetector:
    """
    Основной детектор ворот.
    Обрабатывает видео и возвращает список пролётов.
    """

    def __init__(
        self,
        min_lap_sec: float = 3.0,
        peak_height: float = 0.08,
        peak_prominence: float = 0.04,
        sample_every_n: int = 3,    # анализируем каждый N-й кадр для скорости
        verbose: bool = False,
    ):
        self.min_lap_sec = min_lap_sec
        self.peak_height = peak_height
        self.peak_prominence = peak_prominence
        self.sample_every_n = sample_every_n
        self.verbose = verbose

    def process_video(
        self,
        video_path: str,
        osd_reader=None,
        progress_callback=None,
    ) -> Tuple[List[GatePass], np.ndarray]:
        """
        Обрабатывает видеофайл.

        Args:
            video_path: путь к видео
            osd_reader: опциональный OSDReader для чтения таймера
            progress_callback: callback(current_frame, total_frames)

        Returns:
            (список GatePass, массив gate scores)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        if self.verbose:
            print(f"Видео: {total_frames} кадров, {fps:.1f} fps, {duration:.1f}с")

        # Собираем gate scores для каждого кадра
        frame_idx = 0
        scores_sampled = []   # scores для каждого N-го кадра
        sampled_indices = []  # соответствующие индексы кадров
        osd_times = {}        # frame_idx -> osd_time

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sample_every_n == 0:
                score = compute_gate_score(frame)
                scores_sampled.append(score)
                sampled_indices.append(frame_idx)

                # Читаем OSD таймер (реже, чтобы не тормозить)
                if osd_reader and frame_idx % (self.sample_every_n * 3) == 0:
                    t = osd_reader.read_timer(frame)
                    if t is not None:
                        osd_times[frame_idx] = t

                if progress_callback and frame_idx % 30 == 0:
                    progress_callback(frame_idx, total_frames)

            frame_idx += 1

        cap.release()

        scores_arr = np.array(scores_sampled)
        effective_fps = fps / self.sample_every_n

        # Находим пики (пролёты через ворота)
        raw_passes = detect_gate_passes(
            scores_arr,
            fps=effective_fps,
            min_lap_sec=self.min_lap_sec,
            peak_height=self.peak_height,
            peak_prominence=self.peak_prominence,
        )

        # Конвертируем в GatePass объекты
        gate_passes = []
        for sampled_idx, _ in raw_passes:
            real_frame_idx = sampled_indices[sampled_idx]
            time_sec = real_frame_idx / fps

            # Ищем ближайший OSD-таймер
            osd_time = None
            if osd_times:
                closest = min(osd_times.keys(), key=lambda k: abs(k - real_frame_idx))
                if abs(closest - real_frame_idx) < fps * 2:  # не дальше 2 секунд
                    osd_time = osd_times[closest]

            gate_passes.append(GatePass(
                frame_idx=real_frame_idx,
                timestamp_sec=time_sec,
                osd_time=osd_time,
                gate_score=float(scores_arr[sampled_idx]),
            ))

        if self.verbose:
            print(f"Найдено пролётов: {len(gate_passes)}")

        # Строим полный массив scores (интерполированный до реальных кадров)
        full_scores = np.interp(
            np.arange(total_frames),
            sampled_indices,
            scores_sampled,
        )

        return gate_passes, full_scores

    def save_debug_frames(
        self,
        video_path: str,
        gate_passes: List[GatePass],
        output_dir: str = "output",
        context_sec: float = 0.5,
    ):
        """Сохраняет кадры вокруг каждого пролёта для отладки."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, gp in enumerate(gate_passes):
            target_frame = max(0, gp.frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret:
                # Рисуем отладочную инфу
                led_mask = compute_led_mask(frame)
                h, w = frame.shape[:2]
                overlay = frame.copy()
                overlay[led_mask > 0] = [0, 255, 0]
                debug = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

                label = f"Pass #{i+1} t={gp.timestamp_sec:.2f}s score={gp.gate_score:.3f}"
                cv2.putText(debug, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                path = os.path.join(output_dir, f"gate_pass_{i+1:03d}_t{gp.timestamp_sec:.1f}.jpg")
                cv2.imwrite(path, debug)

        cap.release()
        print(f"Отладочные кадры сохранены в {output_dir}/")
