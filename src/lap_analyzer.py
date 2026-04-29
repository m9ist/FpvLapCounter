"""
Lap Analyzer
Вычисляет времена кругов и находит лучшие 3 подряд.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from src.gate_detector import GatePass


@dataclass
class Lap:
    """Один круг."""
    lap_number: int
    start_pass: GatePass
    end_pass: GatePass
    duration_sec: float

    @property
    def duration_str(self) -> str:
        """Время в формате MM:SS.ss"""
        total = self.duration_sec
        minutes = int(total // 60)
        seconds = total % 60
        return f"{minutes:02d}:{seconds:05.2f}"


@dataclass
class BestConsecutive:
    """Лучшие N подряд идущих кругов."""
    laps: List[Lap]
    total_duration_sec: float
    average_lap_sec: float

    @property
    def total_str(self) -> str:
        total = self.total_duration_sec
        minutes = int(total // 60)
        seconds = total % 60
        return f"{minutes:02d}:{seconds:05.2f}"

    @property
    def average_str(self) -> str:
        total = self.average_lap_sec
        minutes = int(total // 60)
        seconds = total % 60
        return f"{minutes:02d}:{seconds:05.2f}"


@dataclass
class VideoAnalysis:
    """Результат анализа одного видеофайла."""
    video_path: str
    gate_passes: List[GatePass]
    laps: List[Lap]
    best_lap: Optional[Lap]
    best_3_consecutive: Optional[BestConsecutive]
    best_n_consecutive: dict = field(default_factory=dict)  # n -> BestConsecutive


def compute_laps(gate_passes: List[GatePass], use_osd: bool = True) -> List[Lap]:
    """
    Вычисляет времена кругов из списка пролётов через ворота.

    Args:
        gate_passes: список пролётов, отсортированный по времени
        use_osd: использовать OSD таймер если доступен (точнее, чем frame timestamp)

    Returns:
        Список кругов
    """
    if len(gate_passes) < 2:
        return []

    # Сортируем по времени
    sorted_passes = sorted(gate_passes, key=lambda p: p.timestamp_sec)

    laps = []
    for i in range(1, len(sorted_passes)):
        start = sorted_passes[i - 1]
        end = sorted_passes[i]

        # Предпочитаем OSD время (точнее), fallback на frame timestamp
        if use_osd and start.osd_time is not None and end.osd_time is not None:
            duration = end.osd_time - start.osd_time
        else:
            duration = end.timestamp_sec - start.timestamp_sec

        # Фильтруем аномалии (отрицательные или слишком большие)
        if duration <= 0:
            continue

        laps.append(Lap(
            lap_number=i,
            start_pass=start,
            end_pass=end,
            duration_sec=duration,
        ))

    # Перенумеруем
    for i, lap in enumerate(laps, 1):
        lap.lap_number = i

    return laps


def find_best_consecutive(laps: List[Lap], n: int = 3) -> Optional[BestConsecutive]:
    """
    Находит лучшие N подряд идущих кругов (с минимальным суммарным временем).

    Args:
        laps: список кругов
        n: количество подряд идущих кругов

    Returns:
        BestConsecutive или None если кругов меньше N
    """
    if len(laps) < n:
        return None

    best_total = float('inf')
    best_start_idx = 0

    for i in range(len(laps) - n + 1):
        window = laps[i:i + n]
        total = sum(lap.duration_sec for lap in window)
        if total < best_total:
            best_total = total
            best_start_idx = i

    best_laps = laps[best_start_idx:best_start_idx + n]
    return BestConsecutive(
        laps=best_laps,
        total_duration_sec=best_total,
        average_lap_sec=best_total / n,
    )


def analyze_video(
    video_path: str,
    gate_passes: List[GatePass],
    use_osd: bool = True,
    max_consecutive: int = 5,
) -> VideoAnalysis:
    """
    Полный анализ видео: считает круги и находит лучшие N подряд.

    Args:
        video_path: путь к видео
        gate_passes: обнаруженные пролёты через ворота
        use_osd: использовать OSD таймер
        max_consecutive: считать best N для N в диапазоне 1..max_consecutive

    Returns:
        VideoAnalysis с результатами
    """
    laps = compute_laps(gate_passes, use_osd=use_osd)

    best_lap = min(laps, key=lambda l: l.duration_sec) if laps else None

    best_n = {}
    for n in range(1, max_consecutive + 1):
        result = find_best_consecutive(laps, n)
        if result:
            best_n[n] = result

    return VideoAnalysis(
        video_path=video_path,
        gate_passes=gate_passes,
        laps=laps,
        best_lap=best_lap,
        best_3_consecutive=best_n.get(3),
        best_n_consecutive=best_n,
    )


def compare_analyses(analyses: List[VideoAnalysis]) -> dict:
    """
    Сравнивает результаты нескольких видео.
    Возвращает общий best lap, best 3 consecutive и рейтинг.
    """
    all_laps = []
    for analysis in analyses:
        for lap in analysis.laps:
            all_laps.append((analysis.video_path, lap))

    if not all_laps:
        return {}

    # Глобальный лучший круг
    global_best_lap = min(all_laps, key=lambda x: x[1].duration_sec)

    # Лучшие 3 подряд по каждому видео
    best_3_per_video = [
        (a.video_path, a.best_3_consecutive)
        for a in analyses
        if a.best_3_consecutive is not None
    ]
    global_best_3 = min(
        best_3_per_video,
        key=lambda x: x[1].total_duration_sec,
        default=None,
    )

    return {
        "global_best_lap": global_best_lap,
        "global_best_3_consecutive": global_best_3,
        "rankings": sorted(analyses, key=lambda a: (
            a.best_3_consecutive.total_duration_sec
            if a.best_3_consecutive else float('inf')
        )),
    }
