"""
Надёжное извлечение кадров из HEVC/.ts.

Голый seek к произвольному кадру в long-GOP HEVC часто попадает в середину GOP
без опорных кадров и даёт битый кадр. Поэтому отматываем назад на WARMUP кадров
и декодируем последовательно — декодер успевает восстановить опорные кадры.
"""
from __future__ import annotations

import cv2
import numpy as np

WARMUP = 90          # кадров назад перед целевым (≈1.5 с @60fps / 3 с @30fps)
CACHE_LIMIT = 200    # макс. кадров в одном per-video кеше


def grab_frame(cap: cv2.VideoCapture, frame_idx: int, warmup: int = WARMUP) -> np.ndarray | None:
    """Один чистый кадр из уже открытого VideoCapture (или None, если не прочитался)."""
    start = max(0, frame_idx - warmup)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame = None
    for _ in range(start, frame_idx + 1):
        ok, fr = cap.read()
        if not ok:
            break
        frame = fr
    return frame


def grab_frames(video_path: str, frame_numbers: list[int], warmup: int = WARMUP) -> list[np.ndarray | None]:
    """Чистые кадры по списку номеров (видео открывается/закрывается один раз)."""
    cap = cv2.VideoCapture(video_path)
    try:
        return [grab_frame(cap, fno, warmup) for fno in frame_numbers]
    finally:
        cap.release()


def load_frame_cached(
    video_path: str,
    frame_idx: int,
    cache: dict,
    limit: int = CACHE_LIMIT,
    warmup: int = WARMUP,
) -> np.ndarray | None:
    """Чистый кадр с кешированием в переданном dict (FIFO-вытеснение при переполнении)."""
    if frame_idx in cache:
        return cache[frame_idx]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        frame = grab_frame(cap, frame_idx, warmup)
    finally:
        cap.release()
    if frame is None:
        return None
    if len(cache) >= limit:
        del cache[next(iter(cache))]
    cache[frame_idx] = frame
    return frame
