"""
OSD Timer Reader
Читает таймер полёта из OSD-оверлея FPV видео.
Таймер находится в верхнем левом углу в формате MM:SS.s (00:20.0)
"""

import re
import numpy as np
import cv2
from typing import Optional


# Область OSD-таймера относительно размера кадра
# На скриншотах таймер занимает примерно левые 20% ширины и верхние 15% высоты
OSD_ROI_X = 0.03   # от левого края
OSD_ROI_Y = 0.07   # от верхнего края
OSD_ROI_W = 0.22   # ширина области
OSD_ROI_H = 0.10   # высота области

# Паттерн таймера: MM:SS.d или MM:SS
TIMER_PATTERN = re.compile(r'(\d{1,2})[:\s](\d{2})[.\s](\d)')


def extract_osd_region(frame: np.ndarray) -> np.ndarray:
    """Вырезает область OSD таймера из кадра."""
    h, w = frame.shape[:2]
    x1 = int(w * OSD_ROI_X)
    y1 = int(h * OSD_ROI_Y)
    x2 = int(w * (OSD_ROI_X + OSD_ROI_W))
    y2 = int(h * (OSD_ROI_Y + OSD_ROI_H))
    return frame[y1:y2, x1:x2]


def preprocess_for_ocr(roi: np.ndarray) -> np.ndarray:
    """
    Препроцессинг для OCR: белый текст на тёмном фоне.
    OSD-текст обычно белый с тонкой чёрной обводкой.
    """
    # Конвертируем в grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Увеличиваем для лучшего распознавания
    scale = 3
    h, w = gray.shape
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Пороговая обработка: белый текст
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Небольшая морфология для удаления шума
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


def parse_timer_text(text: str) -> Optional[float]:
    """
    Парсит текст таймера в секунды.
    Форматы: '00:20.0', '0:20.0', '1:25.3'
    """
    text = text.strip().replace(' ', '').replace('O', '0').replace('o', '0')
    match = TIMER_PATTERN.search(text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        tenths = int(match.group(3))
        return minutes * 60 + seconds + tenths * 0.1
    return None


class OSDReader:
    """
    Читает OSD таймер с помощью EasyOCR.
    Инициализация тяжёлая (загрузка модели), поэтому делаем один раз.
    """

    def __init__(self, use_gpu: bool = True, verbose: bool = False):
        import easyocr
        self._verbose = verbose
        if verbose:
            print("Загрузка EasyOCR модели...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        if verbose:
            print("EasyOCR готов.")

    def read_timer(self, frame: np.ndarray) -> Optional[float]:
        """
        Читает таймер из кадра.
        Возвращает время в секундах или None если не удалось распознать.
        """
        roi = extract_osd_region(frame)
        processed = preprocess_for_ocr(roi)

        # EasyOCR принимает numpy array
        results = self.reader.readtext(processed, detail=0, paragraph=False)

        for text in results:
            t = parse_timer_text(text)
            if t is not None:
                if self._verbose:
                    print(f"  OCR таймер: '{text}' -> {t:.1f}s")
                return t

        # Если не нашли — пробуем весь текст объединить
        combined = ' '.join(results)
        t = parse_timer_text(combined)
        if t is not None and self._verbose:
            print(f"  OCR комбинированный: '{combined}' -> {t:.1f}s")
        return t

    def read_timer_from_path(self, image_path: str) -> Optional[float]:
        """Читает таймер из файла изображения."""
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        return self.read_timer(frame)


def test_on_image(image_path: str):
    """Быстрый тест на одном изображении."""
    reader = OSDReader(verbose=True)
    result = reader.read_timer_from_path(image_path)
    print(f"Результат: {result} секунд")

    # Показываем что вырезали
    frame = cv2.imread(image_path)
    if frame is not None:
        roi = extract_osd_region(frame)
        processed = preprocess_for_ocr(roi)
        cv2.imwrite("output/osd_roi.png", roi)
        cv2.imwrite("output/osd_processed.png", processed)
        print("Сохранены: output/osd_roi.png, output/osd_processed.png")
