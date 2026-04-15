# FPV Lap Counter

Анализирует FPV-видео, считает времена кругов, находит лучшие 3 подряд.

## Установка

```bash
pip install -r requirements.txt
```

> RTX 3060 автоматически используется для EasyOCR (CUDA).

## Использование

### Анализ одного видео

```bash
python main.py analyze race1.mp4
```

### Анализ нескольких и сравнение

```bash
python main.py analyze race1.mp4 race2.mp4 race3.mp4
```

### Сохранить результат в JSON

```bash
python main.py analyze race1.mp4 --json results.json
```

### Калибровка чувствительности детектора

Если ворота не детектируются — используй этот режим:

```bash
python main.py calibrate race1.mp4
```

Смотри на сохранённые кадры и значение `Gate Score`. Кадры около ворот должны иметь высокий score. Затем подбери `--peak-height`.

### Тонкая настройка

```bash
python main.py analyze race1.mp4 \
  --min-lap 4.0 \         # минимальное время круга (сек)
  --peak-height 0.06 \    # снизить если ворота не детектируются
  --prominence 0.03 \     # снизить если пропускаются пролёты
  --best-n 5              # искать лучшие 5 подряд
```

### С LM Studio (опционально)

```bash
python main.py analyze race1.mp4 --lm-studio
```

Нужна vision-модель в LM Studio (llava, qwen-vl и т.п.).

## Архитектура

```
main.py                 — CLI (typer + rich)
src/
  gate_detector.py      — детекция LED ворот (OpenCV HSV)
  osd_reader.py         — OCR таймера из OSD (EasyOCR)
  lap_analyzer.py       — расчёт кругов, поиск лучших N подряд
  lm_studio_client.py   — опциональная верификация через LM Studio
```

## Как работает детекция ворот

1. Для каждого кадра считается **gate score** = плотность ярких LED-пикселей в центре кадра
2. Временной ряд score сглаживается (Savitzky-Golay фильтр)
3. Пики ряда = пролёты через ворота (`scipy.find_peaks`)
4. Время пролёта берётся из OCR OSD-таймера (формат `00:20.0` в левом верхнем углу)
