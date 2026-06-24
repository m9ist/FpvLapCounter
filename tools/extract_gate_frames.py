"""
Извлекает кадры пролётов из всех проанализированных видео для разметки/дообучения.

Идёт по всем *.fpv.json в ROOT, берёт кадры пролётов (поле frame), достаёт их из
видео и раскладывает по OUT/{true,false}/ согласно верификации.

HEVC .ts плохо перематываются в OpenCV (битые кадры), поэтому кадр берётся приёмом
«перемотка на K кадров назад + последовательная докрутка до цели» — декодер
ресинкается на ближайшем keyframe и отдаёт чистый кадр.

Запуск:  python tools/extract_gate_frames.py
"""
import glob
import json
import os

import cv2

ROOT = r"D:/fpv/!video"
OUT = r"D:/fpv/!gate_dataset"
SEEK_BACK = 90  # кадров отмотать назад перед докруткой (~1.5 c при 60 fps)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def grab(cap, fno: int):
    """Чистый кадр fno: отмотать назад и докрутить вперёд."""
    start = max(0, fno - SEEK_BACK)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame = None
    for _ in range(start, fno + 1):
        ok, fr = cap.read()
        if not ok:
            break
        frame = fr
    return frame


def imwrite(path: str, img) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def main() -> None:
    for sub in ("true", "false"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    jsons = sorted(glob.glob(os.path.join(ROOT, "**", "*.fpv.json"), recursive=True))
    n_ok = n_skip = 0
    for j in jsons:
        d = json.load(open(j, encoding="utf-8"))
        video = j[: -len(".fpv.json")] + ".ts"
        if not os.path.exists(video):
            print(f"SKIP (нет видео): {j}")
            continue
        folder = _safe(os.path.basename(os.path.dirname(j)))
        stem = _safe(os.path.basename(video)[:-3])
        passes = sorted(d.get("passes", []), key=lambda p: p.get("frame", 0))

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"SKIP (не открылось): {video}")
            continue
        for p in passes:
            fno = int(p["frame"])
            label = "true" if p.get("verified") is True else "false"
            frame = grab(cap, fno)
            if frame is None:
                n_skip += 1
                continue
            name = f"{folder}__{stem}__t{p['time_sec']:06.1f}__f{fno}.jpg"
            imwrite(os.path.join(OUT, label, name), frame)
            n_ok += 1
        cap.release()
        print(f"done: {folder}/{stem}  ({len(passes)} passes)")

    print(f"\nИзвлечено {n_ok} кадров в {OUT}  (пропущено {n_skip}).")


if __name__ == "__main__":
    main()
