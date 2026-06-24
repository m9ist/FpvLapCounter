"""
Обучает классификатор «настоящий пролёт через стартовые ворота / ложный» поверх
эмбеддингов кадра — как фильтр кандидатов-пиков существующего детектора.

Данные: D:/fpv/!gate_dataset/{true,false}/*.jpg (кадры пиков, размеченные верификацией).
Признаки: эмбеддинг кадра (DINOv2/CLIP) — тот же бэкбон, что в приложении.
Оценка: leave-one-session-out CV — честная имитация работы на НОВОМ видео.

Запуск: python tools/train_pass_classifier.py [model_key]
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from core.detector import GateDetector

DATA = r"D:/fpv/!gate_dataset"
MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "dinov2_vit_b14"
DEVICE = "cuda:1"


def session_of(fname: str) -> str:
    # "{date}__{videostem}__t..__f..jpg" -> "{date}__{videostem}"
    return "__".join(os.path.basename(fname).split("__")[:2])


def collect():
    items = []  # (path, label, session)
    for label, sub in ((1, "true"), (0, "false")):
        for f in sorted(glob.glob(os.path.join(DATA, sub, "*.jpg"))):
            items.append((f, label, session_of(f)))
    return items


def embed_all(items):
    det = GateDetector(MODEL_KEY)
    det.load(DEVICE)
    frames = [cv2.imread(p) for p, _, _ in items]
    X = det.embed_images(frames)  # (N, D), L2-normalized
    y = np.array([lab for _, lab, _ in items])
    groups = np.array([s for _, _, s in items])
    return X, y, groups


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut

    items = collect()
    print(f"{MODEL_KEY}: {len(items)} frames "
          f"({sum(l for _,l,_ in items)} true / {sum(1-l for _,l,_ in items)} false), "
          f"{len(set(s for *_,s in items))} sessions")
    X, y, groups = embed_all(items)

    logo = LeaveOneGroupOut()
    proba = np.zeros(len(y))
    for tr, te in logo.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1]

    # Practical metric: classifier as a FILTER on candidate peaks.
    print("\nLOSO CV — фильтр кандидатов (порог 0.5):")
    for thr in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        pred = proba >= thr
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        n_sess = len(set(groups))
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        print(f"  thr={thr}: keep {tp+fp:3d}  TP={tp} FP={fp} FN={fn}  "
              f"prec={prec:.2f} rec={rec:.2f}  ~FP/видео={fp/n_sess:.1f} (было ~{int((y==0).sum())//n_sess})")

    # Baseline = keep ALL candidates (current behaviour)
    print(f"\nБаза (без фильтра): keep {len(y)} = TP={int(y.sum())} FP={int((1-y).sum())} "
          f"(~{int((1-y).sum())/len(set(groups)):.1f} FP/видео)")

    # Fit final model on ALL data and save.
    import joblib
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X, y)
    out = os.path.join(os.path.dirname(__file__), os.pardir, "storage", "pass_classifier.joblib")
    out = os.path.abspath(out)
    joblib.dump({"clf": clf, "model_key": MODEL_KEY, "threshold": 0.4,
                 "embed_dim": int(X.shape[1]), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}, out)
    print(f"\nСохранена финальная модель: {out}  (model_key={MODEL_KEY}, threshold=0.4)")


if __name__ == "__main__":
    main()
