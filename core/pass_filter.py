"""
Фильтр ложных пролётов: лёгкий классификатор поверх эмбеддингов кадра, отсеивающий
кандидаты-пики, которые не являются пролётом через стартовые ворота.

Обучается отдельно (tools/train_pass_classifier.py) на верифицированных кадрах и
сохраняется в storage/pass_classifier.joblib. На leave-one-session-out CV убирает
~90% ложных при recall ~0.93 — без тюнинга порога/времени круга.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "storage" / "pass_classifier.joblib"


def model_available() -> bool:
    return MODEL_PATH.exists()


class PassFilter:
    """Бинарный классификатор «настоящий пролёт / ложный» поверх эмбеддингов кадра."""

    def __init__(self) -> None:
        self._clf = None
        self.model_key: str = "dinov2_vit_b14"
        self.threshold: float = 0.4
        self._device = None    # задаётся в load()
        self._embedder = None  # свой GateDetector, если активный детектор не совпадает

    def load(self, device: str | None = None) -> None:
        import joblib

        bundle = joblib.load(MODEL_PATH)
        self._clf = bundle["clf"]
        self.model_key = bundle["model_key"]
        self.threshold = float(bundle.get("threshold", 0.4))
        self._device = device
        logger.info("PassFilter loaded (model_key=%s, threshold=%.2f).", self.model_key, self.threshold)

    def _get_embedder(self, detector):
        """Переиспользовать активный детектор, если он того же бэкбона; иначе свой."""
        if detector is not None and getattr(detector, "model_key", None) == self.model_key \
                and hasattr(detector, "embed_images"):
            return detector
        if self._embedder is None:
            from core.detector import GateDetector
            self._embedder = GateDetector(self.model_key)
            self._embedder.load(self._device)
        return self._embedder

    def scores(self, frames: list[np.ndarray], detector=None) -> np.ndarray:
        """P(настоящий пролёт) для каждого кадра. Битые кадры (None) → 1.0 (не отсеивать)."""
        valid = [(i, f) for i, f in enumerate(frames) if f is not None]
        proba = np.ones(len(frames), dtype=np.float32)
        if not valid:
            return proba
        emb = self._get_embedder(detector).embed_images([f for _, f in valid])
        p = self._clf.predict_proba(emb)[:, 1]
        for (i, _), pi in zip(valid, p):
            proba[i] = pi
        return proba

    def keep_mask(self, frames: list[np.ndarray], detector=None) -> tuple[np.ndarray, np.ndarray]:
        """(bool-маска оставить, вероятности)."""
        proba = self.scores(frames, detector)
        return proba >= self.threshold, proba
