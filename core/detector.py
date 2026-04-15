"""
Neural network gate detector for FPV lap timing.
Supports open_clip (CLIP variants) and DINOv2 backends.
"""
from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from core.models import MODELS, ModelInfo

logger = logging.getLogger(__name__)


class GateDetector:
    """
    Detects FPV gate passes by comparing frame embeddings against
    reference images of the gate using a vision backbone.
    """

    batch_size: int = 16

    def __init__(self, model_key: str) -> None:
        if model_key not in MODELS:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Available: {list(MODELS.keys())}"
            )
        self.model_key: str = model_key
        self.info: ModelInfo = MODELS[model_key]

        # Populated by load()
        self._model = None
        self._preprocess = None          # open_clip transform or None
        self._processor = None           # HF AutoImageProcessor or None
        self._device: str = "cpu"

        # Populated by set_references()
        self._ref_embeddings: np.ndarray | None = None  # (K, D)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the backbone and move it to GPU if available."""
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading model '%s' (backend=%s) on %s …",
            self.info.name,
            self.info.backend,
            self._device,
        )

        if self.info.backend == "open_clip":
            self._load_open_clip()
        elif self.info.backend == "dinov2":
            self._load_dinov2()
        else:
            raise RuntimeError(f"Unsupported backend: {self.info.backend}")

        logger.info("Model loaded.")

    def _load_open_clip(self) -> None:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.info.model_id,
            pretrained=self.info.pretrained,
        )
        model = model.to(self._device).eval()
        self._model = model
        self._preprocess = preprocess

    def _load_dinov2(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(self.info.model_id)
        model = AutoModel.from_pretrained(self.info.model_id)
        model = model.to(self._device).eval()
        self._model = model
        self._processor = processor

    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_images(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Convert a list of BGR frames (OpenCV format) to L2-normalized
        embeddings of shape (N, D).
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        if not frames:
            return np.empty((0, self.info.embed_dim), dtype=np.float32)

        if self.info.backend == "open_clip":
            return self._embed_open_clip(frames)
        elif self.info.backend == "dinov2":
            return self._embed_dinov2(frames)
        else:
            raise RuntimeError(f"Unsupported backend: {self.info.backend}")

    def _bgr_to_pil(self, frame: np.ndarray):
        """Convert BGR numpy array to PIL Image."""
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _embed_open_clip(self, frames: list[np.ndarray]) -> np.ndarray:
        import torch

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(frames), self.batch_size):
            batch_frames = frames[start : start + self.batch_size]
            tensors = torch.stack(
                [self._preprocess(self._bgr_to_pil(f)) for f in batch_frames]
            ).to(self._device)

            with torch.no_grad(), torch.autocast(
                device_type=self._device, enabled=(self._device == "cuda")
            ):
                feats = self._model.encode_image(tensors)

            feats = feats.float().cpu().numpy()
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            all_embeddings.append(feats / norms)

        return np.concatenate(all_embeddings, axis=0)

    def _embed_dinov2(self, frames: list[np.ndarray]) -> np.ndarray:
        import torch

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(frames), self.batch_size):
            batch_frames = frames[start : start + self.batch_size]
            pil_images = [self._bgr_to_pil(f) for f in batch_frames]

            inputs = self._processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad(), torch.autocast(
                device_type=self._device, enabled=(self._device == "cuda")
            ):
                outputs = self._model(**inputs)

            # CLS token is the first token of last_hidden_state
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            feats = cls_tokens.float().cpu().numpy()
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            all_embeddings.append(feats / norms)

        return np.concatenate(all_embeddings, axis=0)

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def set_references(self, frames: list[np.ndarray]) -> None:
        """
        Compute embeddings for reference gate images and store them.
        Subsequent calls to compute_similarities() will compare against
        the mean of these embeddings.
        """
        if not frames:
            raise ValueError("At least one reference frame is required.")
        embeddings = self.embed_images(frames)          # (K, D)
        mean = embeddings.mean(axis=0, keepdims=True)   # (1, D)
        norm = np.linalg.norm(mean, axis=1, keepdims=True) + 1e-8
        self._ref_embeddings = mean / norm              # (1, D)
        logger.info("References set: %d frame(s) used.", len(frames))

    # ------------------------------------------------------------------
    # Similarity scan
    # ------------------------------------------------------------------

    def compute_similarities(
        self,
        video_path: str,
        sample_every: int = 1,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Scan the video and return frame-level similarity to the reference gate.

        Parameters
        ----------
        video_path:   Path to the video file.
        sample_every: Process every N-th frame (1 = every frame).
        progress_cb:  Optional callback(current_frame, total_frames).

        Returns
        -------
        timestamps   : np.ndarray shape (M,)  — seconds for each sample
        similarities : np.ndarray shape (M,)  — cosine similarity in [−1, 1]
        fps          : float                  — video FPS
        """
        if self._ref_embeddings is None:
            raise RuntimeError(
                "No reference embeddings. Call set_references() first."
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        timestamps: list[float] = []
        similarities: list[float] = []
        batch_frames: list[np.ndarray] = []
        batch_ts: list[float] = []

        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every == 0:
                    batch_frames.append(frame)
                    batch_ts.append(frame_idx / fps)

                    if len(batch_frames) >= self.batch_size:
                        sims = self._score_batch(batch_frames)
                        similarities.extend(sims.tolist())
                        timestamps.extend(batch_ts)
                        batch_frames = []
                        batch_ts = []

                if progress_cb is not None:
                    progress_cb(frame_idx, total_frames)

                frame_idx += 1

            # Flush remainder
            if batch_frames:
                sims = self._score_batch(batch_frames)
                similarities.extend(sims.tolist())
                timestamps.extend(batch_ts)

        finally:
            cap.release()

        return (
            np.array(timestamps, dtype=np.float64),
            np.array(similarities, dtype=np.float32),
            fps,
        )

    def _score_batch(self, frames: list[np.ndarray]) -> np.ndarray:
        """Embed a batch and return cosine similarities against references."""
        embeddings = self.embed_images(frames)      # (N, D)
        # ref_embeddings is (1, D), already L2-normalised
        sims = (embeddings @ self._ref_embeddings.T).squeeze(axis=1)  # (N,)
        return sims


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def find_passes(
    similarities: np.ndarray,
    timestamps: np.ndarray,
    effective_fps: float,
    threshold: float = 0.6,
    min_lap_sec: float = 3.0,
    prominence: float = 0.05,
) -> np.ndarray:
    """
    Detect gate passes as peaks in the similarity signal.

    Parameters
    ----------
    similarities  : 1-D array of cosine similarities.
    timestamps    : Corresponding timestamps in seconds.
    effective_fps : Samples per second (video_fps / sample_every).
    threshold     : Minimum similarity value to consider a peak.
    min_lap_sec   : Minimum time between successive peaks (seconds).
    prominence    : scipy peak prominence threshold.

    Returns
    -------
    peak_indices : np.ndarray of integer indices into similarities/timestamps.
    """
    from scipy.signal import find_peaks, savgol_filter

    if len(similarities) == 0:
        return np.array([], dtype=int)

    # Smooth the similarity curve to reduce noise
    window = max(3, int(effective_fps * 0.5) | 1)   # odd, ~0.5 s window
    if window >= len(similarities):
        window = max(3, len(similarities) // 2 * 2 - 1)
    if window < 3:
        smoothed = similarities.copy()
    else:
        smoothed = savgol_filter(similarities, window_length=window, polyorder=2)

    min_distance = max(1, int(min_lap_sec * effective_fps))

    peak_indices, _ = find_peaks(
        smoothed,
        height=threshold,
        distance=min_distance,
        prominence=prominence,
    )

    return peak_indices
