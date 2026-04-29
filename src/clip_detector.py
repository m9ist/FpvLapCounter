"""
CLIP-based gate detector (open_clip_torch)
Ищет кадры, похожие на референсные фотографии ворот.
Не требует ручной настройки цветов и порогов.
"""

import numpy as np
import cv2
import torch
from typing import List
from PIL import Image


MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


class CLIPGateDetector:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.ref_embeddings = None  # (N, 512)

    def load_model(self):
        import open_clip
        if self.model is not None:
            return
        if self.verbose:
            print(f"Загрузка CLIP ({MODEL_NAME}) на {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.model = self.model.to(self.device).eval()
        if self.verbose:
            print("CLIP готов.")

    def _embed_images(self, frames: List[np.ndarray]) -> torch.Tensor:
        """BGR numpy → нормализованные CLIP-эмбеддинги."""
        imgs = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            imgs.append(self.preprocess(pil))
        batch = torch.stack(imgs).to(self.device)
        with torch.no_grad(), torch.autocast(self.device):
            feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu()

    def set_reference_frames(self, frames: List[np.ndarray]):
        if self.model is None:
            self.load_model()
        self.ref_embeddings = self._embed_images(frames)
        if self.verbose:
            print(f"Референс: {len(frames)} кадров.")

    def compute_similarities(
        self,
        video_path: str,
        sample_every: int = 3,
        progress_callback=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.ref_embeddings is None:
            raise RuntimeError("Сначала вызови set_reference_frames()")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        BATCH = 16
        timestamps, similarities = [], []
        batch_frames, batch_times = [], []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                batch_frames.append(cv2.resize(frame, (224, 224)))
                batch_times.append(frame_idx / fps)
                if len(batch_frames) >= BATCH:
                    similarities.extend(self._batch_similarity(batch_frames))
                    timestamps.extend(batch_times)
                    batch_frames, batch_times = [], []
                    if progress_callback:
                        progress_callback(frame_idx, total)
            frame_idx += 1

        if batch_frames:
            similarities.extend(self._batch_similarity(batch_frames))
            timestamps.extend(batch_times)

        cap.release()
        return np.array(timestamps), np.array(similarities), fps

    def _batch_similarity(self, frames: List[np.ndarray]) -> List[float]:
        frame_embs = self._embed_images(frames)           # (B, 512)
        sims = frame_embs @ self.ref_embeddings.T         # (B, N_ref)
        return sims.max(dim=1).values.tolist()


_detector_instance: CLIPGateDetector | None = None


def get_detector() -> CLIPGateDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = CLIPGateDetector(verbose=True)
    return _detector_instance
