"""
Реестр доступных моделей для детекции ворот.
"""
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelInfo:
    name: str           # display name
    backend: Literal["open_clip", "dinov2"]
    model_id: str       # id for open_clip or HF repo
    pretrained: str     # for open_clip only
    embed_dim: int
    description: str
    size_mb: int
    speed_note: str     # e.g. "~200 fps on RTX 3060"

MODELS: dict[str, ModelInfo] = {
    "clip_vit_b32": ModelInfo(
        name="CLIP ViT-B/32",
        backend="open_clip",
        model_id="ViT-B-32",
        pretrained="openai",
        embed_dim=512,
        description="Быстрая, хорошая точность. Рекомендуется.",
        size_mb=350,
        speed_note="~200 кадр/с",
    ),
    "clip_vit_l14": ModelInfo(
        name="CLIP ViT-L/14",
        backend="open_clip",
        model_id="ViT-L-14",
        pretrained="openai",
        embed_dim=768,
        description="Точнее, медленнее.",
        size_mb=890,
        speed_note="~80 кадр/с",
    ),
    "clip_vit_h14": ModelInfo(
        name="CLIP ViT-H/14",
        backend="open_clip",
        model_id="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        embed_dim=1024,
        description="Максимальная точность.",
        size_mb=2500,
        speed_note="~30 кадр/с",
    ),
    "dinov2_vit_b14": ModelInfo(
        name="DINOv2 ViT-B/14",
        backend="dinov2",
        model_id="facebook/dinov2-base",
        pretrained="",
        embed_dim=768,
        description="Хорош для визуального сходства.",
        size_mb=330,
        speed_note="~150 кадр/с",
    ),
    "dinov2_vit_l14": ModelInfo(
        name="DINOv2 ViT-L/14",
        backend="dinov2",
        model_id="facebook/dinov2-large",
        pretrained="",
        embed_dim=1024,
        description="Лучшее качество среди DINOv2.",
        size_mb=1100,
        speed_note="~60 кадр/с",
    ),
}

DEFAULT_MODEL = "clip_vit_b32"
