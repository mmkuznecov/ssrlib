"""Computer-vision embedders."""

from .clip import CLIPEmbedder
from .dino import DINOEmbedder
from .dinov2 import DINOv2Embedder
from .vicreg import VICRegEmbedder

__all__ = [
    "CLIPEmbedder",
    "DINOEmbedder",
    "DINOv2Embedder",
    "VICRegEmbedder",
]
