"""Embedders package for ssrlib."""

from __future__ import annotations

from typing import Dict, List, Type

from .base import BaseEmbedder
from .cv import CLIPEmbedder, DINOEmbedder, DINOv2Embedder, VICRegEmbedder
from .mock import IdentityEmbedder
from .nlp import BERTBaseEmbedder, BERTEmbedder, E5Embedder, ModernBERTEmbedder

_EMBEDDER_CLASSES: List[Type[BaseEmbedder]] = [
    IdentityEmbedder,
    DINOv2Embedder,
    DINOEmbedder,
    CLIPEmbedder,
    VICRegEmbedder,
    BERTEmbedder,
    BERTBaseEmbedder,
    E5Embedder,
    ModernBERTEmbedder,
]

_REGISTRY: Dict[str, Type[BaseEmbedder]] = {
    cls.__name__: cls for cls in _EMBEDDER_CLASSES
}


def list_embedders() -> List[str]:
    return list(_REGISTRY.keys())


def get_available_embedders() -> Dict[str, Type[BaseEmbedder]]:
    return dict(_REGISTRY)


def create_embedder(name: str, **kwargs) -> BaseEmbedder:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown embedder '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = [
    "BaseEmbedder",
    "IdentityEmbedder",
    "DINOv2Embedder",
    "DINOEmbedder",
    "CLIPEmbedder",
    "VICRegEmbedder",
    "BERTEmbedder",
    "BERTBaseEmbedder",
    "E5Embedder",
    "ModernBERTEmbedder",
    "list_embedders",
    "get_available_embedders",
    "create_embedder",
]
