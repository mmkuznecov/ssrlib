"""DINOv1 vision embedder (Meta, original DINO)."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict

import torch

from ..base import BaseEmbedder

logger = logging.getLogger(__name__)


class DINOEmbedder(BaseEmbedder):
    """Original DINOv1 image embedder via torch.hub.

    Args:
        model_size: one of {"vits16", "vitb16", "vits8", "vitb8"}.
    """

    _embedder_category: ClassVar[str] = "vision"
    _embedder_modality: ClassVar[str] = "vision"
    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "framework": "torch.hub",
        "ssl_method": "DINO",
    }

    AVAILABLE_MODELS: ClassVar[Dict[str, int]] = {
        "vits16": 384,
        "vitb16": 768,
        "vits8": 384,
        "vitb8": 768,
    }

    def __init__(self, model_size: str = "vitb16", device: str = "cpu", **kwargs):
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model_size {model_size!r}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )
        super().__init__(f"DINO-{model_size}", device=device, **kwargs)
        self.model_size = model_size
        self._embedding_dim = self.AVAILABLE_MODELS[model_size]
        self._metadata.update({"model_size": model_size})

    def load_model(self) -> None:
        if self._loaded:
            return
        logger.info("Loading DINO %s via torch.hub", self.model_size)
        self.model = torch.hub.load(
            "facebookresearch/dino:main", f"dino_{self.model_size}"
        )
        self.model.eval().to(self.device)
        self._loaded = True

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            self.load_model()
        return self.model(batch.to(self.device))
