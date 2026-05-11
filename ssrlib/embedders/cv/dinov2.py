"""DINOv2 vision embedder (Meta)."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict

import torch

from ..base import BaseEmbedder

logger = logging.getLogger(__name__)


class DINOv2Embedder(BaseEmbedder):
    """DINOv2 image embedder loaded via torch.hub.

    Args:
        model_size: one of {"vits14", "vitb14", "vitl14", "vitg14"}.
        device: torch device string.
    """

    _embedder_category: ClassVar[str] = "vision"
    _embedder_modality: ClassVar[str] = "vision"
    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "framework": "torch.hub",
        "ssl_method": "DINOv2",
    }

    AVAILABLE_MODELS: ClassVar[Dict[str, int]] = {
        "vits14": 384,
        "vitb14": 768,
        "vitl14": 1024,
        "vitg14": 1536,
    }

    def __init__(self, model_size: str = "vitb14", device: str = "cpu", **kwargs):
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model_size {model_size!r}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )
        super().__init__(f"DINOv2-{model_size}", device=device, **kwargs)
        self.model_size = model_size
        self._embedding_dim = self.AVAILABLE_MODELS[model_size]
        self._metadata.update(
            {"model_size": model_size, "embedding_dim": self._embedding_dim}
        )

    def load_model(self) -> None:
        if self._loaded:
            return
        logger.info("Loading DINOv2 %s via torch.hub", self.model_size)
        self.model = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_{self.model_size}"
        )
        self.model.eval().to(self.device)
        self._loaded = True

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            self.load_model()
        return self.model(batch.to(self.device))
