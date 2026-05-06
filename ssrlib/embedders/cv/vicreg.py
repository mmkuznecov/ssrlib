"""VICReg vision embedder via torch.hub."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict

import torch

from ..base import BaseEmbedder

logger = logging.getLogger(__name__)


class VICRegEmbedder(BaseEmbedder):
    """VICReg image embedder loaded via torch.hub (Bardes et al. 2022)."""

    _embedder_category: ClassVar[str] = "vision"
    _embedder_modality: ClassVar[str] = "vision"
    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "framework": "torch.hub",
        "ssl_method": "VICReg",
    }

    AVAILABLE_MODELS: ClassVar[Dict[str, int]] = {
        "resnet50": 2048,
        "resnet50x2": 4096,
    }

    def __init__(self, model_size: str = "resnet50", device: str = "cpu", **kwargs):
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model_size {model_size!r}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )
        super().__init__(f"VICReg-{model_size}", device=device, **kwargs)
        self.model_size = model_size
        self._embedding_dim = self.AVAILABLE_MODELS[model_size]
        self._metadata.update({"model_size": model_size})

    def load_model(self) -> None:
        if self._loaded:
            return
        logger.info("Loading VICReg %s via torch.hub", self.model_size)
        self.model = torch.hub.load("facebookresearch/vicreg:main", self.model_size)
        self.model.eval().to(self.device)
        self._loaded = True

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            self.load_model()
        return self.model(batch.to(self.device))
