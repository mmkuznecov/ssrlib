"""OpenAI CLIP image embedder via HuggingFace transformers."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict

import torch

from ..base import BaseEmbedder

logger = logging.getLogger(__name__)


class CLIPEmbedder(BaseEmbedder):
    """CLIP image embedder.

    Args:
        model_name: HuggingFace model id, e.g. ``openai/clip-vit-base-patch32``.
    """

    _embedder_category: ClassVar[str] = "vision"
    _embedder_modality: ClassVar[str] = "vision"
    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "framework": "transformers",
        "ssl_method": "CLIP",
    }

    AVAILABLE_MODELS: ClassVar[Dict[str, int]] = {
        "openai/clip-vit-base-patch32": 512,
        "openai/clip-vit-base-patch16": 512,
        "openai/clip-vit-large-patch14": 768,
    }

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(f"CLIP-{model_name.split('/')[-1]}", device=device, **kwargs)
        self.model_name = model_name
        self._embedding_dim = self.AVAILABLE_MODELS.get(model_name, 512)
        self._metadata.update({"model_name": model_name})

    def load_model(self) -> None:
        if self._loaded:
            return
        from transformers import CLIPVisionModel

        logger.info("Loading CLIP %s", self.model_name)
        self.model = CLIPVisionModel.from_pretrained(self.model_name)
        self.model.eval().to(self.device)
        self._loaded = True

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            self.load_model()
        outputs = self.model(pixel_values=batch.to(self.device))
        # Use the [CLS] token / pooler output
        return outputs.pooler_output
