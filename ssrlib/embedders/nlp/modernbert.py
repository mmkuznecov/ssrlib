"""ModernBERT text embedder (Warner et al. 2024)."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

from .bert import BERTEmbedder


class ModernBERTEmbedder(BERTEmbedder):
    """ModernBERT thin wrapper. Defaults to ``answerdotai/ModernBERT-base``."""

    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "framework": "transformers",
        "ssl_method": "ModernBERT",
    }

    AVAILABLE_MODELS: ClassVar[Dict[str, int]] = {
        "answerdotai/ModernBERT-base": 768,
        "answerdotai/ModernBERT-large": 1024,
    }

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        pooling: str = "mean",
        max_length: int = 512,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            pooling=pooling,
            max_length=max_length,
            device=device,
            **kwargs,
        )
        self.name = f"ModernBERT-{model_name.split('/')[-1]}"
