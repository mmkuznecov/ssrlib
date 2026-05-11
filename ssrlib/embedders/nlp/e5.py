"""E5 retrieval text embedder.

E5 models (Wang et al. 2022) require ``query: `` / ``passage: `` prefixes for
optimal performance. This subclass injects those automatically when
``embed_dataset`` is called on a list of strings.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List

import numpy as np

from .bert import BERTEmbedder


class E5Embedder(BERTEmbedder):
    """E5 text embedder using last-hidden-state mean pooling (the recommended
    pooling for E5 models)."""

    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "framework": "transformers",
        "ssl_method": "E5",
    }

    AVAILABLE_MODELS: ClassVar[Dict[str, int]] = {
        "intfloat/e5-small-v2": 384,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-large-v2": 1024,
    }

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        prefix: str = "passage: ",
        max_length: int = 512,
        device: str = "cpu",
        **kwargs,
    ):
        kwargs.pop("pooling", None)  # E5 always uses mean pooling
        super().__init__(
            model_name=model_name,
            pooling="mean",
            max_length=max_length,
            device=device,
            **kwargs,
        )
        self.prefix = prefix
        self.name = f"E5-{model_name.split('/')[-1]}"
        self._metadata.update({"prefix": prefix})

    def _embed_text_batch(self, texts: List[str]) -> np.ndarray:
        prefixed = [self.prefix + t for t in texts]
        return super()._embed_text_batch(prefixed)
