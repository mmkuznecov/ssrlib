"""Network-free embedders for testing and quick experimentation."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseEmbedder


class IdentityEmbedder(BaseEmbedder):
    """Random-projection embedder requiring no model download.

    Flattens the input then linearly projects to ``output_dim`` via a fixed,
    seeded random matrix. Useful for unit tests, smoke tests, and shape
    validation without paying the cost of downloading DINOv2 / CLIP weights.

    Args:
        output_dim: target embedding dimension.
        seed: seed for the random projection matrix (also reused for any
            downstream ops that consume the embedder's RNG).
    """

    _embedder_category: ClassVar[str] = "test"
    _embedder_modality: ClassVar[str] = "synthetic"
    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "test_only": True,
        "no_network": True,
        "deterministic": True,
    }

    def __init__(
        self,
        output_dim: int = 32,
        seed: int = 0,
        device: str = "cpu",
        name: str = "Identity",
        **kwargs,
    ):
        super().__init__(name=name, device=device, **kwargs)
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self._proj: Optional[torch.Tensor] = None
        self._metadata.update({"output_dim": self.output_dim, "seed": self.seed})

    def load_model(self) -> None:
        if self._loaded:
            return
        # Defer construction of the projection matrix until first forward
        # pass, when we actually know the input dimension.
        self.model = nn.Identity()
        self._loaded = True

    def get_embedding_dim(self) -> int:
        return self.output_dim

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            self.load_model()
        flat = batch.reshape(batch.shape[0], -1).float()
        if self._proj is None:
            d_in = flat.shape[1]
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            self._proj = torch.randn(d_in, self.output_dim, generator=gen).to(
                self.device
            ) / (
                d_in**0.5
            )  # unit-variance initialization
        return flat.to(self.device) @ self._proj
