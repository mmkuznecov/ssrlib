"""SimCLR / NT-Xent contrastive loss."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import torch
import torch.nn.functional as F

from .base import BaseLoss


class ContrastiveLoss(BaseLoss):
    """NT-Xent (normalized temperature-scaled cross-entropy) loss.

    Reference: SimCLR (Chen et al. 2020).

    Expects two views ``z1, z2`` of shape ``(B, D)`` such that ``z1[i]`` and
    ``z2[i]`` form a positive pair.
    """

    _loss_category: ClassVar[str] = "contrastive"
    _loss_modality: ClassVar[str] = "any"
    _loss_properties: ClassVar[Dict[str, Any]] = {"requires_two_views": True}

    def __init__(self, temperature: float = 0.5, **kwargs):
        super().__init__("ContrastiveLoss", **kwargs)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self._metadata.update({"temperature": self.temperature})

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"Shape mismatch: {z1.shape} vs {z2.shape}")
        batch_size = z1.shape[0]

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        sim = z @ z.T / self.temperature  # (2B, 2B)

        # Mask self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positives: i <-> i+B
        targets = torch.cat(
            [torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)]
        ).to(z.device)
        return F.cross_entropy(sim, targets)
