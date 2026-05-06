"""Margin-based triplet loss."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import torch
import torch.nn.functional as F

from .base import BaseLoss


class TripletLoss(BaseLoss):
    """Triplet margin loss: ``max(0, d(a,p) - d(a,n) + margin)``."""

    _loss_category: ClassVar[str] = "metric"
    _loss_modality: ClassVar[str] = "any"
    _loss_properties: ClassVar[Dict[str, Any]] = {
        "requires_triplets": True,
    }

    def __init__(self, margin: float = 1.0, p: int = 2, **kwargs):
        super().__init__("TripletLoss", **kwargs)
        if margin <= 0:
            raise ValueError("margin must be positive")
        if p not in (1, 2):
            raise ValueError("p must be 1 or 2")
        self.margin = float(margin)
        self.p = int(p)
        self._metadata.update({"margin": self.margin, "p": self.p})

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return F.triplet_margin_loss(
            anchor, positive, negative, margin=self.margin, p=self.p
        )
