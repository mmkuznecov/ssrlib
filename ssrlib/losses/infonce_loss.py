"""InfoNCE loss."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import torch
import torch.nn.functional as F

from .base import BaseLoss


class InfoNCELoss(BaseLoss):
    """InfoNCE: contrast a query against one positive and ``K`` negatives.

    Reference: Oord et al. 2018; popularised in SSL by MoCo.

    Expects:
        query: (B, D)
        positive: (B, D) — the matched positive for each query.
        negatives: (B, K, D) or (K, D) shared across the batch.
    """

    _loss_category: ClassVar[str] = "contrastive"
    _loss_modality: ClassVar[str] = "any"
    _loss_properties: ClassVar[Dict[str, Any]] = {"requires_negatives": True}

    def __init__(self, temperature: float = 0.07, **kwargs):
        super().__init__("InfoNCELoss", **kwargs)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self._metadata.update({"temperature": self.temperature})

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        q = F.normalize(query, dim=1)
        p = F.normalize(positive, dim=1)

        if negatives.dim() == 2:
            n = F.normalize(negatives, dim=1)
            l_pos = (q * p).sum(dim=1, keepdim=True)
            l_neg = q @ n.T
            logits = torch.cat([l_pos, l_neg], dim=1)
        elif negatives.dim() == 3:
            n = F.normalize(negatives, dim=2)
            l_pos = (q * p).sum(dim=1, keepdim=True)
            l_neg = torch.einsum("bd,bkd->bk", q, n)
            logits = torch.cat([l_pos, l_neg], dim=1)
        else:
            raise ValueError(f"negatives must be 2D or 3D, got {negatives.shape}")

        logits = logits / self.temperature
        targets = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, targets)
