"""Base class for SSL loss functions."""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """Base class for self-supervised loss functions in ssrlib."""

    _loss_category: ClassVar[str] = "general"
    _loss_modality: ClassVar[str] = "any"
    _loss_properties: ClassVar[Dict[str, Any]] = {}

    def __init__(self, name: str, **kwargs):
        super().__init__()
        self.name = name
        self._metadata: Dict[str, Any] = {}

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self._loss_category,
            "modality": self._loss_modality,
            **self._metadata,
        }

    @classmethod
    def get_loss_category(cls) -> str:
        return cls._loss_category

    @classmethod
    def get_loss_modality(cls) -> str:
        return cls._loss_modality

    @classmethod
    def get_loss_properties(cls) -> Dict[str, Any]:
        return cls._loss_properties.copy()

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError("Subclasses must implement forward()")
