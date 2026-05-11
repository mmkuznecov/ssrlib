"""Losses package for ssrlib."""

from __future__ import annotations

from typing import Dict, List, Type

from .base import BaseLoss
from .contrastive_loss import ContrastiveLoss
from .deepinfomax_loss import DeepInfoMaxLoss
from .infonce_loss import InfoNCELoss
from .triplet_loss import TripletLoss

_LOSS_CLASSES: List[Type[BaseLoss]] = [
    ContrastiveLoss,
    InfoNCELoss,
    TripletLoss,
    DeepInfoMaxLoss,
]

_REGISTRY: Dict[str, Type[BaseLoss]] = {cls.__name__: cls for cls in _LOSS_CLASSES}


def list_losses() -> List[str]:
    return list(_REGISTRY.keys())


def get_available_losses() -> Dict[str, Type[BaseLoss]]:
    return dict(_REGISTRY)


def create_loss(name: str, **kwargs) -> BaseLoss:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = [
    "BaseLoss",
    "ContrastiveLoss",
    "InfoNCELoss",
    "TripletLoss",
    "DeepInfoMaxLoss",
    "list_losses",
    "get_available_losses",
    "create_loss",
]
