"""Datasets package for ssrlib.

Public API: dataset classes (explicit imports), plus the HF registry helpers.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import BaseDataset
from .celeba import CelebADataset
from .hf_registry import (
    HFDatasetInfo,
    HF_DATASET_REGISTRY,
    get_hf_dataset_info,
    list_hf_datasets,
)
from .hf_vision import CIFAR10Dataset, Food101Dataset, HFVisionDataset
from .imagenet100 import ImageNet100Dataset
from .kaggle_mixin import KaggleDatasetMixin
from .synthtest_dataset import SynthTestDataset

_DATASET_CLASSES: List[Type[BaseDataset]] = [
    SynthTestDataset,
    HFVisionDataset,
    CIFAR10Dataset,
    Food101Dataset,
    CelebADataset,
    ImageNet100Dataset,
]

_REGISTRY: Dict[str, Type[BaseDataset]] = {
    cls.__name__: cls for cls in _DATASET_CLASSES
}


def list_datasets() -> List[str]:
    return list(_REGISTRY.keys())


def get_available_datasets() -> Dict[str, Type[BaseDataset]]:
    return dict(_REGISTRY)


def create_dataset(name: str, **kwargs) -> BaseDataset:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = [
    "BaseDataset",
    "KaggleDatasetMixin",
    "SynthTestDataset",
    "HFVisionDataset",
    "CIFAR10Dataset",
    "Food101Dataset",
    "CelebADataset",
    "ImageNet100Dataset",
    "HFDatasetInfo",
    "HF_DATASET_REGISTRY",
    "get_hf_dataset_info",
    "list_hf_datasets",
    "list_datasets",
    "get_available_datasets",
    "create_dataset",
]
