"""Base class for all datasets in ssrlib."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Iterator

import torch


class BaseDataset(ABC):
    """Base class for all datasets in ssrlib with self-describing metadata."""

    _dataset_category: ClassVar[str] = "general"
    _dataset_modality: ClassVar[str] = "unknown"
    _dataset_properties: ClassVar[Dict[str, Any]] = {}

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._metadata: Dict[str, Any] = {}
        self._downloaded = False

    @abstractmethod
    def download(self) -> None:
        """Download dataset if not already present."""

    @abstractmethod
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over dataset returning tensors."""

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": len(self),
            "downloaded": self._downloaded,
            **self._metadata,
        }

    @classmethod
    def get_dataset_category(cls) -> str:
        return cls._dataset_category

    @classmethod
    def get_dataset_modality(cls) -> str:
        return cls._dataset_modality

    @classmethod
    def get_dataset_properties(cls) -> Dict[str, Any]:
        return cls._dataset_properties.copy()
