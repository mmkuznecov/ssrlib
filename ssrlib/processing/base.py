"""Base class for all processors in ssrlib."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseProcessor(ABC):
    """Base class for all processors in ssrlib.

    A processor consumes a 2-D embedding matrix of shape (N, D) and produces
    either a derived statistic (scalar, vector, or matrix). The output shape
    convention used in ssrlib is:

        - scalar statistic       -> shape (1,)
        - vector statistic       -> shape (k,) for some k
        - matrix statistic       -> shape (D, D) or (N, D)

    Returning a shape-(1,) array (rather than a Python float) keeps the
    pipeline's results dict uniformly typed as ``np.ndarray``.
    """

    def __init__(self, name: str, **kwargs):
        """Initialize processor.

        Args:
            name: Human-readable identifier used as the dict key inside
                  ``PipelineResults.processed``.
            **kwargs: Captured but unused — present so subclasses can pass
                      arbitrary metadata-shaped kwargs through ``super().__init__``.
        """
        self.name = name
        self._metadata: Dict[str, Any] = {}

    @abstractmethod
    def process(self, embeddings: np.ndarray) -> np.ndarray:
        """Process embeddings of shape (N, D) and return a derived array."""

    def get_metadata(self) -> Dict[str, Any]:
        """Get processor metadata, including everything written by ``process``."""
        return {"name": self.name, **self._metadata}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
