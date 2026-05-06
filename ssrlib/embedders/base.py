"""Base class for all embedders."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Base class for all embedders.

    Subclasses set the class-level ``_embedder_*`` attributes and implement:
        - ``load_model()``: download / instantiate the underlying model.
        - ``get_embedding_dim()``: return the output dimension.
        - ``forward(batch)``: produce embeddings for a batch tensor.

    The default ``embed_dataset`` collects samples in mini-batches of the given
    size, stacks them with ``torch.stack`` (so each dataset must yield single
    tensors of the same shape), runs ``forward``, and returns a single 2-D
    numpy array of shape ``(N, embedding_dim)``.
    """

    _embedder_category: ClassVar[str] = "general"
    _embedder_modality: ClassVar[str] = "unknown"
    _embedder_properties: ClassVar[Dict[str, Any]] = {}

    def __init__(self, name: str, device: str = "cpu", **kwargs):
        self.name = name
        self.device = device
        self.model: Any = None
        self._loaded = False
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------- abstract
    @abstractmethod
    def load_model(self) -> None:
        """Load / download the model. Must be idempotent."""

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""

    @abstractmethod
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Produce embeddings for a batch."""

    # ------------------------------------------------------------- helpers
    @torch.no_grad()
    def embed_dataset(self, dataset, batch_size: int = 32) -> np.ndarray:
        """Embed every sample in a dataset.

        The dataset is expected to be iterable, yielding single tensors of the
        same shape (see SynthTestDataset / HFVisionDataset.__iter__). For
        labelled datasets, the iter path yields image-only tensors.

        Returns:
            numpy array of shape (N, embedding_dim).
        """
        if not self._loaded:
            self.load_model()

        all_embeddings: List[np.ndarray] = []
        batch: List[torch.Tensor] = []

        for sample in dataset:
            batch.append(sample)
            if len(batch) >= batch_size:
                all_embeddings.append(self._process_batch(batch))
                batch = []
        if batch:
            all_embeddings.append(self._process_batch(batch))

        if not all_embeddings:
            return np.empty((0, self.get_embedding_dim()), dtype=np.float32)
        return np.concatenate(all_embeddings, axis=0)

    def _process_batch(self, batch: List[torch.Tensor]) -> np.ndarray:
        stacked = torch.stack(batch).to(self.device)
        out = self.forward(stacked)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        return np.asarray(out, dtype=np.float32)

    def get_metadata(self) -> Dict[str, Any]:
        meta = {
            "name": self.name,
            "device": self.device,
            "loaded": self._loaded,
            "embedding_dim": self.get_embedding_dim() if self._loaded else None,
            "category": self._embedder_category,
            "modality": self._embedder_modality,
        }
        meta.update(self._metadata)
        return meta

    @classmethod
    def get_embedder_category(cls) -> str:
        return cls._embedder_category

    @classmethod
    def get_embedder_modality(cls) -> str:
        return cls._embedder_modality

    @classmethod
    def get_embedder_properties(cls) -> Dict[str, Any]:
        return cls._embedder_properties.copy()
