"""Synthetic test dataset that yields random image-like tensors."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterator, List, Optional, Union

import numpy as np
import torch

from .base import BaseDataset


class SynthTestDataset(BaseDataset):
    """Synthetic test dataset yielding random image-like tensors.

    Useful for unit tests and pipeline smoke tests without downloading any
    real data. Tensors are deterministic when ``seed`` is provided.
    """

    _dataset_category: ClassVar[str] = "synthetic"
    _dataset_modality: ClassVar[str] = "synthetic"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "default_tensor_shape": (3, 224, 224),
        "default_num_tensors": 100,
        "deterministic": True,
        "task_type": "testing",
        "supports_custom_shapes": True,
        "value_range": (-2.0, 2.0),
    }

    def __init__(
        self,
        tensors_num: int = 100,
        seed: Optional[int] = None,
        tensor_shape: tuple = (3, 224, 224),
        **kwargs,
    ):
        super().__init__("SynthTest", **kwargs)
        if tensors_num <= 0:
            raise ValueError(f"tensors_num must be positive, got {tensors_num}")
        if len(tensor_shape) != 3:
            raise ValueError(
                f"tensor_shape must have 3 dimensions, got {len(tensor_shape)}"
            )
        self.tensors_num = tensors_num
        self.seed = seed
        self.tensor_shape = tensor_shape
        self._metadata.update(
            {
                "tensors_num": tensors_num,
                "tensor_shape": tensor_shape,
                "seed": seed,
                "synthetic": True,
                "dataset_type": "synthetic_test",
            }
        )
        self._downloaded = True

    def download(self) -> None:
        # No-op for synthetic data; mark as ready.
        self._downloaded = True

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(idx, slice):
            indices = range(*idx.indices(self.tensors_num))
            return [self._get_single_item(i) for i in indices]
        return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> torch.Tensor:
        if idx >= self.tensors_num or idx < -self.tensors_num:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.tensors_num}"
            )
        if idx < 0:
            idx = self.tensors_num + idx
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed + idx)
            tensor = torch.randn(*self.tensor_shape, generator=generator)
        else:
            tensor = torch.randn(*self.tensor_shape)
        return torch.clamp(tensor, -2.0, 2.0)

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Use the same per-index generator as __getitem__ to keep iter/index
        # consistent and to avoid depending on global RNG state.
        for idx in range(self.tensors_num):
            yield self._get_single_item(idx)

    def __len__(self) -> int:
        return self.tensors_num

    def __repr__(self) -> str:
        return (
            f"SynthTestDataset(size={self.tensors_num}, "
            f"shape={self.tensor_shape}, seed={self.seed})"
        )

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self._metadata["seed"] = seed

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "num_samples": self.tensors_num,
                "image_shape": str(self.tensor_shape),
                "synthetic": True,
                "deterministic": self.seed is not None,
                "seed_used": self.seed,
            }
        )
        return metadata
