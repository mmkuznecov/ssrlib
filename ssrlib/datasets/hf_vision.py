"""Unified HuggingFace vision dataset.

This single class replaces the previous four-file stack
(``hf_mixin.py`` + ``hf_vision.py`` + ``cifar10.py`` + ``food101.py``).
For backward compatibility, ``CIFAR10Dataset`` and ``Food101Dataset`` are
defined as one-line shims at the bottom of this file.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple, Union

import torch
from PIL import Image
from torchvision import transforms

from .base import BaseDataset
from .hf_registry import HFDatasetInfo, get_hf_dataset_info

logger = logging.getLogger(__name__)


class HFVisionDataset(BaseDataset):
    """HuggingFace vision dataset, configured by short name from the registry.

    Example:

        >>> from ssrlib.datasets import HFVisionDataset
        >>> ds = HFVisionDataset(dataset_name="cifar10", split="train")
    """

    _dataset_category: ClassVar[str] = "vision"
    _dataset_modality: ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {"source": "huggingface"}

    DEFAULT_TRANSFORM: ClassVar[transforms.Compose] = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        cache_dir: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize HF vision dataset.

        Args:
            dataset_name: short key into the HF registry (e.g. "cifar10").
            split: "train", "test", "val", or any HF-native split name.
            transform: torchvision transform pipeline. If None uses ImageNet defaults.
            cache_dir: optional HF cache directory.
            name: optional override for the human-readable dataset name; defaults
                to ``dataset_name.upper()``.
        """
        super().__init__(name=name or dataset_name.upper(), **kwargs)

        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.hf_info: HFDatasetInfo = get_hf_dataset_info(dataset_name)
        self.transform = transform if transform is not None else self.DEFAULT_TRANSFORM

        # Populated after download()
        self.hf_dataset = None
        self.image_key: Optional[str] = None
        self.label_key: Optional[str] = None
        self.label_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_label: Optional[Dict[int, str]] = None

        self._metadata.update(
            {
                "split": split,
                "hf_id": self.hf_info.hf_id,
                "num_classes": self.hf_info.num_classes,
                "registry_key": dataset_name,
            }
        )

        self.download()
        self._downloaded = True

    # --------------------------------------------------------- HF lifecycle
    def download(self) -> None:
        """Load dataset from HuggingFace Hub (cached)."""
        if self._downloaded:
            return
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for HuggingFace datasets. "
                "Install it with: pip install datasets"
            ) from exc

        hf_split = self._map_split_name(self.split)
        logger.info("Loading %s (%s) split=%s", self.name, self.hf_info.hf_id, hf_split)

        try:
            self.hf_dataset = load_dataset(
                self.hf_info.hf_id, split=hf_split, cache_dir=self.cache_dir
            )
        except ValueError as exc:
            if "trust_remote_code" in str(exc).lower():
                raise RuntimeError(
                    f"Cannot load {self.hf_info.hf_id}: it uses a deprecated loading "
                    "script. Check whether the dataset has been updated to Parquet."
                ) from exc
            raise

        self.image_key = self.hf_info.image_key
        self.label_key = self.hf_info.label_key
        self._setup_label_mapping()
        self._downloaded = True

    def _map_split_name(self, split: str) -> str:
        if split == "train":
            return self.hf_info.train_split
        if split in ("test", "val", "validation"):
            return self.hf_info.test_split
        return split

    # ------------------------------------------------------- label mapping
    def _setup_label_mapping(self) -> None:
        """Build str->int mapping when the dataset has string labels.

        Most HF vision datasets use ClassLabel features (already int-coded);
        this is just a defensive fallback for edge-case datasets.
        """
        if not self.hf_dataset or len(self.hf_dataset) == 0:
            return
        try:
            first_label = self.hf_dataset[0][self.label_key]
        except Exception as exc:
            logger.warning("Could not access first label: %s", exc)
            return

        if isinstance(first_label, (int, float)):
            return

        if isinstance(first_label, str):
            all_labels: Optional[set] = None
            try:
                if hasattr(self.hf_dataset, "features"):
                    feat = self.hf_dataset.features.get(self.label_key)
                    if hasattr(feat, "names"):
                        all_labels = set(feat.names)
            except Exception:
                pass
            if all_labels is None:
                try:
                    all_labels = set(self.hf_dataset[self.label_key])
                except Exception:
                    all_labels = self._sample_unique_labels()

            sorted_labels = sorted(all_labels)
            self.label_to_idx = {l: i for i, l in enumerate(sorted_labels)}
            self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}
            logger.info("Built string-label map for %d classes", len(self.label_to_idx))

    def _sample_unique_labels(self) -> set:
        all_labels = set()
        sample_size = min(1000, len(self.hf_dataset))
        step = max(1, len(self.hf_dataset) // sample_size)
        for i in range(0, len(self.hf_dataset), step):
            try:
                all_labels.add(self.hf_dataset[i][self.label_key])
            except Exception:
                continue
        return all_labels

    def _convert_label(self, label: Any) -> int:
        if isinstance(label, str) and self.label_to_idx is not None:
            return self.label_to_idx[label]
        if isinstance(label, (int, float)):
            return int(label)
        return label  # type: ignore

    # --------------------------------------------------------- item access
    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        if self.hf_dataset is None:
            self.download()
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.hf_dataset)))
            return [self._get_single_item(i) for i in indices]
        return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.hf_dataset) or idx < -len(self.hf_dataset):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.hf_dataset)}"
            )
        if idx < 0:
            idx = len(self.hf_dataset) + idx

        example = self.hf_dataset[idx]
        image = example[self.image_key]
        if not isinstance(image, Image.Image):
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            else:
                image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self._convert_label(example[self.label_key])
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return image, label

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.hf_dataset is None:
            self.download()
        for idx in range(len(self.hf_dataset)):
            try:
                image, _ = self._get_single_item(idx)
                yield image
            except Exception as exc:
                logger.warning("Skipping sample %d: %s", idx, exc)
                continue

    def __len__(self) -> int:
        if self.hf_dataset is None:
            self.download()
        return len(self.hf_dataset)

    def __repr__(self) -> str:
        return (
            f"HFVisionDataset(name={self.dataset_name!r}, split={self.split!r}, "
            f"size={len(self) if self.hf_dataset is not None else 'Unknown'})"
        )

    # -------------------------------------------------------------- class info
    def get_classes(self) -> Dict[str, Any]:
        """Class names + index mappings, when available."""
        class_names = self._get_class_names()
        if class_names:
            class_to_idx = {n: i for i, n in enumerate(class_names)}
            idx_to_class = {i: n for n, i in class_to_idx.items()}
        else:
            class_to_idx = {}
            idx_to_class = {}
        return {
            "num_classes": self._get_num_classes(),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        }

    def _get_num_classes(self) -> int:
        if self.label_to_idx is not None:
            return len(self.label_to_idx)
        try:
            if hasattr(self.hf_dataset, "features"):
                feat = self.hf_dataset.features.get(self.label_key)
                if hasattr(feat, "num_classes"):
                    return feat.num_classes
                if hasattr(feat, "names"):
                    return len(feat.names)
        except Exception:
            pass
        return self.hf_info.num_classes

    def _get_class_names(self) -> Optional[List[str]]:
        if self.idx_to_label is not None:
            return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]
        try:
            if hasattr(self.hf_dataset, "features"):
                feat = self.hf_dataset.features.get(self.label_key)
                if hasattr(feat, "names"):
                    return list(feat.names)
        except Exception:
            pass
        return None

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        if self.hf_dataset is not None:
            metadata.update(
                {
                    "num_samples": len(self.hf_dataset),
                    "num_classes": self._get_num_classes(),
                    "split": self.split,
                    "hf_id": self.hf_info.hf_id,
                }
            )
        return metadata


# ---------------------------------------------------------------- shims
class CIFAR10Dataset(HFVisionDataset):
    """Backward-compatible shim around HFVisionDataset(dataset_name='cifar10')."""

    def __init__(self, split: str = "train", **kwargs):
        kwargs.pop("dataset_name", None)
        super().__init__(dataset_name="cifar10", split=split, name="CIFAR10", **kwargs)


class Food101Dataset(HFVisionDataset):
    """Backward-compatible shim around HFVisionDataset(dataset_name='food101')."""

    def __init__(self, split: str = "train", **kwargs):
        kwargs.pop("dataset_name", None)
        super().__init__(dataset_name="food101", split=split, name="Food101", **kwargs)
