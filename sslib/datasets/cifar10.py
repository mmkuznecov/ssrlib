"""CIFAR-10 dataset from HuggingFace."""

import torch
from PIL import Image
from torchvision import transforms
from typing import Iterator, Dict, Any, Optional, Tuple, List, Union, ClassVar
import logging

from .base import BaseDataset
from .hf_mixin import HFDatasetMixin
from .hf_registry import get_hf_dataset_info

logger = logging.getLogger(__name__)


class CIFAR10Dataset(HFDatasetMixin, BaseDataset):
    """CIFAR-10 Dataset from HuggingFace Hub."""

    # Class-level metadata
    _dataset_category: ClassVar[str] = "vision"
    _dataset_modality: ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "num_classes": 10,
        "image_format": "png",
        "original_image_size": (32, 32),
        "processed_image_size": (224, 224),
        "task_type": "multi_class_classification",
        "source": "huggingface",
        "hf_id": "uoft-cs/cifar10",
    }

    def __init__(
        self,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize CIFAR-10 dataset.

        Args:
            split: Dataset split ('train' or 'test')
            transform: Optional image transforms
            cache_dir: Optional cache directory for HF datasets
            **kwargs: Additional arguments
        """
        super().__init__("CIFAR10", **kwargs)

        self.split = split
        self.cache_dir = cache_dir

        # Get dataset info from registry
        self.hf_info = get_hf_dataset_info("cifar10")

        # Set default transform (upscale from 32x32 to 224x224)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

        # Will be set after loading
        self.hf_dataset = None
        self.image_key = None
        self.label_key = None
        self.hf_keys = None

        self._metadata.update(
            {
                "split": split,
                "hf_id": self.hf_info.hf_id,
                "num_classes": self.hf_info.num_classes,
            }
        )

        # Load dataset
        self.download()
        self._downloaded = True

    def _get_hf_dataset_id(self) -> str:
        """Get HuggingFace dataset ID."""
        return self.hf_info.hf_id

    def _get_hf_split_name(self, split: str) -> str:
        """Map split name to HF split."""
        if split == "train":
            return self.hf_info.train_split
        elif split in ["test", "val", "validation"]:
            return self.hf_info.test_split
        else:
            return split

    def _get_hf_keys(self) -> Dict[str, str]:
        """Get HF dataset column keys."""
        return {
            "image": self.hf_info.image_key,
            "label": self.hf_info.label_key,
        }

    def download(self) -> None:
        """Load dataset from HuggingFace."""
        if self._downloaded:
            return

        logger.info(f"Loading CIFAR-10 dataset (split: {self.split})")
        print(f"Loading CIFAR-10 dataset (split: {self.split})")

        self._load_from_huggingface(self.split, self.cache_dir)
        self._downloaded = True

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Get item(s) by index."""
        if self.hf_dataset is None:
            self.download()

        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.hf_dataset)))
            return [self._get_single_item(i) for i in indices]
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item by index."""
        if idx >= len(self.hf_dataset) or idx < -len(self.hf_dataset):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.hf_dataset)}"
            )

        if idx < 0:
            idx = len(self.hf_dataset) + idx

        example = self.hf_dataset[idx]

        # Get image (CIFAR-10 uses 'img' key)
        image = example[self.image_key]

        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            else:
                image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Get label
        label = example[self.label_key]
        label = self._convert_label(label)

        # Convert to tensor
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return image, label

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over dataset returning image tensors only."""
        if self.hf_dataset is None:
            self.download()

        for idx in range(len(self.hf_dataset)):
            try:
                image, _ = self._get_single_item(idx)
                yield image
            except Exception as e:
                logger.warning(f"Skipping sample {idx}: {str(e)}")
                continue

    def __len__(self) -> int:
        """Return dataset size."""
        if self.hf_dataset is None:
            self.download()
        return len(self.hf_dataset)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CIFAR10Dataset(split='{self.split}', "
            f"size={len(self) if self.hf_dataset is not None else 'Unknown'})"
        )

    def get_classes(self) -> Dict[str, Any]:
        """Get class information."""
        class_names = self._get_class_names()

        if class_names:
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        else:
            class_to_idx = {}
            idx_to_class = {}

        return {
            "num_classes": self._get_num_classes(),
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        }

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific sample."""
        if self.hf_dataset is None:
            self.download()

        if idx >= len(self.hf_dataset) or idx < -len(self.hf_dataset):
            raise IndexError(f"Index {idx} out of range")

        if idx < 0:
            idx = len(self.hf_dataset) + idx

        example = self.hf_dataset[idx]

        return {
            "index": idx,
            "label": example[self.label_key],
            "label_idx": self._convert_label(example[self.label_key]),
            "split": self.split,
            "has_image": self.image_key in example,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        metadata = super().get_metadata()
        if self.hf_dataset is not None:
            metadata.update(
                {
                    "num_samples": len(self.hf_dataset),
                    "num_classes": self._get_num_classes(),
                    "image_shape": "(3, 224, 224)",
                    "original_size": "(3, 32, 32)",
                    "split": self.split,
                    "hf_id": self.hf_info.hf_id,
                }
            )
        return metadata
