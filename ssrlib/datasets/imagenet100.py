"""ImageNet-100 dataset (Kaggle download)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple, Union

import torch
from PIL import Image
from torchvision import transforms

from .base import BaseDataset
from .kaggle_mixin import KaggleDatasetMixin

logger = logging.getLogger(__name__)


class ImageNet100Dataset(KaggleDatasetMixin, BaseDataset):
    """ImageNet-100 dataset for ssrlib.

    Downloaded from Kaggle ``ambityga/imagenet100``.
    """

    _dataset_category: ClassVar[str] = "vision"
    _dataset_modality: ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "num_classes": 100,
        "image_format": "JPEG",
        "task_type": "classification",
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        **kwargs,
    ):
        super().__init__("ImageNet100", **kwargs)
        self.root = Path(root) / "ImageNet100"
        self.split = split
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        self._metadata.update({"split": split, "root": str(self.root)})

        if not self._check_exists():
            logger.info("ImageNet-100 not found, downloading to %s", self.root)
            self._download()
        self._load_data()
        self._downloaded = True

    def _get_kaggle_dataset_id(self) -> str:
        return "ambityga/imagenet100"

    def _check_exists(self) -> bool:
        return self.root.exists() and self._verify_structure()

    def _verify_structure(self) -> bool:
        if not self.root.exists():
            return False
        # Loose check: at least one subdirectory with images.
        for sub in self.root.iterdir():
            if sub.is_dir() and any(sub.rglob("*.JPEG")) or any(sub.rglob("*.jpg")):
                return True
        return False

    def download(self) -> None:
        if self._downloaded:
            return
        if not self._check_exists():
            self._download()
            self._load_data()
        self._downloaded = True

    def _download(self) -> None:
        self._download_from_kaggle(zip_filename="imagenet100.zip")

    def _load_data(self) -> None:
        # Convention: train/<class>/<image> and val/<class>/<image>
        split_dir = self.root / self.split
        if not split_dir.exists():
            # Some Kaggle releases unpack into nested folders; pick best match.
            for cand in [self.root / "train", self.root / "train.X1", self.root]:
                if cand.exists() and any(cand.iterdir()):
                    split_dir = cand
                    break
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        for class_dir in class_dirs:
            idx = self.class_to_idx[class_dir.name]
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_path, idx))
        logger.info(
            "Loaded %d ImageNet-100 samples across %d classes",
            len(self.samples),
            len(self.class_to_idx),
        )

    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.samples) or idx < -len(self.samples):
            raise IndexError(f"Index {idx} out of range (size={len(self.samples)})")
        if idx < 0:
            idx = len(self.samples) + idx
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx: Union[int, slice]):
        if not self.samples:
            self.download()
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.samples)))
            return [self._get_single_item(i) for i in indices]
        return self._get_single_item(idx)

    def __iter__(self) -> Iterator[torch.Tensor]:
        if not self.samples:
            self.download()
        for i in range(len(self.samples)):
            try:
                img, _ = self._get_single_item(i)
                yield img
            except Exception as exc:
                logger.warning("Skipping sample %d: %s", i, exc)

    def __len__(self) -> int:
        if not self.samples:
            self.download()
        return len(self.samples)

    def get_classes(self) -> Dict[str, Any]:
        return {
            "num_classes": len(self.class_to_idx),
            "class_names": list(self.class_to_idx.keys()),
            "class_to_idx": dict(self.class_to_idx),
            "idx_to_class": {i: n for n, i in self.class_to_idx.items()},
        }
