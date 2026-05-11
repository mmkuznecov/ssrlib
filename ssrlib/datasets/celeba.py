"""CelebA dataset (Kaggle download)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .base import BaseDataset
from .kaggle_mixin import KaggleDatasetMixin

logger = logging.getLogger(__name__)


class CelebADataset(KaggleDatasetMixin, BaseDataset):
    """CelebA Dataset for ssrlib framework."""

    _dataset_category: ClassVar[str] = "vision"
    _dataset_modality: ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "num_attributes": 40,
        "image_format": "jpg",
        "default_image_size": (178, 218),
        "processed_image_size": (224, 224),
        "num_identities": 10177,
        "total_images": 202599,
        "supports_multi_label": True,
        "task_type": "binary_classification",
    }

    REQUIRED_FILES = ["list_eval_partition.csv", "list_attr_celeba.csv"]
    REQUIRED_DIRS = ["img_align_celeba/img_align_celeba"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        task_name: str = "Attractive",
        transform: Optional[transforms.Compose] = None,
        **kwargs,
    ):
        super().__init__("CelebA", **kwargs)
        self.root = Path(root) / "CelebA"
        self.split = split
        self.task_name = task_name
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
        self.split_csv = None
        self.attr_csv = None
        self.images_dir = None
        self.data: Optional[pd.DataFrame] = None
        self.attr_names: Optional[List[str]] = None

        self._metadata.update(
            {"split": split, "task_name": task_name, "root": str(self.root)}
        )

        if not self._check_exists():
            logger.info("CelebA not found, downloading to %s", self.root)
            self._download()
        self._load_data()
        self._downloaded = True

    def _get_kaggle_dataset_id(self) -> str:
        return "jessicali9530/celeba-dataset"

    def _check_exists(self) -> bool:
        if not self.root.exists():
            return False
        return self._verify_structure()

    def _verify_structure(self) -> bool:
        logger.info("Verifying CelebA structure...")
        self.split_csv = self._find_required_file("list_eval_partition.csv")
        self.attr_csv = self._find_required_file("list_attr_celeba.csv")
        self.images_dir = self._find_required_directory("img_align_celeba")
        if not (self.split_csv and self.attr_csv and self.images_dir):
            return False
        image_files = list(self.images_dir.glob("*.jpg"))
        if len(image_files) == 0:
            logger.error("No images found in img_align_celeba directory")
            return False
        logger.info("Structure verified: %d images found", len(image_files))
        return True

    def _find_required_file(self, filename: str) -> Optional[Path]:
        target_path = self.root / filename
        if target_path.exists():
            return target_path
        found_path = self._find_file(filename)
        if found_path:
            return self._move_to_root(found_path, filename)
        logger.error("%s not found", filename)
        return None

    def _find_required_directory(self, dirname: str) -> Optional[Path]:
        target_path = self.root / "img_align_celeba" / "img_align_celeba"
        if target_path.exists() and target_path.is_dir():
            return target_path

        parent_dir = self.root / "img_align_celeba"
        if parent_dir.exists():
            image_files = list(parent_dir.glob("*.jpg"))
            if image_files:
                target_path.mkdir(parents=True, exist_ok=True)
                for img_file in image_files:
                    img_file.rename(target_path / img_file.name)
                return target_path
            if target_path.exists():
                return target_path

        found_dir = self._find_directory("img_align_celeba")
        if found_dir:
            if found_dir.parent != self.root:
                found_dir = self._move_to_root(found_dir, "img_align_celeba")
            if found_dir == parent_dir:
                image_files = list(found_dir.glob("*.jpg"))
                if image_files:
                    target_path.mkdir(parents=True, exist_ok=True)
                    for img_file in image_files:
                        img_file.rename(target_path / img_file.name)
            return target_path

        logger.error("%s directory not found", dirname)
        return None

    def download(self) -> None:
        if self._downloaded:
            return
        if not self._check_exists():
            self._download()
            self._load_data()
        self._downloaded = True

    def _download(self) -> None:
        self._download_from_kaggle(zip_filename="celeba_dataset.zip")

    def _load_data(self) -> None:
        if not self.split_csv or not self.attr_csv:
            raise RuntimeError("Dataset files not found. Run download() first.")
        split_df = pd.read_csv(self.split_csv)
        split_map = {"train": 0, "valid": 1, "test": 2}
        split_df = split_df[split_df["partition"] == split_map[self.split]]

        attr_df = pd.read_csv(self.attr_csv)
        self.attr_names = list(attr_df.columns[1:])
        if self.task_name not in self.attr_names:
            raise ValueError(
                f"Unknown task '{self.task_name}'. Available: {self.attr_names}"
            )
        self.data = pd.merge(
            split_df, attr_df[["image_id", self.task_name]], on="image_id", how="left"
        )
        logger.info("Loaded %d samples for split '%s'", len(self.data), self.split)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        if self.data is None:
            self.download()
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.data)))
            return [self._get_single_item(i) for i in indices]
        return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.data) or idx < -len(self.data):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.data)}"
            )
        if idx < 0:
            idx = len(self.data) + idx

        row = self.data.iloc[idx]
        img_path = self.images_dir / row["image_id"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image {img_path} not found")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(1 if row[self.task_name] == 1 else 0, dtype=torch.long)
        return image, target

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.data is None:
            self.download()
        for idx in range(len(self.data)):
            try:
                image, _ = self._get_single_item(idx)
                yield image
            except Exception as exc:
                logger.warning("Skipping sample %d: %s", idx, exc)
                continue

    def __len__(self) -> int:
        if self.data is None:
            self.download()
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"CelebADataset(split='{self.split}', task='{self.task_name}', "
            f"size={len(self) if self.data is not None else 'Unknown'}, "
            f"root='{self.root}')"
        )

    def get_classes(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "num_classes": 2,
            "class_names": ["No", "Yes"],
            "class_to_idx": {"No": 0, "Yes": 1},
        }

    def get_all_attributes(self) -> List[str]:
        if self.attr_names is None:
            self.download()
        return list(self.attr_names)

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        if self.data is not None:
            metadata.update(
                {
                    "num_samples": len(self.data),
                    "image_shape": "(3, 224, 224)",
                    "split": self.split,
                    "task_name": self.task_name,
                    "num_attributes": (len(self.attr_names) if self.attr_names else 0),
                }
            )
        return metadata
