import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from typing import Iterator, Dict, Any, Optional, Tuple, List, Union, ClassVar
from pathlib import Path
import logging

from .base import BaseDataset
from .kaggle_mixin import KaggleDatasetMixin

logger = logging.getLogger(__name__)


class CelebADataset(KaggleDatasetMixin, BaseDataset):
    """CelebA Dataset for ssrlib framework."""

    # Class-level metadata
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

    # Expected files and directories
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
        """Initialize CelebA dataset."""
        super().__init__("CelebA", **kwargs)

        self.root = Path(root) / "CelebA"
        self.split = split
        self.task_name = task_name

        # Set default transform
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

        # File paths (set after verification)
        self.split_csv = None
        self.attr_csv = None
        self.images_dir = None

        self.data = None
        self.attr_names = None

        self._metadata.update({"split": split, "task_name": task_name, "root": str(self.root)})

        # Download if needed
        if not self._check_exists():
            logger.info(f"CelebA not found, downloading to {self.root}")
            print(f"CelebA not found, downloading to {self.root}")
            self._download()

        self._load_data()
        self._downloaded = True

    def _get_kaggle_dataset_id(self) -> str:
        """Get Kaggle dataset ID."""
        return "jessicali9530/celeba-dataset"

    def _check_exists(self) -> bool:
        """Check if dataset exists and is properly structured."""
        if not self.root.exists():
            return False

        return self._verify_structure()

    def _verify_structure(self) -> bool:
        """
        Verify CelebA structure is correct.

        Expected structure:
        CelebA/
        ├── list_eval_partition.csv
        ├── list_attr_celeba.csv
        └── img_align_celeba/
            └── img_align_celeba/
                ├── 000001.jpg
                ├── 000002.jpg
                └── ...
        """
        logger.info("Verifying CelebA structure...")

        # Find required files
        self.split_csv = self._find_required_file("list_eval_partition.csv")
        self.attr_csv = self._find_required_file("list_attr_celeba.csv")
        self.images_dir = self._find_required_directory("img_align_celeba")

        # Check all found
        if not (self.split_csv and self.attr_csv and self.images_dir):
            return False

        # Verify images exist
        image_files = list(self.images_dir.glob("*.jpg"))
        if len(image_files) == 0:
            logger.error("No images found in img_align_celeba directory")
            return False

        logger.info(f"✓ Structure verified: {len(image_files)} images found")
        print(f"✓ CelebA structure verified: {len(image_files)} images found")

        return True

    def _find_required_file(self, filename: str) -> Optional[Path]:
        """Find required file and move to root if needed."""
        # Check if already in root
        target_path = self.root / filename
        if target_path.exists():
            logger.info(f"✓ Found {filename}")
            return target_path

        # Search for file
        found_path = self._find_file(filename)
        if found_path:
            logger.info(f"Found {filename} at {found_path}, moving to root")
            return self._move_to_root(found_path, filename)

        logger.error(f"✗ {filename} not found")
        return None

    def _find_required_directory(self, dirname: str) -> Optional[Path]:
        """
        Find required image directory and organize if needed.

        Handles various extraction patterns:
        - img_align_celeba/img_align_celeba/  (correct)
        - img_align_celeba/                  (needs nesting)
        - Images directly in root             (needs organization)
        """
        target_path = self.root / "img_align_celeba" / "img_align_celeba"

        # Already correct structure
        if target_path.exists() and target_path.is_dir():
            logger.info(f"✓ Found {dirname}")
            return target_path

        # Find img_align_celeba directory
        parent_dir = self.root / "img_align_celeba"

        if parent_dir.exists():
            # Check if images are directly in parent
            image_files = list(parent_dir.glob("*.jpg"))

            if image_files:
                # Need to create nested structure
                logger.info("Creating nested img_align_celeba directory...")
                target_path.mkdir(parents=True, exist_ok=True)

                # Move images to nested directory
                for img_file in image_files:
                    img_file.rename(target_path / img_file.name)

                logger.info(f"✓ Organized {len(image_files)} images")
                return target_path

            elif target_path.exists():
                # Nested directory exists
                return target_path

        # Try to find directory anywhere
        found_dir = self._find_directory("img_align_celeba")
        if found_dir:
            logger.info(f"Found img_align_celeba at {found_dir}")

            # Check if it needs nesting
            if found_dir.parent != self.root:
                found_dir = self._move_to_root(found_dir, "img_align_celeba")

            # Ensure nested structure
            if found_dir == parent_dir:
                image_files = list(found_dir.glob("*.jpg"))
                if image_files:
                    target_path.mkdir(parents=True, exist_ok=True)
                    for img_file in image_files:
                        img_file.rename(target_path / img_file.name)

            return target_path

        logger.error(f"✗ {dirname} directory not found")
        return None

    def download(self) -> None:
        """Download CelebA dataset if not present."""
        if self._downloaded:
            return

        if not self._check_exists():
            self._download()
            self._load_data()

        self._downloaded = True

    def _download(self) -> None:
        """Download dataset from Kaggle."""
        self._download_from_kaggle(zip_filename="celeba_dataset.zip")

    def _load_data(self) -> None:
        """Load dataset metadata."""
        if not self.split_csv or not self.attr_csv:
            raise RuntimeError("Dataset files not found. Run download() first.")

        # Load split information
        split_df = pd.read_csv(self.split_csv)
        split_map = {"train": 0, "valid": 1, "test": 2}
        split_df = split_df[split_df["partition"] == split_map[self.split]]

        # Load attributes
        attr_df = pd.read_csv(self.attr_csv)
        self.attr_names = list(attr_df.columns[1:])

        # Validate task name
        if self.task_name not in self.attr_names:
            raise ValueError(f"Unknown task '{self.task_name}'. Available: {self.attr_names}")

        # Merge data
        self.data = pd.merge(
            split_df,
            attr_df[["image_id", self.task_name]],
            on="image_id",
            how="left",
        )

        logger.info(f"Loaded {len(self.data)} samples for split '{self.split}'")

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get item(s) by index."""
        if self.data is None:
            self.download()

        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.data)))
            return [self._get_single_item(i) for i in indices]
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item by index."""
        if idx >= len(self.data) or idx < -len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")

        if idx < 0:
            idx = len(self.data) + idx

        row = self.data.iloc[idx]
        img_path = self.images_dir / row["image_id"]

        if not img_path.exists():
            raise FileNotFoundError(f"Image {img_path} not found")

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get target (convert from -1/1 to 0/1)
        target = torch.tensor(1 if row[self.task_name] == 1 else 0, dtype=torch.long)

        return image, target

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over dataset returning image tensors only."""
        if self.data is None:
            self.download()

        for idx in range(len(self.data)):
            try:
                image, _ = self._get_single_item(idx)
                yield image
            except Exception as e:
                logger.warning(f"Skipping sample {idx}: {str(e)}")
                continue

    def __len__(self) -> int:
        """Return dataset size."""
        if self.data is None:
            self.download()
        return len(self.data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CelebADataset(split='{self.split}', task='{self.task_name}', "
            f"size={len(self) if self.data is not None else 'Unknown'}, "
            f"root='{self.root}')"
        )

    def get_classes(self) -> Dict[str, Any]:
        """Get class information."""
        return {
            "task_name": self.task_name,
            "num_classes": 2,
            "class_names": ["No", "Yes"],
            "class_to_idx": {"No": 0, "Yes": 1},
        }

    def get_all_attributes(self) -> List[str]:
        """Get list of all available attributes."""
        if self.attr_names is None:
            self.download()
        return self.attr_names.copy()

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific sample."""
        if self.data is None:
            self.download()

        if idx >= len(self.data) or idx < -len(self.data):
            raise IndexError(f"Index {idx} out of range")

        if idx < 0:
            idx = len(self.data) + idx

        row = self.data.iloc[idx]
        img_path = self.images_dir / row["image_id"]

        return {
            "index": idx,
            "image_id": row["image_id"],
            "image_path": str(img_path),
            "target_value": row[self.task_name],
            "target_class": "Yes" if row[self.task_name] == 1 else "No",
            "split": self.split,
            "exists": img_path.exists(),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        metadata = super().get_metadata()
        if self.data is not None:
            metadata.update(
                {
                    "num_samples": len(self.data),
                    "image_shape": "(3, 224, 224)",
                    "split": self.split,
                    "task_name": self.task_name,
                    "num_attributes": len(self.attr_names) if self.attr_names else 0,
                }
            )
        return metadata
