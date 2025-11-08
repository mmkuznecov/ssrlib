import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from typing import Iterator, Dict, Any, Optional, Tuple, List, Union, ClassVar
from pathlib import Path

from .base import BaseDataset
from .kaggle_mixin import KaggleDatasetMixin


class CelebADataset(KaggleDatasetMixin, BaseDataset):
    """CelebA Dataset for SSLib framework."""

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

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        task_name: str = "Attractive",
        transform: Optional[transforms.Compose] = None,
        **kwargs,
    ):
        """Initialize CelebA dataset.

        Args:
            root: Root directory for dataset (default: "data")
            split: Which split to use ('train', 'valid', 'test')
            task_name: Name of the attribute to predict
            transform: Optional transform for images
        """
        super().__init__("CelebA", **kwargs)

        # Set root path with CelebA subdirectory
        self.root = Path(root) / "CelebA"
        self.split = split
        self.task_name = task_name

        # Set default transform if none provided
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

        # Define expected file paths
        self.split_csv = self.root / "list_eval_partition.csv"
        self.attr_csv = self.root / "list_attr_celeba.csv"
        self.images_dir = self.root / "img_align_celeba" / "img_align_celeba"

        # Will be loaded after download/check
        self.data = None
        self.attr_names = None

        # Update metadata
        self._metadata.update(
            {"split": split, "task_name": task_name, "root": str(self.root)}
        )

        # Check if dataset exists, download if not
        if not self._check_dataset():
            print(f"Downloading CelebA to {self.root}")
            self._download()

        # Load the data
        self._load_data()
        self._downloaded = True

    def _get_kaggle_dataset_id(self) -> str:
        """Get Kaggle dataset ID for CelebA."""
        return "jessicali9530/celeba-dataset"

    def _get_manual_download_instructions(self) -> list[str]:
        """Get manual download instructions for CelebA."""
        return [
            f"1. Go to https://www.kaggle.com/datasets/{self._get_kaggle_dataset_id()}",
            f"2. Download the dataset manually to {self.root}",
            "3. Extract and organize with the following structure:",
            "   - list_eval_partition.csv",
            "   - list_attr_celeba.csv",
            "   - img_align_celeba/img_align_celeba/ (directory with images)",
        ]

    def _organize_extracted_files(self) -> None:
        """Organize CelebA files after extraction.

        The Kaggle download typically extracts to the correct structure already.
        We just verify and search for missing files if needed.
        """
        # Check if files are already in the right place
        files_ok = (
            self.split_csv.exists()
            and self.attr_csv.exists()
            and self.images_dir.exists()
        )

        if files_ok:
            print("Dataset structure is already correct")
            return

        # If not, try to find and organize files
        print("Organizing dataset structure...")

        # Find and move CSV files if needed
        csv_files = ["list_eval_partition.csv", "list_attr_celeba.csv"]
        for csv_name in csv_files:
            if not (self.root / csv_name).exists():
                found = self._find_and_move_file(csv_name)
                if not found:
                    raise FileNotFoundError(
                        f"Could not find {csv_name} after extraction"
                    )

        # Find image directory if needed
        if not self.images_dir.exists():
            # First check if img_align_celeba directory exists at root level
            parent_img_dir = self.root / "img_align_celeba"
            if parent_img_dir.exists():
                # Check if images are directly in this directory or in a subdirectory
                image_files = list(parent_img_dir.glob("*.jpg"))
                if image_files:
                    # Images are directly in img_align_celeba, need to create subdirectory
                    print(
                        "Images found directly in img_align_celeba, creating proper structure..."
                    )
                    self.images_dir.mkdir(parents=True, exist_ok=True)
                    for img_file in image_files:
                        img_file.rename(self.images_dir / img_file.name)
                elif (parent_img_dir / "img_align_celeba").exists():
                    # Already in correct structure
                    print("Image directory structure is correct")
                else:
                    raise FileNotFoundError(
                        "img_align_celeba directory exists but contains no images"
                    )
            else:
                # Try to find img_align_celeba anywhere in the root
                found = self._find_and_move_directory("img_align_celeba")
                if not found:
                    raise FileNotFoundError(
                        "Could not find img_align_celeba directory after extraction"
                    )

    def _check_dataset(self) -> bool:
        """Check if dataset is already downloaded and properly structured."""
        required_files = [self.split_csv, self.attr_csv]

        # Check if all required files exist and images directory exists
        files_exist = all(f.exists() for f in required_files)
        images_exist = self.images_dir.exists() and any(self.images_dir.iterdir())

        if files_exist and images_exist:
            print(f"CelebA dataset found at {self.root}")
            return True
        else:
            return False

    def download(self) -> None:
        """Download CelebA dataset if not already present."""
        if self._downloaded:
            return

        if not self._check_dataset():
            print(f"Downloading CelebA to {self.root}")
            self._download()
            self._load_data()

        self._downloaded = True

    def _download(self) -> None:
        """Download CelebA dataset from Kaggle."""
        self._download_from_kaggle(zip_filename="celeba_dataset.zip")

    def _load_data(self):
        """Load dataset metadata."""
        if not self.split_csv.exists() or not self.attr_csv.exists():
            raise FileNotFoundError(f"Required CSV files not found in {self.root}")

        # Load split information
        split_df = pd.read_csv(self.split_csv)
        split_map = {"train": 0, "valid": 1, "test": 2}
        split_df = split_df[split_df["partition"] == split_map[self.split]]

        # Load attributes
        attr_df = pd.read_csv(self.attr_csv)
        self.attr_names = list(attr_df.columns[1:])  # Store all attribute names

        # Validate task name
        if self.task_name not in self.attr_names:
            raise ValueError(
                f"Unknown task {self.task_name}. Available: {self.attr_names}"
            )

        # Merge data
        self.data = pd.merge(
            split_df,
            attr_df[["image_id", self.task_name]],
            on="image_id",
            how="left",
        )

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
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
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.data)}"
            )

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

        # Get target (convert from -1/1 to 0/1 for standard classification)
        target = torch.tensor(1 if row[self.task_name] == 1 else 0, dtype=torch.long)

        return image, target

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over dataset returning image tensors only (for pipeline compatibility)."""
        if self.data is None:
            self.download()

        for idx in range(len(self.data)):
            try:
                image, _ = self._get_single_item(idx)
                yield image
            except (FileNotFoundError, Exception) as e:
                print(f"Warning: Skipping sample {idx}: {e}")
                continue

    def __len__(self) -> int:
        """Return dataset size."""
        if self.data is None:
            self.download()
        return len(self.data)

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"CelebADataset(split='{self.split}', task='{self.task_name}', "
            f"size={len(self) if self.data is not None else 'Unknown'}, "
            f"root='{self.root}')"
        )

    def get_classes(self) -> Dict[str, Any]:
        """Get class information for the current task."""
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
