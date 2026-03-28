import os
import json
import glob
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Tuple, Union, ClassVar
import torch
from PIL import Image
from torchvision import transforms
import logging

from .base import BaseDataset
from .kaggle_mixin import KaggleDatasetMixin

logger = logging.getLogger(__name__)


class ImageNet100Dataset(KaggleDatasetMixin, BaseDataset):
    """ImageNet100 Dataset for ssrlib framework."""

    # Class-level metadata
    _dataset_category: ClassVar[str] = "vision"
    _dataset_modality: ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "num_classes": 100,
        "image_format": "JPEG",
        "processed_image_size": (224, 224),
        "task_type": "multi_class_classification",
        "supports_train_val_split": True,
        "hierarchical_labels": True,
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        labels_path: Optional[str] = None,
        combine_train_splits: bool = True,
        transform: Optional[transforms.Compose] = None,
        **kwargs,
    ):
        """Initialize ImageNet100 dataset."""
        super().__init__("ImageNet100", **kwargs)

        self.root = Path(root) / "ImageNet100"
        self.split = split
        self.combine_train_splits = combine_train_splits
        self.labels_path = labels_path

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

        # Will be loaded after verification
        self.samples = []
        self.synset_to_class = {}
        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._metadata.update(
            {
                "split": split,
                "combine_train_splits": combine_train_splits,
                "root": str(self.root),
            }
        )

        # Download if needed
        if not self._check_exists():
            logger.info(f"ImageNet100 not found, downloading to {self.root}")
            print(f"ImageNet100 not found, downloading to {self.root}")
            self._download()

        self._load_data()
        self._downloaded = True

    def _get_kaggle_dataset_id(self) -> str:
        """Get Kaggle dataset ID."""
        return "ambityga/imagenet100"

    def _check_exists(self) -> bool:
        """Check if dataset exists and is properly structured."""
        if not self.root.exists():
            return False

        return self._verify_structure()

    def _verify_structure(self) -> bool:
        """
        Verify ImageNet100 structure is correct.

        Expected structure:
        ImageNet100/
        ├── train.X1/
        │   ├── n01440764/
        │   ├── n01443537/
        │   └── ...
        ├── train.X2/ (optional)
        ├── val.X/
        │   ├── n01440764/
        │   ├── n01443537/
        │   └── ...
        └── Labels.json (optional)
        """
        logger.info("Verifying ImageNet100 structure...")

        # Find training directories
        train_dirs = self._find_train_directories()
        if not train_dirs:
            logger.error("No training directories found")
            return False

        # Find validation directory
        val_dir = self._find_val_directory()
        if not val_dir:
            logger.error("No validation directory found")
            return False

        # Check for images
        has_images = self._check_has_images(train_dirs + [val_dir])
        if not has_images:
            logger.error("No images found in directories")
            return False

        # Find labels file (optional)
        if self.labels_path is None:
            self.labels_path = self._find_labels_file()

        logger.info(f"✓ Structure verified")
        print(f"✓ ImageNet100 structure verified")
        print(f"  - Training dirs: {len(train_dirs)}")
        print(f"  - Validation dir: {val_dir.name}")
        print(f"  - Labels file: {'Found' if self.labels_path else 'Not found'}")

        return True

    def _find_train_directories(self) -> List[Path]:
        """Find training directories (train.X1, train.X2, etc.)."""
        train_dirs = []

        # Look for train.X* pattern
        for pattern in ["train.X*", "train*", "Train*"]:
            found = list(self.root.glob(pattern))
            for path in found:
                if path.is_dir() and self._has_synset_subdirs(path):
                    train_dirs.append(path)

        # Sort by name
        train_dirs.sort(key=lambda x: x.name)

        if train_dirs:
            logger.info(f"Found {len(train_dirs)} training directories")

        return train_dirs

    def _find_val_directory(self) -> Optional[Path]:
        """Find validation directory (val.X or similar)."""
        # Try exact name first
        val_dir = self.root / "val.X"
        if val_dir.exists() and self._has_synset_subdirs(val_dir):
            logger.info("Found validation directory: val.X")
            return val_dir

        # Try various patterns
        for pattern in ["val*", "Val*", "validation*", "valid*"]:
            found = list(self.root.glob(pattern))
            for path in found:
                if path.is_dir() and self._has_synset_subdirs(path):
                    logger.info(f"Found validation directory: {path.name}")
                    return path

        return None

    def _has_synset_subdirs(self, directory: Path) -> bool:
        """Check if directory contains synset subdirectories (n01...)."""
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        if not subdirs:
            return False

        # Check if any subdirectory starts with 'n' (synset pattern)
        return any(d.name.startswith("n") for d in subdirs)

    def _check_has_images(self, directories: List[Path]) -> bool:
        """Check if directories contain image files."""
        for directory in directories:
            for synset_dir in directory.iterdir():
                if synset_dir.is_dir():
                    # Check for image files
                    image_files = (
                        list(synset_dir.glob("*.JPEG"))
                        + list(synset_dir.glob("*.jpg"))
                        + list(synset_dir.glob("*.png"))
                    )
                    if image_files:
                        return True
        return False

    def _find_labels_file(self) -> Optional[str]:
        """Find Labels.json file if it exists."""
        for pattern in ["Labels.json", "labels.json", "LABELS.json"]:
            labels_path = self.root / pattern
            if labels_path.exists():
                logger.info(f"Found labels file: {pattern}")
                return str(labels_path)

        return None

    def download(self) -> None:
        """Download ImageNet100 dataset if not present."""
        if self._downloaded:
            return

        if not self._check_exists():
            self._download()
            self._load_data()

        self._downloaded = True

    def _download(self) -> None:
        """Download dataset from Kaggle."""
        self._download_from_kaggle(zip_filename="imagenet100.zip")

    def _load_data(self) -> None:
        """Load dataset structure and samples."""
        # Load labels if available
        if self.labels_path and os.path.exists(self.labels_path):
            self._load_labels()

        # Load samples based on split
        if self.split in ["train", "training"]:
            self._load_train_samples()
        elif self.split in ["val", "valid", "validation"]:
            self._load_val_samples()
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Create class mappings
        self._create_class_mappings()

        logger.info(f"Loaded {len(self.samples)} samples for split '{self.split}'")

    def _load_labels(self) -> None:
        """Load synset to class name mapping."""
        with open(self.labels_path, "r") as f:
            labels = json.load(f)

        self.synset_to_class = {
            synset: desc.split(",")[0].strip() for synset, desc in labels.items()
        }

        logger.info(f"Loaded {len(self.synset_to_class)} class labels")

    def _load_train_samples(self) -> None:
        """Load training samples."""
        train_dirs = self._find_train_directories()

        if not train_dirs:
            raise ValueError(f"No training directories found in {self.root}")

        # Use all or just first directory based on combine_train_splits
        if self.combine_train_splits:
            dirs_to_load = train_dirs
        else:
            dirs_to_load = train_dirs[:1]

        logger.info(f"Loading from {len(dirs_to_load)} training directories")

        for train_dir in dirs_to_load:
            self._load_samples_from_directory(train_dir)

    def _load_val_samples(self) -> None:
        """Load validation samples."""
        val_dir = self._find_val_directory()

        if not val_dir:
            raise ValueError(f"Validation directory not found in {self.root}")

        logger.info(f"Loading from validation directory: {val_dir.name}")
        self._load_samples_from_directory(val_dir)

    def _load_samples_from_directory(self, directory: Path) -> None:
        """Load all samples from a directory with synset structure."""
        for synset_dir in directory.iterdir():
            if not synset_dir.is_dir():
                continue

            synset_id = synset_dir.name
            class_name = self.synset_to_class.get(synset_id, synset_id)

            # Find all image files
            image_extensions = ["*.JPEG", "*.jpg", "*.jpeg", "*.png"]
            for ext in image_extensions:
                for img_path in synset_dir.glob(ext):
                    self.samples.append((str(img_path), class_name, synset_id))

    def _create_class_mappings(self) -> None:
        """Create class to index mappings."""
        if self.synset_to_class:
            unique_classes = sorted(set(self.synset_to_class.values()))
        else:
            unique_classes = sorted(set(sample[1] for sample in self.samples))

        self.class_names = unique_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        logger.info(f"Created mappings for {len(self.class_names)} classes")

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get item(s) by index."""
        if not self._downloaded:
            self.download()

        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.samples)))
            return [self._get_single_item(i) for i in indices]
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item by index."""
        if idx >= len(self.samples) or idx < -len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        if idx < 0:
            idx = len(self.samples) + idx

        img_path, class_name, synset_id = self.samples[idx]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_path} not found")

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get class index
        class_idx = self.class_to_idx[class_name]
        target = torch.tensor(class_idx, dtype=torch.long)

        return image, target

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over dataset returning image tensors only."""
        if not self._downloaded:
            self.download()

        for idx in range(len(self.samples)):
            try:
                image, _ = self._get_single_item(idx)
                yield image
            except Exception as e:
                logger.warning(f"Skipping sample {idx}: {str(e)}")
                continue

    def __len__(self) -> int:
        """Return dataset size."""
        if not self._downloaded:
            self.download()
        return len(self.samples)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ImageNet100Dataset(split='{self.split}', "
            f"size={len(self.samples) if self.samples else 'Unknown'}, "
            f"num_classes={len(self.class_names)}, root='{self.root}')"
        )

    def get_classes(self) -> Dict[str, Any]:
        """Get class information."""
        return {
            "num_classes": len(self.class_names),
            "class_names": self.class_names.copy(),
            "class_to_idx": self.class_to_idx.copy(),
            "idx_to_class": self.idx_to_class.copy(),
            "synset_to_class": (self.synset_to_class.copy() if self.synset_to_class else {}),
        }

    def get_class_name(self, idx: int) -> str:
        """Get class name from index."""
        if idx not in self.idx_to_class:
            raise ValueError(f"Class index {idx} not found")
        return self.idx_to_class[idx]

    def get_class_index(self, class_name: str) -> int:
        """Get index from class name."""
        if class_name not in self.class_to_idx:
            raise ValueError(f"Class name '{class_name}' not found")
        return self.class_to_idx[class_name]

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a specific sample."""
        if not self._downloaded:
            self.download()

        if idx >= len(self.samples) or idx < -len(self.samples):
            raise IndexError(f"Index {idx} out of range")

        if idx < 0:
            idx = len(self.samples) + idx

        img_path, class_name, synset_id = self.samples[idx]

        return {
            "index": idx,
            "image_path": img_path,
            "class_name": class_name,
            "class_index": self.class_to_idx[class_name],
            "synset_id": synset_id,
            "split": self.split,
            "exists": os.path.exists(img_path),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        metadata = super().get_metadata()
        metadata.update(
            {
                "num_samples": len(self.samples),
                "num_classes": len(self.class_names),
                "class_names": self.class_names[:10],  # First 10 for brevity
                "image_shape": "(3, 224, 224)",
                "split": self.split,
                "combine_train_splits": self.combine_train_splits,
            }
        )
        return metadata
