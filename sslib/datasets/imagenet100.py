import os
import json
import glob
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Tuple, Union, ClassVar
import torch
from PIL import Image
from torchvision import transforms
import shutil

from .base import BaseDataset
from .kaggle_mixin import KaggleDatasetMixin


class ImageNet100Dataset(KaggleDatasetMixin, BaseDataset):
    """ImageNet100 Dataset for SSLib framework."""

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

        # Set root path with ImageNet100 subdirectory
        self.root = Path(root) / "ImageNet100"
        self.split = split
        self.combine_train_splits = combine_train_splits
        self.labels_path = labels_path

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

        # Will be loaded after download
        self.samples = []
        self.synset_to_class = {}
        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Update metadata
        self._metadata.update(
            {
                "split": split,
                "combine_train_splits": combine_train_splits,
                "root": str(self.root),
            }
        )

        # Check if dataset exists, download if not
        if not self._check_dataset():
            print(f"Downloading ImageNet100 to {self.root}")
            self._download()

        # Load the data
        self._load_data()
        self._downloaded = True

    def _get_kaggle_dataset_id(self) -> str:
        """Get Kaggle dataset ID for ImageNet100."""
        return "ambityga/imagenet100"

    def _get_manual_download_instructions(self) -> list[str]:
        """Get manual download instructions for ImageNet100."""
        return [
            f"1. Go to https://www.kaggle.com/datasets/{self._get_kaggle_dataset_id()}",
            f"2. Download the dataset manually to {self.root}",
            "3. Extract and organize with the following structure:",
            "   - train.X1/, train.X2/, ... (training directories)",
            "   - val.X/ (validation directory)",
            "   - Labels.json (optional, for human-readable class names)",
        ]

    def _organize_extracted_files(self) -> None:
        """Organize ImageNet100 files after extraction."""
        # Flatten if there's a single subdirectory
        self._flatten_single_subdirectory()

        # Verify and rename training directories
        train_dirs = glob.glob(str(self.root / "train.X*"))
        if not train_dirs:
            # Look for alternative naming patterns
            alt_train_dirs = []
            for pattern in ["train*", "Train*", "TRAIN*"]:
                alt_train_dirs.extend(glob.glob(str(self.root / pattern)))

            if alt_train_dirs:
                print("Found alternative training directories, renaming...")
                for i, alt_dir in enumerate(alt_train_dirs):
                    new_name = f"train.X{i+1}"
                    target = self.root / new_name
                    print(f"Renaming {Path(alt_dir).name} to {new_name}")
                    shutil.move(alt_dir, str(target))

        # Verify and rename validation directory
        val_dirs = glob.glob(str(self.root / "val.X*"))
        if not val_dirs:
            # Look for alternative naming patterns
            alt_val_dirs = []
            for pattern in ["val*", "Val*", "VAL*", "validation*", "valid*"]:
                alt_val_dirs.extend(glob.glob(str(self.root / pattern)))

            if alt_val_dirs:
                print("Found alternative validation directory, renaming...")
                alt_dir = alt_val_dirs[0]
                target = self.root / "val.X"
                print(f"Renaming {Path(alt_dir).name} to val.X")
                shutil.move(alt_dir, str(target))

        # Auto-detect labels file if not specified
        if self.labels_path is None:
            for labels_file in ["Labels.json", "labels.json", "LABELS.json"]:
                labels_path = self.root / labels_file
                if labels_path.exists():
                    self.labels_path = str(labels_path)
                    break

    def _check_dataset(self) -> bool:
        """Check if dataset is already downloaded and properly structured."""
        if not self.root.exists():
            return False

        # Check for training directories
        train_dirs = glob.glob(str(self.root / "train.X*"))
        has_train = len(train_dirs) > 0

        # Check for validation directory
        val_dir = self.root / "val.X"
        has_val = val_dir.exists()

        # Check if directories have content
        has_content = False
        if has_train or has_val:
            for pattern in ["train.X*", "val.X"]:
                for dir_path in glob.glob(str(self.root / pattern)):
                    dir_path = Path(dir_path)
                    if dir_path.is_dir():
                        for subdir in dir_path.iterdir():
                            if subdir.is_dir():
                                image_files = [
                                    f
                                    for f in subdir.iterdir()
                                    if f.suffix.lower()
                                    in [".jpg", ".jpeg", ".png", ".bmp"]
                                ]
                                if image_files:
                                    has_content = True
                                    break
                        if has_content:
                            break
                    if has_content:
                        break

        if has_train and has_val and has_content:
            print(f"ImageNet100 dataset found at {self.root}")
            return True
        else:
            return False

    def download(self) -> None:
        """Download ImageNet100 dataset if not already present."""
        if self._downloaded:
            return

        if not self._check_dataset():
            print(f"Downloading ImageNet100 to {self.root}")
            self._download()
            self._load_data()

        self._downloaded = True

    def _download(self) -> None:
        """Download ImageNet100 dataset from Kaggle."""
        self._download_from_kaggle(zip_filename="imagenet100.zip")

    def _load_data(self):
        """Load dataset structure and samples."""
        # Load labels mapping if available
        if self.labels_path and os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                labels = json.load(f)
            self.synset_to_class = {
                synset: desc.split(",")[0].strip() for synset, desc in labels.items()
            }
            self.class_names = sorted(list(set(self.synset_to_class.values())))
        elif self.labels_path is None:
            labels_path = self.root / "Labels.json"
            if labels_path.exists():
                self.labels_path = str(labels_path)
                with open(self.labels_path, "r") as f:
                    labels = json.load(f)
                self.synset_to_class = {
                    synset: desc.split(",")[0].strip()
                    for synset, desc in labels.items()
                }
                self.class_names = sorted(list(set(self.synset_to_class.values())))

        # Load samples based on split
        if self.split in ["train", "training"]:
            self._load_train_data()
        elif self.split in ["val", "valid", "validation"]:
            self._load_val_data()
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Create class mappings after loading samples
        self._create_class_mappings()

    def _load_train_data(self):
        """Load training data from train.X* directories."""
        train_dirs = []

        if self.combine_train_splits:
            train_pattern = str(self.root / "train.X*")
            train_dirs = glob.glob(train_pattern)
            train_dirs.sort()
        else:
            train_dirs = [str(self.root / "train.X1")]

        if not train_dirs:
            raise ValueError(f"No training directories found in {self.root}")

        print(f"Loading training data from: {[Path(d).name for d in train_dirs]}")

        for train_dir in train_dirs:
            if os.path.exists(train_dir):
                self._load_from_directory(train_dir)

    def _load_val_data(self):
        """Load validation data from val.X directory."""
        val_dir = self.root / "val.X"
        if not val_dir.exists():
            raise ValueError(f"Validation directory {val_dir} not found")

        print(f"Loading validation data from: {val_dir.name}")
        self._load_from_directory(str(val_dir))

    def _load_from_directory(self, directory: str):
        """Load samples from a directory with synset subdirectories."""
        dir_path = Path(directory)

        for synset_dir in dir_path.iterdir():
            if synset_dir.is_dir():
                synset_id = synset_dir.name
                class_name = self.synset_to_class.get(synset_id, synset_id)

                for img_path in synset_dir.iterdir():
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        self.samples.append((str(img_path), class_name, synset_id))

    def _create_class_mappings(self):
        """Create class to index mappings."""
        if self.class_names:
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        else:
            unique_classes = sorted(list(set([sample[1] for sample in self.samples])))
            self.class_names = unique_classes
            self.class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}

        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
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
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.samples)}"
            )

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
        """Iterate over dataset returning image tensors only (for pipeline compatibility)."""
        if not self._downloaded:
            self.download()

        for idx in range(len(self.samples)):
            try:
                image, _ = self._get_single_item(idx)
                yield image
            except (FileNotFoundError, Exception) as e:
                print(f"Warning: Skipping sample {idx}: {e}")
                continue

    def __len__(self) -> int:
        """Return dataset size."""
        if not self._downloaded:
            self.download()
        return len(self.samples)

    def __repr__(self) -> str:
        """String representation of dataset."""
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
            "synset_to_class": (
                self.synset_to_class.copy() if self.synset_to_class else {}
            ),
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
                "class_names": self.class_names[:10],
                "image_shape": "(3, 224, 224)",
                "split": self.split,
                "combine_train_splits": self.combine_train_splits,
            }
        )
        return metadata
