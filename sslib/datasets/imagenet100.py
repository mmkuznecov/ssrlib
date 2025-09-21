import os
import json
import glob
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List
import torch
from PIL import Image
from torchvision import transforms

from ..core.base import BaseDataset


class ImageNet100Dataset(BaseDataset):
    """ImageNet100 Dataset for SSLib framework."""
    
    def __init__(self, root: str, split: str = "train", 
                 labels_path: Optional[str] = None,
                 combine_train_splits: bool = True,
                 transform: Optional[transforms.Compose] = None, **kwargs):
        """Initialize ImageNet100 dataset.
        
        Args:
            root: Root directory containing train.X1, train.X2, etc. and val.X
            split: Which split to use ('train', 'val', 'valid')
            labels_path: Path to Labels.json file for synset to class name mapping
            combine_train_splits: Whether to combine all train.X* directories for training
            transform: Optional transform for images
        """
        super().__init__("ImageNet100", **kwargs)
        
        self.root = Path(root)
        self.split = split
        self.combine_train_splits = combine_train_splits
        self.labels_path = labels_path
        
        # Set default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform
            
        # Will be loaded after download
        self.samples = []
        self.synset_to_class = {}
        self.class_names = []
        
        # Update metadata
        self._metadata.update({
            "split": split,
            "combine_train_splits": combine_train_splits,
            "root": str(root)
        })
        
    def download(self) -> None:
        """Download/prepare ImageNet100 dataset."""
        if self._downloaded:
            return
            
        if not self.root.exists():
            print(f"Please manually download ImageNet100 dataset to {self.root}")
            print("Required structure:")
            print("  - train.X1/, train.X2/, ... (training directories)")
            print("  - val.X/ (validation directory)")
            print("  - Labels.json (optional, for human-readable class names)")
            return
            
        # Auto-detect labels path if not provided
        if self.labels_path is None:
            labels_path = self.root / "Labels.json"
            if labels_path.exists():
                self.labels_path = str(labels_path)
                
        self._load_data()
        self._downloaded = True
        
    def _load_data(self):
        """Load dataset structure and samples."""
        # Load labels mapping if available
        if self.labels_path and os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                labels = json.load(f)
            # Process labels: take first part before comma
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
            
    def _load_train_data(self):
        """Load training data from train.X* directories."""
        train_dirs = []
        
        if self.combine_train_splits:
            # Find all train.X* directories
            train_pattern = str(self.root / "train.X*")
            train_dirs = glob.glob(train_pattern)
            train_dirs.sort()
        else:
            # Use only train.X1 by default
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
                
                # Get class name from synset ID
                class_name = self.synset_to_class.get(synset_id, synset_id)
                
                # Find all image files in this synset directory
                for img_path in synset_dir.iterdir():
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        self.samples.append((str(img_path), class_name, synset_id))
                        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over dataset returning image tensors."""
        if not self._downloaded:
            self.download()
            
        for img_path, class_name, synset_id in self.samples:
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                yield image
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
                
    def __len__(self) -> int:
        """Return dataset size."""
        if not self._downloaded:
            self.download()
        return len(self.samples)
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "num_samples": len(self.samples),
            "num_classes": len(self.class_names),
            "class_names": self.class_names[:10],  # First 10 for brevity
            "image_shape": "(3, 224, 224)",  # After transform
            "split": self.split,
            "combine_train_splits": self.combine_train_splits
        })
        return metadata
