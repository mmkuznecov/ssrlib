import os
import json
import glob
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Tuple, Union
import torch
from PIL import Image
from torchvision import transforms
import requests

from ..core.base import BaseDataset


class ImageNet100Dataset(BaseDataset):
    """ImageNet100 Dataset for SSLib framework."""
    
    def __init__(self, root: str = "data", split: str = "train", 
                 labels_path: Optional[str] = None,
                 combine_train_splits: bool = True,
                 transform: Optional[transforms.Compose] = None, **kwargs):
        """Initialize ImageNet100 dataset."""
        super().__init__("ImageNet100", **kwargs)
        
        # Set root path with ImageNet100 subdirectory
        self.root = Path(root) / "ImageNet100"
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
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Update metadata
        self._metadata.update({
            "split": split,
            "combine_train_splits": combine_train_splits,
            "root": str(self.root)
        })
        
        # Check if dataset exists, download if not
        if not self._check_dataset():
            print(f"Downloading ImageNet100 to {self.root}")
            self._download()
        
        # Load the data
        self._load_data()
        self._downloaded = True
        
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
            # Check if there are actual image files
            for pattern in ["train.X*", "val.X"]:
                for dir_path in glob.glob(str(self.root / pattern)):
                    dir_path = Path(dir_path)
                    if dir_path.is_dir():
                        # Look for subdirectories with images
                        for subdir in dir_path.iterdir():
                            if subdir.is_dir():
                                image_files = [f for f in subdir.iterdir() 
                                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
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
        import zipfile
        import shutil
        
        # Create directories
        self.root.mkdir(parents=True, exist_ok=True)
        
        kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/ambityga/imagenet100"
        zip_path = self.root / "imagenet100.zip"
        
        try:
            print("Downloading ImageNet100 dataset from Kaggle...")
            print("Note: This requires Kaggle API authentication.")
            print("Please ensure you have ~/.kaggle/kaggle.json with your API credentials.")
            
            # Download with requests and show progress
            response = requests.get(kaggle_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                if total_size > 0:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            print(f"\nDownload completed: {zip_path}")
            
            # Extract the zip file
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            
            print("Extraction completed")
            
            # Clean up zip file
            zip_path.unlink()
            print("Cleaned up zip file")
            
            # Organize file structure if needed
            self._organize_extracted_files()
            
            print("Dataset structure organized successfully")
            
        except requests.exceptions.RequestException as e:
            print(f"\nError downloading dataset: {e}")
            print("Please check your internet connection and Kaggle API credentials.")
            print("Manual download instructions:")
            print(f"1. Go to https://www.kaggle.com/datasets/ambityga/imagenet100")
            print(f"2. Download the dataset manually to {self.root}")
            print("3. Extract and organize with the following structure:")
            print("   - train.X1/, train.X2/, ... (training directories)")
            print("   - val.X/ (validation directory)")
            print("   - Labels.json (optional, for human-readable class names)")
            raise
            
        except zipfile.BadZipFile as e:
            print(f"\nError extracting zip file: {e}")
            if zip_path.exists():
                zip_path.unlink()
            raise
            
        except Exception as e:
            print(f"\nUnexpected error during download: {e}")
            # Clean up partial download
            if zip_path.exists():
                zip_path.unlink()
            raise
    
    def _organize_extracted_files(self):
        """Organize extracted files into expected structure."""
        import shutil
        
        # Look for common extraction patterns and reorganize if needed
        
        # Check if files were extracted to a subdirectory
        subdirs = [d for d in self.root.iterdir() if d.is_dir() and d.name != '__MACOSX']
        
        # If there's only one subdirectory and it contains the dataset, move contents up
        if len(subdirs) == 1:
            subdir = subdirs[0]
            # Check if this subdirectory contains train/val directories
            subdir_contents = list(subdir.iterdir())
            train_dirs_in_subdir = [d for d in subdir_contents if d.is_dir() and d.name.startswith('train.X')]
            val_dirs_in_subdir = [d for d in subdir_contents if d.is_dir() and d.name.startswith('val.X')]
            
            if train_dirs_in_subdir or val_dirs_in_subdir:
                print(f"Moving contents from {subdir.name} to root directory...")
                for item in subdir_contents:
                    target = self.root / item.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(target))
                # Remove empty subdirectory
                subdir.rmdir()
        
        # Verify expected directories exist
        train_dirs = glob.glob(str(self.root / "train.X*"))
        val_dirs = glob.glob(str(self.root / "val.X*"))
        
        if not train_dirs:
            # Look for alternative training directory names
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
        
        if not val_dirs:
            # Look for alternative validation directory names
            alt_val_dirs = []
            for pattern in ["val*", "Val*", "VAL*", "validation*", "valid*"]:
                alt_val_dirs.extend(glob.glob(str(self.root / pattern)))
            
            if alt_val_dirs:
                print("Found alternative validation directory, renaming...")
                alt_dir = alt_val_dirs[0]  # Take first one
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
        
        print("File organization completed")
        
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
        elif self.labels_path is None:
            # Auto-detect labels path
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
    
    def _create_class_mappings(self):
        """Create class to index mappings."""
        if self.class_names:
            # Use human-readable class names if available
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        else:
            # Fall back to synset IDs
            unique_classes = sorted(list(set([sample[1] for sample in self.samples])))
            self.class_names = unique_classes
            self.class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
        
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get item(s) by index.
        
        Args:
            idx: Index or slice
            
        Returns:
            Single tuple (image, class_index) or list of tuples for slice
        """
        if not self._downloaded:
            self.download()
            
        if isinstance(idx, slice):
            # Handle slice
            indices = range(*idx.indices(len(self.samples)))
            return [self._get_single_item(i) for i in indices]
        else:
            # Handle single index
            return self._get_single_item(idx)
    
    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item by index."""
        if idx >= len(self.samples) or idx < -len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
            
        # Handle negative indexing
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
        return (f"ImageNet100Dataset(split='{self.split}', "
                f"size={len(self.samples) if self.samples else 'Unknown'}, "
                f"num_classes={len(self.class_names)}, root='{self.root}')")
        
    def get_classes(self) -> Dict[str, Any]:
        """Get class information."""
        return {
            "num_classes": len(self.class_names),
            "class_names": self.class_names.copy(),
            "class_to_idx": self.class_to_idx.copy(),
            "idx_to_class": self.idx_to_class.copy(),
            "synset_to_class": self.synset_to_class.copy() if self.synset_to_class else {}
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
            "exists": os.path.exists(img_path)
        }
        
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