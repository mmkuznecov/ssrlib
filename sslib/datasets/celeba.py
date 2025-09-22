import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from typing import Iterator, Dict, Any, Optional, Tuple, List, Union
import requests
from pathlib import Path

from ..core.base import BaseDataset


class CelebADataset(BaseDataset):
    """CelebA Dataset for SSLib framework."""
    
    def __init__(self, root: str = "data", split: str = "train", task_name: str = "Attractive", 
                 transform: Optional[transforms.Compose] = None, **kwargs):
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
            
        # Define expected file paths
        self.split_csv = self.root / "list_eval_partition.csv"
        self.attr_csv = self.root / "list_attr_celeba.csv"
        self.images_dir = self.root / "img_align_celeba/img_align_celeba"
        
        # Will be loaded after download/check
        self.data = None
        self.attr_names = None
        
        # Update metadata
        self._metadata.update({
            "split": split,
            "task_name": task_name,
            "root": str(self.root)
        })
        
        # Check if dataset exists, download if not
        if not self._check_dataset():
            print(f"Downloading CelebA to {self.root}")
            self._download()
        
        # Load the data
        self._load_data()
        self._downloaded = True
        
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
        import zipfile
        import shutil
        
        # Create directories
        self.root.mkdir(parents=True, exist_ok=True)
        
        kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset"
        zip_path = self.root / "celeba_dataset.zip"
        
        try:
            print("Downloading CelebA dataset from Kaggle...")
            
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
            
            # Verify expected structure exists and reorganize if needed
            if not self.images_dir.exists():
                # Sometimes the structure might be nested, try to find the images
                for item in self.root.rglob("img_align_celeba"):
                    if item.is_dir():
                        print(f"Found images directory at {item}, moving to expected location...")
                        shutil.move(str(item), str(self.images_dir))
                        break
                else:
                    raise FileNotFoundError("Could not find img_align_celeba directory after extraction")
            
            # Look for CSV files and move them to root if needed
            for csv_name in ["list_eval_partition.csv", "list_attr_celeba.csv"]:
                expected_path = self.root / csv_name
                if not expected_path.exists():
                    # Search for the file
                    for found_file in self.root.rglob(csv_name):
                        print(f"Found {csv_name} at {found_file}, moving to {expected_path}")
                        shutil.move(str(found_file), str(expected_path))
                        break
                    else:
                        raise FileNotFoundError(f"Could not find {csv_name} after extraction")
            
            print("Dataset structure organized successfully")
            
        except requests.exceptions.RequestException as e:
            print(f"\nError downloading dataset: {e}")
            print("Please check your internet connection and Kaggle API credentials.")
            print("Manual download instructions:")
            print(f"1. Go to https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
            print(f"2. Download the dataset manually to {self.root}")
            print("3. Extract and organize with the following structure:")
            print("   - list_eval_partition.csv")
            print("   - list_attr_celeba.csv") 
            print("   - img_align_celeba/ (directory with images)")
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
            raise ValueError(f"Unknown task {self.task_name}. Available: {self.attr_names}")
            
        # Merge data
        self.data = pd.merge(
            split_df,
            attr_df[["image_id", self.task_name]],
            on="image_id",
            how="left",
        )
        
    def __getitem__(self, idx: Union[int, slice]) -> Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get item(s) by index.
        
        Args:
            idx: Index or slice
            
        Returns:
            Single tuple (image, target) or list of tuples for slice
        """
        if self.data is None:
            self.download()
            
        if isinstance(idx, slice):
            # Handle slice
            indices = range(*idx.indices(len(self.data)))
            return [self._get_single_item(i) for i in indices]
        else:
            # Handle single index
            return self._get_single_item(idx)
    
    def _get_single_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item by index."""
        if idx >= len(self.data) or idx < -len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
            
        # Handle negative indexing
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
        return (f"CelebADataset(split='{self.split}', task='{self.task_name}', "
                f"size={len(self) if self.data is not None else 'Unknown'}, "
                f"root='{self.root}')")
        
    def get_classes(self) -> Dict[str, Any]:
        """Get class information for the current task."""
        return {
            "task_name": self.task_name,
            "num_classes": 2,
            "class_names": ["No", "Yes"],
            "class_to_idx": {"No": 0, "Yes": 1}
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
            "exists": img_path.exists()
        }
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        metadata = super().get_metadata()
        if self.data is not None:
            metadata.update({
                "num_samples": len(self.data),
                "image_shape": "(3, 224, 224)",  # After transform
                "split": self.split,
                "task_name": self.task_name,
                "num_attributes": len(self.attr_names) if self.attr_names else 0
            })
        return metadata