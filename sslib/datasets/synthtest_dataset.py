import torch
import numpy as np
from typing import Iterator, Dict, Any, Optional, Union, List, ClassVar

from .base import BaseDataset


class SynthTestDataset(BaseDataset):
    """Synthetic test dataset that generates random image-like tensors."""
    
    # Class-level metadata
    _dataset_category: ClassVar[str] = "synthetic"
    _dataset_modality: ClassVar[str] = "synthetic"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "default_tensor_shape": (3, 224, 224),
        "default_num_tensors": 100,
        "deterministic": True,
        "task_type": "testing",
        "supports_custom_shapes": True,
        "value_range": (-2.0, 2.0)
    }
    
    def __init__(self, tensors_num: int = 100, seed: Optional[int] = None, 
                 tensor_shape: tuple = (3, 224, 224), **kwargs):
        """Initialize synthetic test dataset.
        
        Args:
            tensors_num: Number of tensors to generate
            seed: Random seed for reproducibility (optional)
            tensor_shape: Shape of generated tensors (default: (3, 224, 224))
            **kwargs: Additional arguments passed to BaseDataset
        """
        super().__init__("SynthTest", **kwargs)
        
        self.tensors_num = tensors_num
        self.seed = seed
        self.tensor_shape = tensor_shape
        
        # Validate inputs
        if tensors_num <= 0:
            raise ValueError(f"tensors_num must be positive, got {tensors_num}")
        
        if len(tensor_shape) != 3:
            raise ValueError(f"tensor_shape must have 3 dimensions, got {len(tensor_shape)}")
        
        # Update metadata
        self._metadata.update({
            "tensors_num": tensors_num,
            "tensor_shape": tensor_shape,
            "seed": seed,
            "synthetic": True,
            "dataset_type": "synthetic_test"
        })
        
        # Mark as already "downloaded" since no actual download is needed
        self._downloaded = True
        
    def download(self) -> None:
        """No-op download method for synthetic data."""
        if not self._downloaded:
            print(f"Synthetic dataset {self.name} ready (no download needed)")
            self._downloaded = True
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Get item(s) by index."""
        if isinstance(idx, slice):
            indices = range(*idx.indices(self.tensors_num))
            return [self._get_single_item(i) for i in indices]
        else:
            return self._get_single_item(idx)
    
    def _get_single_item(self, idx: int) -> torch.Tensor:
        """Get a single tensor by index."""
        if idx >= self.tensors_num or idx < -self.tensors_num:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.tensors_num}")
            
        if idx < 0:
            idx = self.tensors_num + idx
        
        # Generate deterministic tensor based on index and seed
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + idx)
            tensor = torch.randn(*self.tensor_shape, generator=generator)
        else:
            tensor = torch.randn(*self.tensor_shape)
        
        # Clamp to reasonable range
        tensor = torch.clamp(tensor, -2.0, 2.0)
        return tensor
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Generate random tensors."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        for i in range(self.tensors_num):
            tensor = torch.randn(*self.tensor_shape)
            tensor = torch.clamp(tensor, -2.0, 2.0)
            yield tensor
            
    def __len__(self) -> int:
        """Return number of tensors in dataset."""
        return self.tensors_num
    
    def __repr__(self) -> str:
        """String representation of dataset."""
        return (f"SynthTestDataset(size={self.tensors_num}, "
                f"shape={self.tensor_shape}, seed={self.seed})")
        
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific sample."""
        if idx >= self.tensors_num or idx < -self.tensors_num:
            raise IndexError(f"Index {idx} out of range")
            
        if idx < 0:
            idx = self.tensors_num + idx
            
        return {
            "index": idx,
            "tensor_shape": self.tensor_shape,
            "seed_used": self.seed,
            "deterministic": self.seed is not None,
            "synthetic": True
        }
        
    def regenerate(self, new_seed: Optional[int] = None) -> None:
        """Force regeneration with new seed."""
        if new_seed is not None:
            self.seed = new_seed
            self._metadata["seed"] = new_seed
        else:
            self.seed = None
            self._metadata["seed"] = None
        
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible generation."""
        self.seed = seed
        self._metadata["seed"] = seed
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "num_samples": self.tensors_num,
            "image_shape": f"{self.tensor_shape}",
            "synthetic": True,
            "deterministic": self.seed is not None,
            "seed_used": self.seed
        })
        return metadata