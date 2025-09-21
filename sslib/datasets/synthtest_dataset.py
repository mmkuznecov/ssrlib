import torch
import numpy as np
from typing import Iterator, Dict, Any, Optional

from ..core.base import BaseDataset


class SynthTestDataset(BaseDataset):
    """Synthetic test dataset that generates random image-like tensors.
    
    This dataset is useful for testing pipelines without requiring real data downloads.
    It generates random tensors of shape (3, 224, 224) to simulate RGB images.
    """
    
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
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Generate random tensors.
        
        Yields:
            torch.Tensor: Random tensor of shape self.tensor_shape
        """
        # Set seed if provided for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        for i in range(self.tensors_num):
            # Generate random tensor with values in reasonable range for images
            # Using normal distribution with mean=0, std=1, then clamp to [-2, 2]
            # This simulates normalized image data
            tensor = torch.randn(*self.tensor_shape)
            tensor = torch.clamp(tensor, -2.0, 2.0)
            yield tensor
            
    def __len__(self) -> int:
        """Return number of tensors in dataset."""
        return self.tensors_num
        
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
        
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible generation.
        
        Args:
            seed: Random seed value
        """
        self.seed = seed
        self._metadata["seed"] = seed
        
    def regenerate(self) -> None:
        """Force regeneration of data (useful for testing different random states)."""
        # This method doesn't actually store data, so regeneration happens
        # automatically on next iteration. This is here for API completeness.
        pass


# Example usage and testing
def test_synth_dataset():
    """Test function to demonstrate SynthTestDataset usage."""
    print("Testing SynthTestDataset...")
    
    # Basic usage
    dataset = SynthTestDataset(tensors_num=10)
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset metadata: {dataset.get_metadata()}")
    
    # Test iteration
    tensors = list(dataset)
    print(f"Generated {len(tensors)} tensors")
    print(f"First tensor shape: {tensors[0].shape}")
    print(f"First tensor range: [{tensors[0].min():.3f}, {tensors[0].max():.3f}]")
    
    # Test with seed for reproducibility
    dataset1 = SynthTestDataset(tensors_num=5, seed=42)
    dataset2 = SynthTestDataset(tensors_num=5, seed=42)
    
    tensors1 = list(dataset1)
    tensors2 = list(dataset2)
    
    # Should be identical with same seed
    are_equal = all(torch.equal(t1, t2) for t1, t2 in zip(tensors1, tensors2))
    print(f"Tensors identical with same seed: {are_equal}")
    
    # Test custom shape
    custom_dataset = SynthTestDataset(tensors_num=3, tensor_shape=(1, 64, 64))
    custom_tensors = list(custom_dataset)
    print(f"Custom shape tensor: {custom_tensors[0].shape}")
    
    print("SynthTestDataset test completed!")


if __name__ == "__main__":
    test_synth_dataset()