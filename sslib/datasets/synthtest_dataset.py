import torch
import numpy as np
from typing import Iterator, Dict, Any, Optional, Union, List

from ..core.base import BaseDataset


class SynthTestDataset(BaseDataset):
    """Synthetic test dataset that generates random image-like tensors."""
    
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
        """Get item(s) by index.
        
        Args:
            idx: Index or slice
            
        Returns:
            Single tensor or list of tensors for slice
        """
        if isinstance(idx, slice):
            # Handle slice
            indices = range(*idx.indices(self.tensors_num))
            return [self._get_single_item(i) for i in indices]
        else:
            # Handle single index
            return self._get_single_item(idx)
    
    def _get_single_item(self, idx: int) -> torch.Tensor:
        """Get a single tensor by index."""
        if idx >= self.tensors_num or idx < -self.tensors_num:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.tensors_num}")
            
        # Handle negative indexing
        if idx < 0:
            idx = self.tensors_num + idx
        
        # Generate deterministic tensor based on index and seed
        if self.seed is not None:
            # Create deterministic tensor using index-specific seed
            generator = torch.Generator()
            generator.manual_seed(self.seed + idx)
            tensor = torch.randn(*self.tensor_shape, generator=generator)
        else:
            # Use current random state
            tensor = torch.randn(*self.tensor_shape)
        
        # Clamp to reasonable range
        tensor = torch.clamp(tensor, -2.0, 2.0)
        return tensor
        
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
        """Force regeneration with new seed.
        
        Args:
            new_seed: New seed to use (None for random)
        """
        if new_seed is not None:
            self.seed = new_seed
            self._metadata["seed"] = new_seed
        else:
            self.seed = None
            self._metadata["seed"] = None
        
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible generation.
        
        Args:
            seed: Random seed value
        """
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


## Example usage and testing

def test_dataset_functionality():
    """Test the enhanced dataset functionality."""
    
    print("Testing CelebA Dataset...")
    # Note: This would require actual CelebA data to work
    try:
        celeba = CelebADataset(root="./test_data", split="train")
        print(f"CelebA: {celeba}")
        print(f"Classes: {celeba.get_classes()}")
        print(f"Attributes: {celeba.get_all_attributes()[:5]}")  # First 5
        
        # Test indexing (if data exists)
        if len(celeba) > 0:
            image, target = celeba[0]
            print(f"First sample: image {image.shape}, target {target}")
            sample_info = celeba.get_sample_info(0)
            print(f"Sample info: {sample_info}")
    except Exception as e:
        print(f"CelebA test skipped: {e}")
    
    print("\nTesting ImageNet100 Dataset...")
    try:
        imagenet = ImageNet100Dataset(root="./test_data", split="train")
        print(f"ImageNet100: {imagenet}")
        if len(imagenet) > 0:
            image, target = imagenet[0]
            print(f"First sample: image {image.shape}, target {target}")
            print(f"Class info: {imagenet.get_classes()}")
    except Exception as e:
        print(f"ImageNet100 test skipped: {e}")
    
    print("\nTesting SynthTest Dataset...")
    synth = SynthTestDataset(tensors_num=10, seed=42)
    print(f"SynthTest: {synth}")
    
    # Test indexing
    tensor = synth[0]
    print(f"First tensor shape: {tensor.shape}")
    
    # Test slicing
    batch = synth[0:3]
    print(f"Batch of 3: {len(batch)} tensors")
    
    # Test negative indexing
    last_tensor = synth[-1]
    print(f"Last tensor shape: {last_tensor.shape}")
    
    # Test sample info
    info = synth.get_sample_info(0)
    print(f"Sample info: {info}")
    
    # Test reproducibility
    synth2 = SynthTestDataset(tensors_num=10, seed=42)
    assert torch.equal(synth[0], synth2[0]), "Should be identical with same seed"
    print("Reproducibility test passed!")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    test_dataset_functionality()