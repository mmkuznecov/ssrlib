from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any
import torch


class BaseDataset(ABC):
    """Base class for all datasets in SSLib."""
    
    def __init__(self, name: str, **kwargs):
        """Initialize dataset.
        
        Args:
            name: Name of the dataset
            **kwargs: Additional dataset-specific parameters
        """
        self.name = name
        self._metadata = {}
        self._downloaded = False
        
    @abstractmethod
    def download(self) -> None:
        """Download dataset if not already present."""
        pass
        
    @abstractmethod
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over dataset returning tensors."""
        pass
        
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            "name": self.name,
            "size": len(self),
            "downloaded": self._downloaded,
            **self._metadata
        }