from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any
import torch
import numpy as np
from tqdm import tqdm


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


class BaseEmbedder(ABC):
    """Base class for all embedders in SSLib."""
    
    def __init__(self, name: str, device: str = "cpu", **kwargs):
        """Initialize embedder.
        
        Args:
            name: Name of the embedder
            device: Device to run on ('cpu' or 'cuda')
            **kwargs: Additional embedder-specific parameters
        """
        self.name = name
        self.device = device
        self.model = None
        self._loaded = False
        self._metadata = {}
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model."""
        pass
        
    @abstractmethod
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            batch: Input batch of tensors
            
        Returns:
            Embeddings tensor
        """
        pass
        
    def embed_dataset(self, dataset: BaseDataset, batch_size: int = 32) -> np.ndarray:
        """Extract embeddings from entire dataset.
        
        Args:
            dataset: Dataset to embed
            batch_size: Batch size for processing
            
        Returns:
            Embeddings array of shape (n_vectors, n_features)
        """
        if not self._loaded:
            self.load_model()
            
        all_embeddings = []
        current_batch = []
        
        for sample in tqdm(dataset, desc=f"Embedding with {self.name}"):
            current_batch.append(sample)
            
            if len(current_batch) == batch_size:
                batch_tensor = torch.stack(current_batch).to(self.device)
                with torch.no_grad():
                    embeddings = self.forward(batch_tensor)
                all_embeddings.append(embeddings.cpu().numpy())
                current_batch = []
                
        # Process remaining samples
        if current_batch:
            batch_tensor = torch.stack(current_batch).to(self.device)
            with torch.no_grad():
                embeddings = self.forward(batch_tensor)
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.concatenate(all_embeddings, axis=0)
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get embedder metadata."""
        return {
            "name": self.name,
            "device": self.device,
            "loaded": self._loaded,
            **self._metadata
        }


class BaseProcessor(ABC):
    """Base class for all processors in SSLib."""
    
    def __init__(self, name: str, **kwargs):
        """Initialize processor.
        
        Args:
            name: Name of the processor
            **kwargs: Additional processor-specific parameters
        """
        self.name = name
        self._metadata = {}
        
    @abstractmethod
    def process(self, embeddings: np.ndarray) -> np.ndarray:
        """Process embeddings.
        
        Args:
            embeddings: Input embeddings of shape (n_vectors, n_features)
            
        Returns:
            Processed embeddings or computed features
        """
        pass
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get processor metadata."""
        return {
            "name": self.name,
            **self._metadata
        }
