"""Base embedder implementation for SSLib."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, ClassVar
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Base class for all embedders in SSLib with self-describing metadata."""
    
    # Class-level metadata - subclasses should override these
    _embedder_category: ClassVar[str] = "general"
    _embedder_modality: ClassVar[str] = "unknown"
    _embedder_properties: ClassVar[Dict[str, Any]] = {}
    
    def __init__(self, name: str, device: str = "cpu", batch_size: int = 32, **kwargs):
        """Initialize embedder.
        
        Args:
            name: Model name
            device: Device to use ('cpu' or 'cuda')
            batch_size: Default batch size for processing
            **kwargs: Additional configuration
        """
        self.name = name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = None
        self._loaded = False
        self._metadata = {
            "category": self.get_embedder_category(),
            "modality": self.get_embedder_modality()
        }
        self._metadata.update(kwargs)
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the pretrained model."""
        pass
    
    @abstractmethod
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from a batch.
        
        Args:
            batch: Input tensor
            
        Returns:
            Embeddings tensor
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        pass
    
    def embed_dataset(self, dataset, batch_size: Optional[int] = None) -> np.ndarray:
        """Extract embeddings for entire dataset with batching.
        
        Args:
            dataset: Dataset to embed (iterable yielding tensors)
            batch_size: Batch size for processing (uses default if None)
            
        Returns:
            Embeddings array of shape (n_samples, embedding_dim)
        """
        if not self._loaded:
            self.load_model()
        
        batch_size = batch_size or self.batch_size
        embeddings = []
        current_batch = []
        
        logger.info(f"Extracting embeddings with {self.name} (batch_size={batch_size})")
        
        for i, sample in enumerate(dataset):
            current_batch.append(sample)
            
            if len(current_batch) == batch_size:
                batch_tensor = torch.stack(current_batch).to(self.device)
                batch_embeddings = self.forward(batch_tensor)
                embeddings.append(batch_embeddings.cpu().numpy())
                current_batch = []
                
                if (i + 1) % (batch_size * 10) == 0:
                    logger.info(f"  Processed {i + 1} samples")
        
        # Process remaining samples
        if current_batch:
            batch_tensor = torch.stack(current_batch).to(self.device)
            batch_embeddings = self.forward(batch_tensor)
            embeddings.append(batch_embeddings.cpu().numpy())
        
        result = np.concatenate(embeddings, axis=0)
        logger.info(f"  Extracted embeddings shape: {result.shape}")
        return result
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self._loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded model: {self.name}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get embedder metadata.
        
        Returns:
            Dictionary with metadata
        """
        return {
            "name": self.name,
            "device": str(self.device),
            "loaded": self._loaded,
            "embedding_dim": self.get_embedding_dim(),
            **self._metadata
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "name": self.name,
            "category": self.get_embedder_category(),
            "modality": self.get_embedder_modality(),
            "device": str(self.device),
            "is_loaded": self._loaded,
            "embedding_dim": self.get_embedding_dim(),
        }
        
        if self._loaded and self.model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            info["total_parameters"] = total_params
            info["trainable_parameters"] = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        
        return info
    
    @classmethod
    def get_embedder_category(cls) -> str:
        """Get embedder category."""
        return cls._embedder_category
    
    @classmethod
    def get_embedder_modality(cls) -> str:
        """Get embedder modality."""
        return cls._embedder_modality
    
    @classmethod
    def get_embedder_properties(cls) -> Dict[str, Any]:
        """Get embedder properties."""
        return cls._embedder_properties.copy()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_model()
        except:
            pass
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"device='{self.device}', loaded={self._loaded})")