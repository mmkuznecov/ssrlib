import numpy as np
from typing import Dict, Any

from .base import BaseProcessor


class CovarianceProcessor(BaseProcessor):
    """Processor for computing covariance matrix of embeddings."""
    
    def __init__(self, **kwargs):
        """Initialize covariance processor."""
        super().__init__("Covariance", **kwargs)
        
        self._metadata.update({
            "processor_type": "covariance",
            "output_type": "covariance_matrix"
        })
        
    def process(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute covariance matrix of embeddings.
        
        Args:
            embeddings: Input embeddings of shape (n_vectors, n_features)
            
        Returns:
            Covariance matrix of shape (n_features, n_features)
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
            
        # Compute feature covariance matrix
        # np.cov expects features in rows, so we transpose
        covariance_matrix = np.cov(embeddings.T)
        
        # Update metadata with shape info
        self._metadata.update({
            "input_shape": embeddings.shape,
            "output_shape": covariance_matrix.shape,
            "n_vectors": embeddings.shape[0],
            "n_features": embeddings.shape[1]
        })
        
        return covariance_matrix
