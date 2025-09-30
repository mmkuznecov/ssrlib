import numpy as np
from typing import Dict, Any
from scipy.linalg import eigh

from .base import BaseProcessor


class ZCAProcessor(BaseProcessor):
    """Processor for ZCA whitening of embeddings."""
    
    def __init__(self, epsilon: float = 1e-9, **kwargs):
        """Initialize ZCA processor.
        
        Args:
            epsilon: Regularization parameter for numerical stability
        """
        super().__init__("ZCA", **kwargs)
        self.epsilon = epsilon
        
        self._metadata.update({
            "processor_type": "zca_whitening",
            "epsilon": epsilon,
            "output_type": "whitened_embeddings"
        })
        
    def process(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply ZCA whitening to embeddings.
        
        Args:
            embeddings: Input embeddings of shape (n_vectors, n_features)
            
        Returns:
            ZCA whitened embeddings of shape (n_vectors, n_features)
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
            
        # Center the data
        mean = np.mean(embeddings, axis=0, keepdims=True)
        centered = embeddings - mean
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(cov)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # ZCA whitening matrix
        # W = U * D^(-1/2) * U^T where U are eigenvectors and D are eigenvalues
        d_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + self.epsilon))
        zca_matrix = eigenvectors @ d_inv_sqrt @ eigenvectors.T
        
        # Apply whitening
        whitened = centered @ zca_matrix.T
        
        # Update metadata
        self._metadata.update({
            "input_shape": embeddings.shape,
            "output_shape": whitened.shape,
            "n_vectors": embeddings.shape[0],
            "n_features": embeddings.shape[1],
            "mean_eigenvalue": float(np.mean(eigenvalues)),
            "condition_number": float(eigenvalues[0] / eigenvalues[-1])
        })
        
        return whitened
