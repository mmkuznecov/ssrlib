"""ZCA whitening of embeddings."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from ._spectral import covariance_matrix
from .base import BaseProcessor


class ZCAProcessor(BaseProcessor):
    """Apply ZCA whitening: ``Y = (X - mean) @ U @ diag(1/sqrt(lambda + eps)) @ U^T``.

    Output is a whitened embedding matrix of shape (N, D) whose covariance is
    approximately the identity matrix.

    Args:
        epsilon: regularization added inside the inverse-square-root for
            numerical stability with near-zero eigenvalues.
    """

    def __init__(self, epsilon: float = 1e-9, **kwargs):
        super().__init__("ZCA", **kwargs)
        self.epsilon = float(epsilon)
        self._metadata.update(
            {
                "processor_type": "zca_whitening",
                "epsilon": self.epsilon,
                "output_type": "whitened_embeddings",
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        mean = embeddings.mean(axis=0, keepdims=True)
        centered = embeddings - mean

        cov = covariance_matrix(centered)
        eigenvalues, eigenvectors = eigh(cov)

        # eigh returns ascending; sort descending for diagnostics
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clip negative noise (eigh of cov may produce -1e-16 type values)
        eigenvalues_clipped = np.maximum(eigenvalues, 0.0)

        d_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_clipped + self.epsilon))
        zca_matrix = eigenvectors @ d_inv_sqrt @ eigenvectors.T

        whitened = centered @ zca_matrix.T

        # Safe condition number: guard the denominator so we never divide by
        # zero or by a negative value that slipped through.
        denom = max(eigenvalues_clipped[-1], self.epsilon)
        condition_number = float(eigenvalues_clipped[0] / denom)

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "output_shape": whitened.shape,
                "n_vectors": int(embeddings.shape[0]),
                "n_features": int(embeddings.shape[1]),
                "mean_eigenvalue": float(eigenvalues_clipped.mean()),
                "max_eigenvalue": float(eigenvalues_clipped[0]),
                "min_eigenvalue": float(eigenvalues_clipped[-1]),
                "condition_number": condition_number,
            }
        )
        return whitened
