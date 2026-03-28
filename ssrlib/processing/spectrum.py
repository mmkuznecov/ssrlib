import numpy as np
from typing import Dict, Any
from .base import BaseProcessor


class SpectrumProcessor(BaseProcessor):
    """
    Computes the eigenvalue spectrum of the (optionally centered) covariance
    matrix of embeddings.

    By default returns raw eigenvalues λ_i sorted in descending order.
    Optionally can also return normalized spectrum (explained variance ratios).
    """

    def __init__(
        self,
        center: bool = True,
        epsilon: float = 1e-12,
        normalize: bool = False,
        **kwargs,
    ):
        """
        Args:
            center: mean-center before covariance.
            epsilon: small floor for eigenvalues to avoid numerical issues.
            normalize: if True, return explained-variance ratios λ_i / sum_j λ_j
                       instead of raw eigenvalues.
        """
        super().__init__("Spectrum", **kwargs)
        self.center = bool(center)
        self.epsilon = float(epsilon)
        self.normalize = bool(normalize)

        self._metadata.update(
            {
                "processor_type": "spectrum",
                "center": self.center,
                "epsilon": self.epsilon,
                "normalize": self.normalize,
                "output_type": "eigenvalue_spectrum",
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        X = embeddings.astype(np.float64, copy=False)
        if self.center:
            X = X - X.mean(axis=0, keepdims=True)

        # Covariance and eigenvalues
        C = np.cov(X.T)
        evals = np.linalg.eigvalsh(C)  # ascending
        evals = np.maximum(evals, 0.0)

        # Sort descending for convenience
        evals = evals[::-1]

        trace = float(evals.sum())
        top = float(evals[0]) if evals.size > 0 else 0.0

        if self.normalize and trace > self.epsilon:
            spectrum = evals / (trace + self.epsilon)
        else:
            spectrum = evals

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_vectors": int(embeddings.shape[0]),
                "n_features": int(embeddings.shape[1]),
                "trace": trace,
                "spectral_norm": top,
                "n_eigenvalues": int(evals.size),
            }
        )

        return spectrum.astype(np.float64, copy=False)
