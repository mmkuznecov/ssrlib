"""Eigenvalue spectrum of the covariance matrix."""

from __future__ import annotations

import numpy as np

from ._spectral import covariance_eigvals
from .base import BaseProcessor


class SpectrumProcessor(BaseProcessor):
    """Eigenvalue spectrum of the (optionally centered) covariance matrix.

    By default returns raw eigenvalues sorted in descending order. Set
    ``normalize=True`` to return explained-variance ratios instead.
    """

    def __init__(
        self,
        center: bool = True,
        epsilon: float = 1e-12,
        normalize: bool = False,
        **kwargs,
    ):
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

        eig = covariance_eigvals(embeddings, center=self.center)

        trace = float(eig.sum())
        top = float(eig[0]) if eig.size else 0.0

        if self.normalize and trace > self.epsilon:
            spectrum = eig / (trace + self.epsilon)
        else:
            spectrum = eig

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_vectors": int(embeddings.shape[0]),
                "n_features": int(embeddings.shape[1]),
                "trace": trace,
                "spectral_norm": top,
                "n_eigenvalues": int(eig.size),
            }
        )
        return spectrum.astype(np.float64, copy=False)
