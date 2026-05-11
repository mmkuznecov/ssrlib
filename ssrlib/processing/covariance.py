"""Empirical covariance matrix of embeddings."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._spectral import covariance_matrix
from .base import BaseProcessor
from .map_reduce import MapReduceMixin


class CovarianceProcessor(BaseProcessor, MapReduceMixin):
    """Compute the empirical feature covariance matrix.

    Whole-array path uses ``np.cov(X.T)`` (ddof=1).
    Streaming path accumulates ``sum_x``, ``sum_xxT``, ``n`` and finalizes
    ``cov = (sum_xxT - n * mean ⊗ mean) / (n - 1)``, which is mathematically
    equivalent to ``np.cov`` with ddof=1.
    """

    def __init__(self, **kwargs):
        super().__init__("Covariance", **kwargs)
        self._metadata.update(
            {"processor_type": "covariance", "output_type": "covariance_matrix"}
        )
        self._reset_accumulators()

    # ------------------------------------------------------------------ batch
    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        cov = covariance_matrix(embeddings)
        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "output_shape": cov.shape,
                "n_vectors": int(embeddings.shape[0]),
                "n_features": int(embeddings.shape[1]),
            }
        )
        return cov

    # ----------------------------------------------------------- map-reduce
    def _reset_accumulators(self) -> None:
        self._n: int = 0
        self._sum_x: Optional[np.ndarray] = None
        self._sum_xxt: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._reset_accumulators()

    def partial_fit(self, batch: np.ndarray) -> None:
        if batch.ndim != 2:
            raise ValueError(f"Expected 2D batch, got shape {batch.shape}")

        b = batch.astype(np.float64, copy=False)
        if self._sum_x is None:
            d = b.shape[1]
            self._sum_x = np.zeros(d, dtype=np.float64)
            self._sum_xxt = np.zeros((d, d), dtype=np.float64)

        self._sum_x += b.sum(axis=0)
        self._sum_xxt += b.T @ b
        self._n += b.shape[0]

    def finalize(self) -> np.ndarray:
        if self._n < 2:
            raise RuntimeError(
                f"Covariance requires at least 2 samples; got {self._n}. "
                "Did you forget to call partial_fit?"
            )

        mean = self._sum_x / self._n
        # Unbiased (ddof=1) covariance matching np.cov
        cov = (self._sum_xxt - self._n * np.outer(mean, mean)) / (self._n - 1)
        # Symmetrize to remove numerical asymmetry from floating-point error
        cov = 0.5 * (cov + cov.T)

        self._metadata.update(
            {
                "output_shape": cov.shape,
                "n_vectors": int(self._n),
                "n_features": int(cov.shape[0]),
                "computed_via": "streaming",
            }
        )
        return cov
