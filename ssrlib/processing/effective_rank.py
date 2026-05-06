"""Effective rank: exp(H(p)), the soft, scale-invariant dimensionality proxy."""

from __future__ import annotations

import numpy as np

from ._spectral import covariance_eigvals
from .base import BaseProcessor


class EffectiveRankProcessor(BaseProcessor):
    """Effective rank of the (centered) covariance matrix.

    ``erank = exp(-Σ p_i log p_i)``, where ``p_i = λ_i / Σ_j λ_j``.

    Reference: Roy & Vetterli, "The effective rank: a measure of effective
    dimensionality" (2007).

    Returns:
        shape-(1,) array containing the scalar effective rank.
    """

    def __init__(self, epsilon: float = 1e-12, center: bool = True, **kwargs):
        super().__init__("EffectiveRank", **kwargs)
        self.epsilon = float(epsilon)
        self.center = bool(center)

        self._metadata.update(
            {
                "processor_type": "effective_rank",
                "epsilon": self.epsilon,
                "center": self.center,
                "output_type": "scalar_statistic",
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        eig = covariance_eigvals(embeddings, center=self.center)
        total = eig.sum()

        if not np.isfinite(total) or total <= self.epsilon:
            erank = 0.0
        else:
            p = eig / (total + self.epsilon)
            p = np.clip(p, self.epsilon, 1.0)  # avoid log(0)
            entropy = -np.sum(p * np.log(p))
            erank = float(np.exp(entropy))

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_vectors": int(embeddings.shape[0]),
                "n_features": int(embeddings.shape[1]),
                "value": erank,
            }
        )
        return np.array([erank], dtype=np.float64)
