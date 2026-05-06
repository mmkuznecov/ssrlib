"""Row leverage scores from rank-k SVD."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._spectral import centered
from .base import BaseProcessor


class LeverageScoresProcessor(BaseProcessor):
    """Row leverage scores ``diag(U_k U_k^T)`` from a rank-k SVD of (centered) X.

    Sum of scores equals k. Useful for landmark selection, importance sampling,
    and outlier detection.

    Args:
        rank: target rank k in [1, min(N, D)]. If None, choose the smallest k
            such that the cumulative spectral energy reaches ``energy``.
        energy: fraction of spectral energy to retain when ``rank`` is None.
        center: mean-center X before computing.
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        energy: float = 0.9,
        center: bool = True,
        **kwargs,
    ):
        super().__init__("LeverageScores", **kwargs)
        if rank is not None and rank <= 0:
            raise ValueError("rank must be positive when provided.")
        if not (0.0 < energy <= 1.0):
            raise ValueError("energy must be in (0, 1].")

        self.rank = rank
        self.energy = float(energy)
        self.center = bool(center)

        self._metadata.update(
            {
                "processor_type": "leverage_scores",
                "rank": self.rank,
                "energy": self.energy,
                "center": self.center,
                "output_type": "row_scores",
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        X = embeddings.astype(np.float64, copy=False)
        if self.center:
            X = centered(X)

        U, s, _ = np.linalg.svd(X, full_matrices=False)

        if U.size == 0:
            scores = np.zeros((X.shape[0],), dtype=np.float64)
            chosen_k = 0
        else:
            if self.rank is None:
                energy_spectrum = s**2
                cum = np.cumsum(energy_spectrum)
                total = cum[-1] if cum.size > 0 else 0.0
                if total <= 0:
                    chosen_k = 0
                else:
                    k = int(np.searchsorted(cum, self.energy * total) + 1)
                    chosen_k = max(1, min(k, U.shape[1]))
            else:
                chosen_k = max(1, min(self.rank, U.shape[1]))

            if chosen_k == 0:
                scores = np.zeros((X.shape[0],), dtype=np.float64)
            else:
                Uk = U[:, :chosen_k]
                scores = np.sum(Uk * Uk, axis=1)  # diag(Uk Uk^T)

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_vectors": int(embeddings.shape[0]),
                "n_features": int(embeddings.shape[1]),
                "chosen_rank": int(chosen_k),
                "scores_sum": float(scores.sum()),
                "scores_min": float(scores.min()) if scores.size else 0.0,
                "scores_max": float(scores.max()) if scores.size else 0.0,
            }
        )
        return scores
