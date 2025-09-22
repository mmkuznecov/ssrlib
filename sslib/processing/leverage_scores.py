import numpy as np
from typing import Dict, Any, Optional
from ..core.base import BaseProcessor


class LeverageScoresProcessor(BaseProcessor):
    """
    Computes row leverage scores diag(U_k U_k^T) from a rank-k SVD of centered X.
    Sum of scores equals k. Useful for landmark selection, importance sampling,
    and spotting outliers/anomalies.

    If rank is None, chooses the smallest k giving the desired energy threshold.
    """

    def __init__(self, rank: Optional[int] = None, energy: float = 0.9, center: bool = True, **kwargs):
        """
        Args:
            rank: target rank k (1..min(n,d)). If None, choose via 'energy'.
            energy: fraction of spectral energy (sum s_i^2) to retain when rank=None.
            center: mean-center before SVD (recommended).
        """
        super().__init__("LeverageScores", **kwargs)
        if rank is not None and rank <= 0:
            raise ValueError("rank must be positive when provided.")
        if not (0.0 < energy <= 1.0):
            raise ValueError("energy must be in (0, 1].")

        self.rank = rank
        self.energy = float(energy)
        self.center = bool(center)

        self._metadata.update({
            "processor_type": "leverage_scores",
            "rank": self.rank,
            "energy": self.energy,
            "center": self.center,
            "output_type": "row_scores"
        })

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        X = embeddings.astype(np.float64, copy=False)
        if self.center:
            X = X - X.mean(axis=0, keepdims=True)

        # Full SVD is simple & robust for moderate sizes.
        # Swap in a randomized SVD if you need scalability later.
        U, s, _ = np.linalg.svd(X, full_matrices=False)

        if U.size == 0:
            scores = np.zeros((X.shape[0],), dtype=np.float64)
            chosen_k = 0
        else:
            if self.rank is None:
                # Choose k for desired energy in s^2
                energy_spectrum = s ** 2
                cum = np.cumsum(energy_spectrum)
                total = cum[-1] if cum.size > 0 else 0.0
                if total <= 0:
                    chosen_k = 0
                else:
                    k = int(np.searchsorted(cum, self.energy * total) + 1)
                    chosen_k = max(1, min(k, U.shape[1]))
            else:
                chosen_k = max(1, min(self.rank, U.shape[1]))

            Uk = U[:, :chosen_k] if chosen_k > 0 else np.zeros((X.shape[0], 0), dtype=np.float64)
            scores = np.sum(Uk * Uk, axis=1)  # diag(Uk Uk^T)

        self._metadata.update({
            "input_shape": embeddings.shape,
            "n_vectors": int(embeddings.shape[0]),
            "n_features": int(embeddings.shape[1]),
            "chosen_rank": int(self._metadata.get("rank") or chosen_k),
            "scores_sum": float(scores.sum()),
            "scores_min": float(scores.min() if scores.size else 0.0),
            "scores_max": float(scores.max() if scores.size else 0.0)
        })

        return scores
