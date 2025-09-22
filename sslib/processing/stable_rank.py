import numpy as np
from typing import Dict, Any
from ..core.base import BaseProcessor


class StableRankProcessor(BaseProcessor):
    """
    Stable rank of the (optionally centered) data matrix X:
        srank = ||X||_F^2 / ||X||_2^2
    where ||X||_2 is the top singular value.
    """

    def __init__(self, center: bool = True, epsilon: float = 1e-12, **kwargs):
        """
        Args:
            center: mean-center rows before computing norms (often desirable).
            epsilon: small floor for top singular value squared.
        """
        super().__init__("StableRank", **kwargs)
        self.center = bool(center)
        self.epsilon = float(epsilon)

        self._metadata.update({
            "processor_type": "stable_rank",
            "center": self.center,
            "epsilon": self.epsilon,
            "output_type": "scalar_statistic"
        })

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        X = embeddings.astype(np.float64, copy=False)
        if self.center:
            X = X - X.mean(axis=0, keepdims=True)

        fro2 = float(np.sum(X * X))
        # Top singular value via SVD
        # (full SVD is fine for moderate sizes; replace with randomized SVD if needed)
        s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
        s1_sq = float(s[0] ** 2) if s.size > 0 else 0.0

        denom = max(s1_sq, self.epsilon)
        srank = fro2 / denom if denom > 0 else 0.0

        self._metadata.update({
            "input_shape": embeddings.shape,
            "n_vectors": int(embeddings.shape[0]),
            "n_features": int(embeddings.shape[1]),
            "frobenius_sq": fro2,
            "top_singular_sq": s1_sq,
            "stable_rank": srank
        })
        return np.array([srank], dtype=np.float64)
