import numpy as np
from typing import Dict, Any
from .base import BaseProcessor


class EffectiveRankProcessor(BaseProcessor):
    """
    Returns the 'effective rank' of the (centered) covariance:
        erank = exp( - sum_i p_i log p_i ), where p_i = λ_i / sum_j λ_j
    A soft, scale-invariant dimensionality proxy.
    """

    def __init__(self, epsilon: float = 1e-12, center: bool = True, **kwargs):
        """
        Args:
            epsilon: small floor for eigenvalues and probs.
            center: whether to mean-center before covariance.
        """
        super().__init__("EffectiveRank", **kwargs)
        self.epsilon = float(epsilon)
        self.center = bool(center)

        self._metadata.update({
            "processor_type": "effective_rank",
            "epsilon": self.epsilon,
            "center": self.center,
            "output_type": "scalar_statistic"
        })

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        X = embeddings.astype(np.float64, copy=False)
        if self.center:
            X = X - X.mean(axis=0, keepdims=True)

        C = np.cov(X.T)
        evals = np.linalg.eigvalsh(C)  # sorted ascending
        evals = np.maximum(evals, 0.0)
        total = evals.sum()

        if not np.isfinite(total) or total <= self.epsilon:
            erank = 0.0
        else:
            p = evals / (total + self.epsilon)
            p = np.clip(p, self.epsilon, 1.0)  # avoid log(0)
            H = -np.sum(p * np.log(p))
            erank = float(np.exp(H))

        self._metadata.update({
            "input_shape": embeddings.shape,
            "n_vectors": int(embeddings.shape[0]),
            "n_features": int(embeddings.shape[1]),
            "effective_rank": erank
        })
        return np.array([erank], dtype=np.float64)
