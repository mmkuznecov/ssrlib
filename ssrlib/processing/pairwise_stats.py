import numpy as np
from typing import Optional
from .base import BaseProcessor


class PairwiseDistanceStatsProcessor(BaseProcessor):
    """
    Computes summary statistics of pairwise distances between embeddings
    on a (possibly subsampled) set of points.

    Returns a 1D array [mean, std, min, max] for the chosen distance metric.
    """

    def __init__(
        self,
        metric: str = "cosine",
        max_samples: int = 4096,
        center: bool = False,
        seed: Optional[int] = 0,
        **kwargs,
    ):
        """
        Args:
            metric: 'cosine' or 'euclidean'.
            max_samples: maximum number of points to use for pairwise stats.
                         If n_vectors > max_samples, a random subset is taken.
            center: whether to mean-center before computing distances.
            seed: random seed for subsampling (None = use np.random default).
        """
        super().__init__("PairwiseDistanceStats", **kwargs)
        metric = metric.lower()
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        if max_samples <= 1:
            raise ValueError("max_samples must be > 1")

        self.metric = metric
        self.max_samples = int(max_samples)
        self.center = bool(center)
        self.seed = seed

        self._metadata.update(
            {
                "processor_type": "pairwise_distance_stats",
                "metric": self.metric,
                "max_samples": self.max_samples,
                "center": self.center,
                "seed": self.seed,
                "output_type": "distance_summary",
                "stats_order": ["mean", "std", "min", "max"],
            }
        )

    def _subsample(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if n <= self.max_samples:
            return X

        rng = np.random.default_rng(self.seed)
        idx = rng.choice(n, size=self.max_samples, replace=False)
        return X[idx]

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        X = embeddings.astype(np.float64, copy=False)
        if self.center:
            X = X - X.mean(axis=0, keepdims=True)

        X = self._subsample(X)
        m = X.shape[0]

        if m < 2:
            # Нечего считать — вернём нули
            stats = np.zeros(4, dtype=np.float64)
            self._metadata.update(
                {
                    "input_shape": embeddings.shape,
                    "used_samples": int(m),
                    "pairwise_count": 0,
                }
            )
            return stats

        # Вычисляем попарные расстояния/сходства
        if self.metric == "cosine":
            # Нормируем строки
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            Y = X / norms
            sim = Y @ Y.T  # (m, m), cosine similarity
            # Превращаем в расстояние
            dist = 1.0 - sim
        else:  # 'euclidean'
            # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i x_j^T
            sq_norms = np.sum(X * X, axis=1, keepdims=True)  # (m, 1)
            sq_dists = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
            # Численно из-за округления sq_dists могут чуть уходить в минус
            sq_dists = np.maximum(sq_dists, 0.0)
            dist = np.sqrt(sq_dists)

        # Берём только верхний треугольник (без диагонали)
        iu = np.triu_indices(m, k=1)
        dist_vec = dist[iu]

        mean = float(dist_vec.mean())
        std = float(dist_vec.std())
        dmin = float(dist_vec.min())
        dmax = float(dist_vec.max())

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "used_samples": int(m),
                "pairwise_count": int(dist_vec.size),
                "mean": mean,
                "std": std,
                "min": dmin,
                "max": dmax,
            }
        )

        return np.array([mean, std, dmin, dmax], dtype=np.float64)
