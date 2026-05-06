"""Spectral-quality metrics ported from common ML literature.

Each processor returns a shape-(1,) array for scalar metrics, except
``EntropyDecompositionProcessor`` which returns a shape-(6,) vector containing
the three entropy components plus three diagnostics.

References:
    He & Ozay 2022 (NESum)
    Garrido et al. 2022 (RankMe)
    Tsitsulin et al. 2023 (Stable rank, Coherence, Self-cluster)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from ._spectral import EPS, centered, covariance_eigvals
from .base import BaseProcessor


class NESumProcessor(BaseProcessor):
    """Normalized Eigenvalue Sum: ``Σ_i λ_i / λ_0``.

    Bounded in (0, D]. High values indicate a flat (high-rank) spectrum.

    Reference: He & Ozay 2022.
    """

    def __init__(self, center: bool = True, **kwargs):
        super().__init__("NESum", **kwargs)
        self.center = bool(center)
        self._metadata.update(
            {
                "processor_type": "spectral_quality",
                "metric": "nesum",
                "center": self.center,
                "output_type": "scalar_statistic",
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        eig = covariance_eigvals(embeddings, center=self.center)
        value = 0.0 if eig.size == 0 or eig[0] == 0 else float(eig.sum() / eig[0])

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_features": int(embeddings.shape[1]),
                "value": value,
            }
        )
        return np.array([value], dtype=np.float64)


class RankMeProcessor(BaseProcessor):
    """RankMe: ``exp(H(p))`` where ``p_i = λ_i / Σ λ``.

    Reference: Garrido et al. 2022.
    """

    def __init__(self, center: bool = True, **kwargs):
        super().__init__("RankMe", **kwargs)
        self.center = bool(center)
        self._metadata.update(
            {
                "processor_type": "spectral_quality",
                "metric": "rankme",
                "center": self.center,
                "output_type": "scalar_statistic",
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        eig = covariance_eigvals(embeddings, center=self.center)
        s = eig.sum()
        if s <= 0:
            value = 0.0
        else:
            p = eig / s
            p = p[p > EPS]
            value = float(np.exp(-np.sum(p * np.log(p))))

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_features": int(embeddings.shape[1]),
                "value": value,
            }
        )
        return np.array([value], dtype=np.float64)


class AlphaReQProcessor(BaseProcessor):
    """Power-law decay coefficient α of the eigenvalue spectrum, fit by OLS on log-log.

    Returns a shape-(2,) array ``[alpha, r_squared]``. NaNs when the spectrum
    has fewer than ``min_eigvals`` positive values.

    Args:
        center: whether to center embeddings before computing covariance.
        min_eigvals: minimum number of positive eigenvalues required.
        max_rank_fraction: fit only the top fraction of eigenvalues.
    """

    def __init__(
        self,
        center: bool = True,
        min_eigvals: int = 5,
        max_rank_fraction: float = 0.9,
        **kwargs,
    ):
        super().__init__("AlphaReQ", **kwargs)
        if min_eigvals < 2:
            raise ValueError("min_eigvals must be >= 2")
        if not (0.0 < max_rank_fraction <= 1.0):
            raise ValueError("max_rank_fraction must be in (0, 1]")

        self.center = bool(center)
        self.min_eigvals = int(min_eigvals)
        self.max_rank_fraction = float(max_rank_fraction)

        self._metadata.update(
            {
                "processor_type": "spectral_quality",
                "metric": "alpha_req",
                "center": self.center,
                "min_eigvals": self.min_eigvals,
                "max_rank_fraction": self.max_rank_fraction,
                "output_type": "alpha_and_r2",
                "components_order": ["alpha", "r_squared"],
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        eig = covariance_eigvals(embeddings, center=self.center)
        pos = eig[eig > 0]

        if len(pos) < self.min_eigvals:
            alpha, r2 = float("nan"), float("nan")
        else:
            K = max(self.min_eigvals, int(len(pos) * self.max_rank_fraction))
            pos = pos[:K]
            log_i = np.log(np.arange(1, len(pos) + 1, dtype=float))
            slope, _, r_value, _, _ = stats.linregress(log_i, np.log(pos))
            alpha = float(-slope)
            r2 = float(r_value**2)

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_features": int(embeddings.shape[1]),
                "alpha": alpha,
                "r_squared": r2,
            }
        )
        return np.array([alpha, r2], dtype=np.float64)


class ParticipationRatioProcessor(BaseProcessor):
    """Participation Ratio: ``(Σλ)² / Σ(λ²)``.

    Effective number of "active" dimensions.
    """

    def __init__(self, center: bool = True, **kwargs):
        super().__init__("ParticipationRatio", **kwargs)
        self.center = bool(center)
        self._metadata.update(
            {
                "processor_type": "spectral_quality",
                "metric": "participation_ratio",
                "center": self.center,
                "output_type": "scalar_statistic",
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        eig = covariance_eigvals(embeddings, center=self.center)
        s2 = float(np.sum(eig**2))
        value = 0.0 if s2 == 0 else float(eig.sum() ** 2 / s2)

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_features": int(embeddings.shape[1]),
                "value": value,
            }
        )
        return np.array([value], dtype=np.float64)


class CoherenceProcessor(BaseProcessor):
    """Incoherence: ``μ_0 = max_i ||U^T e_i||² · n / r`` on raw X.

    Reference: Tsitsulin et al. 2023. Defaults to ``center=False`` to match
    the published definition.
    """

    def __init__(self, center: bool = False, **kwargs):
        super().__init__("Coherence", **kwargs)
        self.center = bool(center)
        self._metadata.update(
            {
                "processor_type": "spectral_quality",
                "metric": "coherence",
                "center": self.center,
                "output_type": "scalar_statistic",
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        X = (
            centered(embeddings)
            if self.center
            else embeddings.astype(np.float64, copy=False)
        )
        n, d = X.shape
        r = min(n, d)

        try:
            U, _, _ = np.linalg.svd(X, full_matrices=False)
            value = (
                float(np.max(np.sum(U**2, axis=1)) * n / r) if r > 0 else float("nan")
            )
        except np.linalg.LinAlgError:
            value = float("nan")

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_features": int(d),
                "value": value,
            }
        )
        return np.array([value], dtype=np.float64)


class ConditionNumberProcessor(BaseProcessor):
    """Condition number of the (centered) covariance: ``λ_max / λ_min``.

    Returns ``inf`` when the smallest eigenvalue falls below ``epsilon``.
    """

    def __init__(self, center: bool = True, epsilon: float = 1e-12, **kwargs):
        super().__init__("ConditionNumber", **kwargs)
        self.center = bool(center)
        self.epsilon = float(epsilon)
        self._metadata.update(
            {
                "processor_type": "spectral_quality",
                "metric": "condition_number",
                "center": self.center,
                "epsilon": self.epsilon,
                "output_type": "scalar_statistic",
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        eig = covariance_eigvals(embeddings, center=self.center)

        if eig.size == 0 or eig[0] <= self.epsilon:
            value = float("inf")
        else:
            denom = max(eig[-1], self.epsilon)
            value = float(eig[0] / denom)

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_features": int(embeddings.shape[1]),
                "max_eigenvalue": float(eig[0]) if eig.size else 0.0,
                "min_eigenvalue": float(eig[-1]) if eig.size else 0.0,
                "value": value,
            }
        )
        return np.array([value], dtype=np.float64)


class EntropyDecompositionProcessor(BaseProcessor):
    """Decompose Gaussian differential entropy into three components.

    Following the breakdown:

        h(X) = (D/2) log(σ_max)             # scalar inflation
             + (1/2) log det(Σ / σ_max)    # linear collapse
             + h(Σ^{-1/2} X)                 # non-linear collapse

    Returns a shape-(6,) array; consult ``components_order`` in metadata.
    """

    _ORDER = [
        "scalar_inflation",
        "linear_collapse",
        "nonlinear_collapse",
        "entropy_total",
        "entropy_decomposed",
        "decomposition_error",
    ]

    def __init__(self, **kwargs):
        super().__init__("EntropyDecomposition", **kwargs)
        self._metadata.update(
            {
                "processor_type": "spectral_quality",
                "metric": "entropy_decomposition",
                "output_type": "decomposition_components",
                "components_order": list(self._ORDER),
            }
        )

    def process(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        n, d = embeddings.shape
        eig = covariance_eigvals(embeddings, center=True)
        eig = np.maximum(eig, EPS)
        sigma_max = eig[0]

        scalar_inflation = (d / 2) * np.log(sigma_max)
        linear_collapse = 0.5 * np.sum(np.log(eig / sigma_max))

        Xc = centered(embeddings)
        cov = (Xc.T @ Xc) / max(n, 1)

        try:
            U, s, _ = np.linalg.svd(cov)
            sigma_inv_sqrt = U @ np.diag(1.0 / np.sqrt(s + EPS)) @ U.T
        except np.linalg.LinAlgError:
            sigma_inv_sqrt = np.linalg.pinv(cov + EPS * np.eye(d))

        Xw = Xc @ sigma_inv_sqrt
        cov_w = (Xw.T @ Xw) / max(n, 1)
        sv_w = np.maximum(np.linalg.svd(cov_w, compute_uv=False), EPS)
        h_whitened = (d / 2) * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(sv_w))

        h_total = (d / 2) * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(eig))
        h_decomp = scalar_inflation + linear_collapse + h_whitened

        out = np.array(
            [
                float(scalar_inflation),
                float(linear_collapse),
                float(h_whitened),
                float(h_total),
                float(h_decomp),
                float(abs(h_total - h_decomp)),
            ],
            dtype=np.float64,
        )

        self._metadata.update(
            {
                "input_shape": embeddings.shape,
                "n_features": int(d),
                "scalar_inflation": float(scalar_inflation),
                "linear_collapse": float(linear_collapse),
                "nonlinear_collapse": float(h_whitened),
                "decomposition_error": float(abs(h_total - h_decomp)),
            }
        )
        return out
