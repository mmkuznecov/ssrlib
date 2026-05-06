"""Shared spectral helpers for processors.

These are private utilities that consolidate the centering / SVD / eigvals
logic that was previously duplicated across CovarianceProcessor, ZCAProcessor,
SpectrumProcessor, EffectiveRankProcessor, and others.

Conventions used throughout ssrlib:
    - Embedding matrix X has shape (N, D) — N samples, D features.
    - Covariance is computed via numpy convention: features in columns
      (i.e., np.cov(X.T) with default ddof=1, dividing by N-1).
    - Eigenvalues are returned sorted in descending order.
"""

from __future__ import annotations

import numpy as np

EPS: float = 1e-12


def centered(X: np.ndarray) -> np.ndarray:
    """Mean-center X along the sample axis (axis=0)."""
    return X - X.mean(axis=0, keepdims=True)


def covariance_eigvals(X: np.ndarray, center: bool = True) -> np.ndarray:
    """Eigenvalues of np.cov(X.T) (ddof=1), sorted descending.

    Computed via SVD of (centered) X to avoid forming a (D, D) matrix when
    N < D. Mathematically equivalent to ``np.linalg.eigvalsh(np.cov(X.T))[::-1]``
    (modulo numerical noise on near-zero eigenvalues).

    Args:
        X: shape (N, D) embedding matrix.
        center: if True, mean-center X before computing.

    Returns:
        Array of length min(N, D) with eigenvalues sorted descending.
        Negative values from numerical noise are clipped to 0.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D, got shape {X.shape}")
    Xc = centered(X) if center else X
    n = max(Xc.shape[0] - 1, 1)  # ddof=1 to match np.cov default
    sv = np.linalg.svd(Xc, compute_uv=False)
    eig = (sv**2) / n
    eig = np.maximum(eig, 0.0)  # guard against -0.0 / numerical noise
    return np.sort(eig)[::-1]


def top_singular_value(X: np.ndarray, center: bool = False) -> float:
    """Largest singular value of (optionally centered) X.

    Defaults to NOT centering to match the textbook stable-rank definition
    ``||X||_F^2 / ||X||_2^2``.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D, got shape {X.shape}")
    Xc = centered(X) if center else X
    sv = np.linalg.svd(Xc, compute_uv=False)
    return float(sv[0]) if sv.size else 0.0


def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Robust wrapper around np.cov(X.T) that handles 1-feature inputs.

    np.cov returns a 0-d array when the input has a single row of features;
    callers in this codebase always expect a 2-D matrix.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D, got shape {X.shape}")
    cov = np.cov(X.T)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    return cov
