"""Shared pytest fixtures for ssrlib tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def emb_50x8():
    """Standard normal embeddings: 50 samples, 8 features. Seeded."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 8)).astype(np.float64)


@pytest.fixture
def emb_100x16():
    """Standard normal embeddings: 100 samples, 16 features. Seeded."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((100, 16)).astype(np.float64)


@pytest.fixture
def emb_powerlaw_64x32():
    """Embeddings whose covariance has a power-law spectrum (alpha = 1)."""
    rng = np.random.default_rng(123)
    n, d = 64, 32
    eigvals = 1.0 / np.arange(1, d + 1)  # alpha = 1
    L = np.diag(np.sqrt(eigvals))
    Z = rng.standard_normal((n, d))
    return (Z @ L).astype(np.float64)
