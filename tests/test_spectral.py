"""Tests for the shared _spectral helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ssrlib.processing._spectral import (
    EPS,
    centered,
    covariance_eigvals,
    covariance_matrix,
    top_singular_value,
)


def test_centered_zero_mean(emb_50x8):
    Xc = centered(emb_50x8)
    np.testing.assert_allclose(Xc.mean(axis=0), 0.0, atol=1e-12)


def test_centered_unchanged_when_already_centered():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    X = X - X.mean(axis=0)
    np.testing.assert_allclose(centered(X), X, atol=1e-12)


def test_covariance_eigvals_matches_np_cov(emb_100x16):
    eig_via_helper = covariance_eigvals(emb_100x16)
    eig_via_cov = np.sort(np.linalg.eigvalsh(np.cov(emb_100x16.T)))[::-1]
    np.testing.assert_allclose(eig_via_helper, eig_via_cov, atol=1e-9)


def test_covariance_eigvals_descending(emb_50x8):
    eig = covariance_eigvals(emb_50x8)
    assert (eig[:-1] >= eig[1:] - 1e-12).all(), "eigvals should be descending"


def test_covariance_eigvals_nonneg(emb_50x8):
    eig = covariance_eigvals(emb_50x8)
    assert (eig >= 0).all()


def test_covariance_eigvals_no_center_doubles_squared_norms():
    # For mean-free X, centered and not-centered should agree.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))
    X = X - X.mean(axis=0)
    np.testing.assert_allclose(
        covariance_eigvals(X, center=True),
        covariance_eigvals(X, center=False),
        atol=1e-9,
    )


def test_top_singular_value_matches_svd(emb_50x8):
    sv = np.linalg.svd(emb_50x8, compute_uv=False)
    assert top_singular_value(emb_50x8, center=False) == pytest.approx(sv[0])


def test_covariance_matrix_handles_2d():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    cov = covariance_matrix(X)
    np.testing.assert_allclose(cov, np.cov(X.T), atol=1e-12)
    assert cov.shape == (4, 4)


def test_covariance_matrix_single_feature():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 1))
    cov = covariance_matrix(X)
    assert cov.shape == (1, 1)
    np.testing.assert_allclose(cov[0, 0], np.var(X[:, 0], ddof=1), atol=1e-12)


def test_eps_is_small():
    assert 0 < EPS < 1e-9


def test_covariance_eigvals_rejects_1d():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="2D"):
        covariance_eigvals(rng.standard_normal(10))
