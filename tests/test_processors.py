"""Smoke tests for the original (refactored) processors."""

from __future__ import annotations

import numpy as np
import pytest

from ssrlib.processing import (
    CovarianceProcessor,
    EffectiveRankProcessor,
    LeverageScoresProcessor,
    PairwiseDistanceStatsProcessor,
    SpectrumProcessor,
    StableRankProcessor,
    ZCAProcessor,
    create_processor,
    list_processors,
)


# ---------------------------------------------------- registry / construction
def test_list_processors_contains_old_and_new():
    procs = list_processors()
    for name in (
        "CovarianceProcessor",
        "ZCAProcessor",
        "SpectrumProcessor",
        "EffectiveRankProcessor",
        "StableRankProcessor",
        "LeverageScoresProcessor",
        "PairwiseDistanceStatsProcessor",
        "NESumProcessor",
        "RankMeProcessor",
        "ConditionNumberProcessor",
    ):
        assert name in procs


def test_create_processor_returns_instance():
    p = create_processor("CovarianceProcessor")
    assert isinstance(p, CovarianceProcessor)


def test_create_processor_unknown_raises():
    with pytest.raises(ValueError, match="Unknown processor"):
        create_processor("NoSuchProcessor")


# ------------------------------------------------------------------ Covariance
def test_covariance_shape(emb_50x8):
    cov = CovarianceProcessor().process(emb_50x8)
    assert cov.shape == (8, 8)


def test_covariance_symmetric(emb_50x8):
    cov = CovarianceProcessor().process(emb_50x8)
    np.testing.assert_allclose(cov, cov.T, atol=1e-9)


def test_covariance_matches_np_cov(emb_50x8):
    cov = CovarianceProcessor().process(emb_50x8)
    np.testing.assert_allclose(cov, np.cov(emb_50x8.T), atol=1e-12)


def test_covariance_rejects_1d():
    with pytest.raises(ValueError, match="2D"):
        CovarianceProcessor().process(np.zeros(10))


# ------------------------------------------------------------------- ZCA
def test_zca_whitens(emb_50x8):
    whitened = ZCAProcessor(epsilon=1e-9).process(emb_50x8)
    cov_w = np.cov(whitened.T)
    # ZCA on small N is approximate; check we're close to identity
    np.testing.assert_allclose(cov_w, np.eye(8), atol=0.5)


def test_zca_preserves_shape(emb_50x8):
    out = ZCAProcessor().process(emb_50x8)
    assert out.shape == emb_50x8.shape


def test_zca_condition_number_finite_on_random_data(emb_50x8):
    proc = ZCAProcessor(epsilon=1e-9)
    proc.process(emb_50x8)
    cn = proc.get_metadata()["condition_number"]
    assert np.isfinite(cn) and cn > 0


def test_zca_condition_number_safe_on_degenerate():
    """A nearly rank-deficient matrix used to crash on division by zero."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 5))
    X[:, -1] = 0.0  # last feature is constant -> eigval ~ 0
    proc = ZCAProcessor(epsilon=1e-9)
    proc.process(X)
    cn = proc.get_metadata()["condition_number"]
    assert np.isfinite(cn)


# ------------------------------------------------------------------- Spectrum
def test_spectrum_shape(emb_50x8):
    spec = SpectrumProcessor().process(emb_50x8)
    # min(N, D) = 8 here
    assert spec.shape == (8,)


def test_spectrum_descending(emb_50x8):
    spec = SpectrumProcessor().process(emb_50x8)
    assert (spec[:-1] >= spec[1:] - 1e-12).all()


def test_spectrum_normalize_sums_to_one(emb_50x8):
    spec = SpectrumProcessor(normalize=True).process(emb_50x8)
    assert spec.sum() == pytest.approx(1.0, abs=1e-6)


# -------------------------------------------------------------- EffectiveRank
def test_effective_rank_shape_one(emb_50x8):
    r = EffectiveRankProcessor().process(emb_50x8)
    assert r.shape == (1,)


def test_effective_rank_in_range(emb_50x8):
    r = EffectiveRankProcessor().process(emb_50x8)[0]
    assert 0 < r <= emb_50x8.shape[1]


def test_effective_rank_uniform_spectrum_equals_d():
    """Identity covariance -> effective rank == D."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 6))  # roughly identity covariance
    r = EffectiveRankProcessor().process(X)[0]
    assert 5 <= r <= 6


# ------------------------------------------------------------ StableRank
def test_stable_rank_shape_one(emb_50x8):
    r = StableRankProcessor().process(emb_50x8)
    assert r.shape == (1,)


def test_stable_rank_positive(emb_50x8):
    r = StableRankProcessor().process(emb_50x8)[0]
    assert r > 0


def test_stable_rank_default_is_no_center():
    proc = StableRankProcessor()
    assert proc.center is False, "default must be center=False"


def test_stable_rank_no_center_matches_textbook():
    """With center=False, stable rank == ||X||_F^2 / s_1^2."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))
    fro2 = float(np.sum(X * X))
    s1 = float(np.linalg.svd(X, compute_uv=False)[0])
    expected = fro2 / (s1**2)

    r = StableRankProcessor(center=False).process(X)[0]
    assert r == pytest.approx(expected, rel=1e-6)


# ------------------------------------------------------------ LeverageScores
def test_leverage_scores_sum_equals_rank(emb_50x8):
    proc = LeverageScoresProcessor(rank=4)
    scores = proc.process(emb_50x8)
    assert scores.sum() == pytest.approx(4.0, abs=1e-6)


def test_leverage_scores_length_eq_n(emb_50x8):
    scores = LeverageScoresProcessor(rank=2).process(emb_50x8)
    assert scores.shape == (emb_50x8.shape[0],)


def test_leverage_scores_invalid_energy():
    with pytest.raises(ValueError):
        LeverageScoresProcessor(energy=0.0)


# ----------------------------------------------------------- PairwiseDistance
def test_pairwise_stats_shape(emb_50x8):
    stats = PairwiseDistanceStatsProcessor().process(emb_50x8)
    assert stats.shape == (4,)


def test_pairwise_stats_order_is_consistent(emb_50x8):
    proc = PairwiseDistanceStatsProcessor()
    stats = proc.process(emb_50x8)
    assert proc.get_metadata()["stats_order"] == ["mean", "std", "min", "max"]
    assert stats[0] == pytest.approx(proc.get_metadata()["mean"], abs=1e-12)
    assert stats[3] == pytest.approx(proc.get_metadata()["max"], abs=1e-12)


def test_pairwise_stats_subsamples_large_inputs():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10000, 4))  # > default max_samples
    proc = PairwiseDistanceStatsProcessor(max_samples=100, seed=0)
    proc.process(X)
    assert proc.get_metadata()["used_samples"] == 100


def test_pairwise_stats_invalid_metric():
    with pytest.raises(ValueError, match="metric"):
        PairwiseDistanceStatsProcessor(metric="manhattan")
