"""Tests for the new spectral-quality processors."""

from __future__ import annotations

import numpy as np
import pytest

from ssrlib.processing import (
    AlphaReQProcessor,
    CoherenceProcessor,
    ConditionNumberProcessor,
    EntropyDecompositionProcessor,
    NESumProcessor,
    ParticipationRatioProcessor,
    RankMeProcessor,
)


# --------------------------------------------------------------------- NESum
def test_nesum_shape(emb_50x8):
    out = NESumProcessor().process(emb_50x8)
    assert out.shape == (1,)


def test_nesum_in_range(emb_50x8):
    val = NESumProcessor().process(emb_50x8)[0]
    assert 0 < val <= emb_50x8.shape[1]


def test_nesum_identity_eq_d():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 4))
    val = NESumProcessor().process(X)[0]
    assert val == pytest.approx(4.0, rel=0.2)


# --------------------------------------------------------------------- RankMe
def test_rankme_shape(emb_50x8):
    out = RankMeProcessor().process(emb_50x8)
    assert out.shape == (1,)


def test_rankme_in_range(emb_50x8):
    val = RankMeProcessor().process(emb_50x8)[0]
    assert 1 <= val <= emb_50x8.shape[1]


# --------------------------------------------------------------------- AlphaReQ
def test_alpha_req_shape(emb_powerlaw_64x32):
    out = AlphaReQProcessor().process(emb_powerlaw_64x32)
    assert out.shape == (2,)


def test_alpha_req_recovers_known_alpha(emb_powerlaw_64x32):
    """Eigenvalues are 1/i (alpha=1) plus sample noise."""
    out = AlphaReQProcessor().process(emb_powerlaw_64x32)
    alpha, r2 = out
    assert 0.4 < alpha < 1.6, f"recovered alpha={alpha} should be near 1"
    assert r2 > 0.7, f"R^2={r2} should be high for clean power law"


def test_alpha_req_few_eigvals_returns_nan():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 2))
    out = AlphaReQProcessor(min_eigvals=5).process(X)
    assert np.isnan(out).all()


# ------------------------------------------------------------- ParticipationRatio
def test_participation_ratio_shape(emb_50x8):
    out = ParticipationRatioProcessor().process(emb_50x8)
    assert out.shape == (1,)


def test_participation_ratio_in_range(emb_50x8):
    val = ParticipationRatioProcessor().process(emb_50x8)[0]
    assert 0 < val <= emb_50x8.shape[1]


# ---------------------------------------------------------------------- Coherence
def test_coherence_shape(emb_50x8):
    out = CoherenceProcessor().process(emb_50x8)
    assert out.shape == (1,)


def test_coherence_positive(emb_50x8):
    val = CoherenceProcessor().process(emb_50x8)[0]
    assert val > 0


# --------------------------------------------------------------- ConditionNumber
def test_condition_number_shape(emb_50x8):
    out = ConditionNumberProcessor().process(emb_50x8)
    assert out.shape == (1,)


def test_condition_number_finite_random(emb_50x8):
    val = ConditionNumberProcessor().process(emb_50x8)[0]
    assert np.isfinite(val) and val >= 1


def test_condition_number_inf_on_zero():
    """All-zero data has condition number inf."""
    X = np.zeros((10, 4))
    val = ConditionNumberProcessor().process(X)[0]
    assert val == float("inf")


# ------------------------------------------------------ EntropyDecomposition
def test_entropy_decomp_shape(emb_50x8):
    out = EntropyDecompositionProcessor().process(emb_50x8)
    assert out.shape == (6,)


def test_entropy_decomp_components_exposed(emb_50x8):
    proc = EntropyDecompositionProcessor()
    out = proc.process(emb_50x8)
    md = proc.get_metadata()
    assert md["components_order"][0] == "scalar_inflation"
    assert out[0] == pytest.approx(md["scalar_inflation"], abs=1e-9)
    assert out[1] == pytest.approx(md["linear_collapse"], abs=1e-9)
    assert out[2] == pytest.approx(md["nonlinear_collapse"], abs=1e-9)
