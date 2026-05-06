"""Tests for EmbeddingProbe and the embedding_probe decorator."""

from __future__ import annotations

import numpy as np
import pytest

from ssrlib.analysis import EmbeddingProbe, embedding_probe
from ssrlib.processing import (
    ConditionNumberProcessor,
    CovarianceProcessor,
    EffectiveRankProcessor,
    NESumProcessor,
)


# ------------------------------------------------------- basic invocation
def test_probe_returns_metrics(emb_50x8):
    probe = EmbeddingProbe(processors=[NESumProcessor(), ConditionNumberProcessor()])
    metrics = probe(emb_50x8)
    assert "NESum" in metrics
    assert "ConditionNumber" in metrics
    assert isinstance(metrics["NESum"], float)


def test_probe_matrix_summarised_to_norm(emb_50x8):
    probe = EmbeddingProbe(processors=[CovarianceProcessor()])
    metrics = probe(emb_50x8)
    assert "Covariance.fro" in metrics
    assert "Covariance.shape" in metrics


def test_probe_step_and_epoch_propagate(emb_50x8):
    probe = EmbeddingProbe(processors=[NESumProcessor()])
    metrics = probe(emb_50x8, step=42, epoch=3)
    assert metrics["step"] == 42
    assert metrics["epoch"] == 3


def test_probe_sink_called(emb_50x8):
    sink_calls = []
    probe = EmbeddingProbe(
        processors=[NESumProcessor()], sink=lambda m: sink_calls.append(m)
    )
    probe(emb_50x8)
    assert len(sink_calls) == 1
    assert "NESum" in sink_calls[0]


# --------------------------------------------------------------- scheduling
def test_should_run_every_n_epochs():
    probe = EmbeddingProbe(processors=[NESumProcessor()], every_n_epochs=5)
    assert probe.should_run(epoch=0) is True
    assert probe.should_run(epoch=4) is False
    assert probe.should_run(epoch=5) is True


def test_should_run_every_n_steps():
    probe = EmbeddingProbe(processors=[NESumProcessor()], every_n_steps=10)
    assert probe.should_run(step=0) is True
    assert probe.should_run(step=5) is False
    assert probe.should_run(step=10) is True


# --------------------------------------------------------------- robustness
def test_probe_handles_processor_exception(emb_50x8):
    """Failure of one processor must not crash the probe."""

    class _BadProcessor(NESumProcessor):
        def process(self, X):
            raise RuntimeError("boom")

    probe = EmbeddingProbe(
        processors=[_BadProcessor(), ConditionNumberProcessor()],
    )
    metrics = probe(emb_50x8)
    assert "NESum.error" in metrics
    assert "ConditionNumber" in metrics


def test_probe_torch_input(emb_50x8):
    """Torch tensor inputs should be auto-converted."""
    pytest.importorskip("torch")
    import torch

    t = torch.from_numpy(emb_50x8)
    probe = EmbeddingProbe(processors=[NESumProcessor()])
    metrics = probe(t)
    assert "NESum" in metrics


# ----------------------------------------------------------------- decorator
def test_decorator_wraps_function(emb_50x8):
    @embedding_probe(processors=[NESumProcessor()])
    def encode(_):
        return emb_50x8

    out, metrics = encode(None)
    assert out is emb_50x8
    assert "NESum" in metrics


def test_probe_requires_at_least_one_processor():
    with pytest.raises(ValueError):
        EmbeddingProbe(processors=[])


def test_probe_scalar_only_skips_vectors(emb_50x8):
    probe = EmbeddingProbe(
        processors=[EffectiveRankProcessor(), CovarianceProcessor()],
        scalar_only=True,
    )
    metrics = probe(emb_50x8)
    # EffectiveRank returns shape (1,) -> scalar -> kept directly
    assert "EffectiveRank" in metrics
    # Covariance is a matrix -> summary scalars only
    assert "Covariance.fro" in metrics
