"""Tests for the MapReduceMixin streaming-covariance implementation."""

from __future__ import annotations

import numpy as np
import pytest

from ssrlib.processing import CovarianceProcessor
from ssrlib.processing.map_reduce import MapReduceMixin


def test_covariance_implements_mapreduce():
    proc = CovarianceProcessor()
    assert isinstance(proc, MapReduceMixin)


def test_streaming_covariance_matches_batch(emb_100x16):
    proc_batch = CovarianceProcessor()
    cov_batch = proc_batch.process(emb_100x16)

    proc_stream = CovarianceProcessor()
    proc_stream.reset()
    proc_stream.partial_fit(emb_100x16[:30])
    proc_stream.partial_fit(emb_100x16[30:75])
    proc_stream.partial_fit(emb_100x16[75:])
    cov_stream = proc_stream.finalize()

    np.testing.assert_allclose(cov_stream, cov_batch, atol=1e-9)


def test_streaming_covariance_single_batch(emb_100x16):
    proc = CovarianceProcessor()
    proc.partial_fit(emb_100x16)
    cov_stream = proc.finalize()

    cov_batch = CovarianceProcessor().process(emb_100x16)
    np.testing.assert_allclose(cov_stream, cov_batch, atol=1e-9)


def test_streaming_covariance_too_few_samples_raises():
    proc = CovarianceProcessor()
    proc.partial_fit(np.array([[1.0, 2.0]]))
    with pytest.raises(RuntimeError, match="at least 2 samples"):
        proc.finalize()


def test_reset_allows_reuse(emb_100x16):
    proc = CovarianceProcessor()
    proc.partial_fit(emb_100x16)
    cov1 = proc.finalize()
    proc.reset()
    proc.partial_fit(emb_100x16)
    cov2 = proc.finalize()
    np.testing.assert_allclose(cov1, cov2, atol=1e-12)


def test_streaming_partial_fit_rejects_1d():
    proc = CovarianceProcessor()
    with pytest.raises(ValueError, match="2D"):
        proc.partial_fit(np.zeros(5))
