"""End-to-end pipeline smoke tests using the network-free IdentityEmbedder."""

from __future__ import annotations

import numpy as np
import pytest

from ssrlib import Pipeline, PipelineResults
from ssrlib.datasets import SynthTestDataset
from ssrlib.embedders import IdentityEmbedder
from ssrlib.processing import (
    CovarianceProcessor,
    NESumProcessor,
    SpectrumProcessor,
)


def _build_pipeline(processors=None, num_samples: int = 20):
    if processors is None:
        processors = [CovarianceProcessor(), NESumProcessor()]
    return Pipeline(
        [
            ("dataset", SynthTestDataset(tensors_num=num_samples, seed=42)),
            ("embedder", IdentityEmbedder(output_dim=16, seed=0)),
            ("processors", processors),
        ]
    )


def test_pipeline_runs_end_to_end():
    pipeline = _build_pipeline()
    results = pipeline.execute()

    assert isinstance(results, PipelineResults)
    assert len(results.embeddings) == 1
    assert len(results.processed) == 2

    emb = list(results.embeddings.values())[0]
    assert emb.shape == (20, 16)


def test_pipeline_results_keys_use_dataset_and_embedder():
    pipeline = _build_pipeline()
    results = pipeline.execute()

    keys = list(results.embeddings.keys())
    assert keys == [("SynthTest", "Identity")]

    proc_keys = list(results.processed.keys())
    assert ("SynthTest", "Identity", "Covariance") in proc_keys
    assert ("SynthTest", "Identity", "NESum") in proc_keys


def test_pipeline_streaming_matches_batch():
    """Streaming covariance via the pipeline should match whole-array path."""
    proc_batch = CovarianceProcessor()
    pipeline_batch = Pipeline(
        [
            ("dataset", SynthTestDataset(tensors_num=40, seed=1)),
            ("embedder", IdentityEmbedder(output_dim=8, seed=0)),
            ("processors", [proc_batch]),
        ]
    )
    results_batch = pipeline_batch.execute()
    cov_batch = results_batch.processed[("SynthTest", "Identity", "Covariance")]

    proc_stream = CovarianceProcessor()
    pipeline_stream = Pipeline(
        [
            ("dataset", SynthTestDataset(tensors_num=40, seed=1)),
            ("embedder", IdentityEmbedder(output_dim=8, seed=0)),
            ("processors", [proc_stream]),
        ]
    )
    results_stream = pipeline_stream.execute(streaming=True)
    cov_stream = results_stream.processed[("SynthTest", "Identity", "Covariance")]

    np.testing.assert_allclose(cov_batch, cov_stream, atol=1e-9)


def test_pipeline_metadata_complete():
    pipeline = _build_pipeline()
    results = pipeline.execute()
    md = results.metadata
    assert "datasets" in md
    assert "embedders" in md
    assert "processors" in md
    assert "config" in md
    assert "total_time" in results.timing


def test_pipeline_requires_dataset():
    with pytest.raises(ValueError, match="dataset"):
        Pipeline([("embedder", IdentityEmbedder())]).execute()


def test_pipeline_requires_embedder():
    with pytest.raises(ValueError, match="embedder"):
        Pipeline([("dataset", SynthTestDataset(tensors_num=5, seed=0))]).execute()


def test_pipeline_unique_dataset_keys_for_duplicates():
    """Two datasets with the same name should get unique keys."""
    pipeline = Pipeline(
        [
            (
                "datasets",
                [
                    SynthTestDataset(tensors_num=5, seed=0),
                    SynthTestDataset(tensors_num=5, seed=1),
                ],
            ),
            ("embedder", IdentityEmbedder(output_dim=8)),
            ("processors", [SpectrumProcessor()]),
        ]
    )
    results = pipeline.execute()
    keys = sorted({k[0] for k in results.embeddings.keys()})
    assert keys == ["SynthTest", "SynthTest[1]"]


def test_pipeline_no_processors_still_runs():
    pipeline = Pipeline(
        [
            ("dataset", SynthTestDataset(tensors_num=5, seed=0)),
            ("embedder", IdentityEmbedder(output_dim=8)),
        ]
    )
    results = pipeline.execute()
    assert results.processed == {}
    assert len(results.embeddings) == 1
