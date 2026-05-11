"""Tests for NeuralCollapseProcessor and EmbeddingProbe context forwarding."""

from __future__ import annotations

import numpy as np
import pytest

from ssrlib.analysis import EmbeddingProbe
from ssrlib.processing import NESumProcessor, NeuralCollapseProcessor


# ----------------------------------------------------------- helpers / fixtures
def _simplex_etf(C: int) -> np.ndarray:
    """C × C standard Simplex ETF (rows are unit-norm, pairwise cos = -1/(C-1))."""
    return np.sqrt(C / (C - 1)) * (np.eye(C) - np.ones((C, C)) / C)


@pytest.fixture
def labelled_random():
    """Embeddings + labels with structure (10 classes, 16 features, 200 samples)."""
    rng = np.random.default_rng(0)
    n_per_class, C, D = 20, 10, 16
    centers = rng.standard_normal((C, D)) * 3.0
    X_blocks, y_blocks = [], []
    for c in range(C):
        X_blocks.append(centers[c] + rng.standard_normal((n_per_class, D)) * 1.0)
        y_blocks.append(np.full(n_per_class, c))
    return np.concatenate(X_blocks).astype(np.float64), np.concatenate(y_blocks)


# --------------------------------------------------------------- registration
def test_neural_collapse_registered():
    from ssrlib.processing import list_processors

    assert "NeuralCollapseProcessor" in list_processors()


def test_neural_collapse_creatable_by_name():
    from ssrlib.processing import create_processor

    proc = create_processor("NeuralCollapseProcessor")
    assert isinstance(proc, NeuralCollapseProcessor)


# ---------------------------------------------------------- core NC behaviour
def test_nc_processor_requires_labels():
    proc = NeuralCollapseProcessor()
    with pytest.raises(ValueError, match="requires labels"):
        proc.process(np.zeros((10, 4)))


def test_nc_processor_rejects_label_length_mismatch():
    proc = NeuralCollapseProcessor()
    X = np.zeros((10, 4))
    y = np.zeros(7, dtype=np.int64)
    with pytest.raises(ValueError, match="labels"):
        proc.process(X, labels=y)


def test_nc_processor_rejects_single_class():
    proc = NeuralCollapseProcessor()
    X = np.random.default_rng(0).standard_normal((10, 4))
    y = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="2 classes"):
        proc.process(X, labels=y)


def test_nc_processor_default_output_shape(labelled_random):
    X, y = labelled_random
    out = NeuralCollapseProcessor().process(X, labels=y)
    assert out.shape == (4,)


def test_nc_processor_components_order(labelled_random):
    X, y = labelled_random
    proc = NeuralCollapseProcessor()
    proc.process(X, labels=y)
    md = proc.get_metadata()
    assert md["components_order"] == [
        "nc1",
        "nc2_equinorm",
        "nc2_equiangle",
        "nc2_max_equiangle",
    ]


def test_nc_processor_metadata_matches_components(labelled_random):
    X, y = labelled_random
    proc = NeuralCollapseProcessor()
    out = proc.process(X, labels=y)
    md = proc.get_metadata()
    for i, key in enumerate(md["components_order"]):
        assert out[i] == pytest.approx(md[key], abs=1e-12)


# ------------------------------------------------------------- correctness
def test_nc1_zero_when_no_within_class_variance():
    """If every sample is exactly its class mean, ΣW = 0 and NC1 = 0."""
    rng = np.random.default_rng(0)
    means = rng.standard_normal((5, 16))
    X = np.repeat(means, 10, axis=0)
    y = np.repeat(np.arange(5), 10)

    out = NeuralCollapseProcessor().process(X, labels=y)
    assert out[0] == pytest.approx(0.0, abs=1e-9)


def test_nc2_zero_for_simplex_etf():
    """When class means form a Simplex ETF, all NC2 metrics are 0."""
    C = 5
    M = _simplex_etf(C)  # (C, C); rows are ETF vertices
    n_per = 5
    # Each "class" has n_per copies of its ETF vertex (no within-class noise)
    X = np.repeat(M, n_per, axis=0)
    y = np.repeat(np.arange(C), n_per)

    out = NeuralCollapseProcessor().process(X, labels=y)
    nc1, equinorm, equiangle, max_eq = out
    assert nc1 == pytest.approx(0.0, abs=1e-9)
    assert equinorm == pytest.approx(0.0, abs=1e-9)
    assert equiangle == pytest.approx(0.0, abs=1e-9)
    assert max_eq == pytest.approx(0.0, abs=1e-9)


def test_nc1_decreases_when_within_variance_decreases():
    """Lowering within-class noise should reduce NC1."""
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((4, 8)) * 5.0
    n_per = 50

    def build(noise_std: float):
        Xb, yb = [], []
        for c in range(4):
            Xb.append(centers[c] + rng.standard_normal((n_per, 8)) * noise_std)
            yb.append(np.full(n_per, c))
        return np.concatenate(Xb), np.concatenate(yb)

    X_hi, y_hi = build(noise_std=2.0)
    X_lo, y_lo = build(noise_std=0.1)

    nc1_hi = NeuralCollapseProcessor().process(X_hi, labels=y_hi)[0]
    nc1_lo = NeuralCollapseProcessor().process(X_lo, labels=y_lo)[0]
    assert nc1_lo < nc1_hi


# ------------------------------------------------------------- NC3 + NC4
def test_nc3_zero_when_classifier_equals_centered_means(labelled_random):
    """W = centered class means → perfect self-duality."""
    X, y = labelled_random
    classes = np.unique(y)
    means = np.stack([X[y == c].mean(axis=0) for c in classes])
    centered_means = means - X.mean(axis=0)

    out = NeuralCollapseProcessor().process(
        X, labels=y, classifier_weights=centered_means
    )
    assert out.shape == (6,)
    nc3 = out[4]
    assert nc3 == pytest.approx(0.0, abs=1e-12)


def test_nc4_zero_when_classifier_is_ncc():
    """Use uncentered class means as classifier — predictions match NCC exactly."""
    rng = np.random.default_rng(0)
    n_per, C, D = 30, 5, 8
    centers = rng.standard_normal((C, D)) * 3.0
    X_blocks, y_blocks = [], []
    for c in range(C):
        X_blocks.append(centers[c] + rng.standard_normal((n_per, D)) * 0.3)
        y_blocks.append(np.full(n_per, c))
    X = np.concatenate(X_blocks)
    y = np.concatenate(y_blocks)

    classes = np.unique(y)
    means = np.stack([X[y == c].mean(axis=0) for c in classes])

    # NCC decision argmin ‖x − μ‖² = argmax (μᵀ x − ½‖μ‖²)
    # So an equivalent linear classifier has W = means and b = -½ ‖μ_c‖²
    W = means
    b = -0.5 * (means * means).sum(axis=1)

    out = NeuralCollapseProcessor().process(
        X, labels=y, classifier_weights=W, classifier_bias=b
    )
    nc4 = out[5]
    assert nc4 == pytest.approx(0.0, abs=1e-12)


def test_nc3_nc4_appear_in_metadata_when_classifier_supplied(labelled_random):
    X, y = labelled_random
    classes = np.unique(y)
    C, D = len(classes), X.shape[1]
    rng = np.random.default_rng(1)
    W = rng.standard_normal((C, D))
    proc = NeuralCollapseProcessor()
    proc.process(X, labels=y, classifier_weights=W)
    md = proc.get_metadata()
    assert "nc3_selfdual" in md
    assert "nc4_ncc_mismatch" in md
    assert md["components_order"] == [
        "nc1",
        "nc2_equinorm",
        "nc2_equiangle",
        "nc2_max_equiangle",
        "nc3_selfdual",
        "nc4_ncc_mismatch",
    ]


def test_nc_classifier_wrong_shape_raises(labelled_random):
    X, y = labelled_random
    proc = NeuralCollapseProcessor()
    bad_W = np.zeros((3, X.shape[1]))  # wrong number of classes
    with pytest.raises(ValueError, match="classifier_weights"):
        proc.process(X, labels=y, classifier_weights=bad_W)


# ------------------------------------------------------------- probe wiring
def test_probe_passes_labels_to_nc(labelled_random):
    X, y = labelled_random
    probe = EmbeddingProbe(processors=[NeuralCollapseProcessor()])
    metrics = probe(X, labels=y)
    assert "NeuralCollapse.0" in metrics
    assert "NeuralCollapse.1" in metrics
    assert "NeuralCollapse.2" in metrics
    assert "NeuralCollapse.3" in metrics


def test_probe_passes_classifier_weights_to_nc(labelled_random):
    X, y = labelled_random
    classes = np.unique(y)
    C, D = len(classes), X.shape[1]
    rng = np.random.default_rng(0)
    W = rng.standard_normal((C, D))
    b = rng.standard_normal(C)

    probe = EmbeddingProbe(processors=[NeuralCollapseProcessor()])
    metrics = probe(X, labels=y, classifier_weights=W, classifier_bias=b)
    # Output is now shape-(6,), so we get 6 keys NeuralCollapse.0 … 5
    for i in range(6):
        assert f"NeuralCollapse.{i}" in metrics


def test_probe_does_not_break_old_processors_with_extra_context(labelled_random):
    """Existing processors that don't take labels should ignore the kwarg."""
    X, y = labelled_random
    probe = EmbeddingProbe(processors=[NESumProcessor()])
    # Pass labels — NESum's signature doesn't list 'labels', so it's filtered out
    metrics = probe(X, labels=y)
    assert "NESum" in metrics
    assert "NESum.error" not in metrics


def test_probe_mixed_processors_in_one_call(labelled_random):
    X, y = labelled_random
    probe = EmbeddingProbe(processors=[NESumProcessor(), NeuralCollapseProcessor()])
    metrics = probe(X, labels=y, epoch=3)
    assert "NESum" in metrics
    assert "NeuralCollapse.0" in metrics
    assert metrics["epoch"] == 3


def test_probe_without_labels_records_error_for_nc(labelled_random):
    """If labels are missing, NC fails gracefully (per-processor try/except)."""
    X, _ = labelled_random
    probe = EmbeddingProbe(processors=[NESumProcessor(), NeuralCollapseProcessor()])
    metrics = probe(X)  # no labels
    assert "NESum" in metrics
    assert "NeuralCollapse.error" in metrics
    assert "labels" in metrics["NeuralCollapse.error"]
