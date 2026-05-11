"""Tests for the dataset registries and the network-free SynthTestDataset."""

from __future__ import annotations

import pytest
import torch

from ssrlib.datasets import (
    HF_DATASET_REGISTRY,
    HFVisionDataset,
    SynthTestDataset,
    create_dataset,
    get_hf_dataset_info,
    list_datasets,
    list_hf_datasets,
)


# -------------------------------------------------------- HF registry only
def test_hf_registry_has_expected_entries():
    keys = set(list_hf_datasets())
    for must_have in ("cifar10", "cifar100", "food101"):
        assert must_have in keys


def test_hf_dataset_info_for_cifar10():
    info = get_hf_dataset_info("cifar10")
    assert info.hf_id == "uoft-cs/cifar10"
    assert info.num_classes == 10


def test_hf_dataset_info_unknown_raises():
    with pytest.raises(ValueError, match="Unknown HuggingFace"):
        get_hf_dataset_info("not_a_dataset")


def test_hf_visionn_dataset_unknown_name_raises():
    """Constructing HFVisionDataset with unknown name fails fast."""
    with pytest.raises(ValueError, match="Unknown"):
        HFVisionDataset(dataset_name="absolutely_not_a_dataset")


# ----------------------------------------------------- list_datasets registry
def test_list_datasets_includes_synthetic():
    names = list_datasets()
    assert "SynthTestDataset" in names
    assert "HFVisionDataset" in names
    assert "CIFAR10Dataset" in names  # backward-compat shim
    assert "Food101Dataset" in names  # backward-compat shim


def test_create_dataset_synthtest():
    ds = create_dataset("SynthTestDataset", tensors_num=10, seed=0)
    assert isinstance(ds, SynthTestDataset)
    assert len(ds) == 10


# ----------------------------------------------------------- SynthTestDataset
def test_synth_dataset_iter_shapes():
    ds = SynthTestDataset(tensors_num=5, seed=0)
    tensors = list(ds)
    assert len(tensors) == 5
    for t in tensors:
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3, 224, 224)


def test_synth_dataset_indexing():
    ds = SynthTestDataset(tensors_num=10, seed=42)
    t0 = ds[0]
    assert t0.shape == (3, 224, 224)


def test_synth_dataset_negative_index():
    ds = SynthTestDataset(tensors_num=5, seed=42)
    last = ds[-1]
    other_last = ds[4]
    assert torch.allclose(last, other_last)


def test_synth_dataset_oob_raises():
    ds = SynthTestDataset(tensors_num=3, seed=0)
    with pytest.raises(IndexError):
        ds[5]


def test_synth_dataset_seed_deterministic():
    ds1 = SynthTestDataset(tensors_num=3, seed=7)
    ds2 = SynthTestDataset(tensors_num=3, seed=7)
    for a, b in zip(ds1, ds2):
        assert torch.allclose(a, b)


def test_hf_registry_is_not_empty():
    assert len(HF_DATASET_REGISTRY) > 0
