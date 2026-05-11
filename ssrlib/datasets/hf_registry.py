"""Registry of supported HuggingFace datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HFDatasetInfo:
    """Static info for one HuggingFace dataset."""

    hf_id: str
    num_classes: int
    train_split: str = "train"
    test_split: str = "test"
    image_key: str = "image"
    label_key: str = "label"
    description: str = ""


HF_DATASET_REGISTRY: Dict[str, HFDatasetInfo] = {
    "food101": HFDatasetInfo(
        hf_id="ethz/food101",
        num_classes=101,
        train_split="train",
        test_split="validation",
        image_key="image",
        label_key="label",
        description="Food-101 dataset with 101 food categories",
    ),
    "cifar10": HFDatasetInfo(
        hf_id="uoft-cs/cifar10",
        num_classes=10,
        image_key="img",
        label_key="label",
        description="CIFAR-10 dataset with 10 classes",
    ),
    "cifar100": HFDatasetInfo(
        hf_id="uoft-cs/cifar100",
        num_classes=100,
        image_key="img",
        label_key="fine_label",
        description="CIFAR-100 dataset with 100 fine-grained classes",
    ),
    "sun397": HFDatasetInfo(
        hf_id="tanganke/sun397",
        num_classes=397,
        description="SUN397 scene recognition dataset with 397 categories",
    ),
    "stanford_cars": HFDatasetInfo(
        hf_id="tanganke/stanford_cars",
        num_classes=196,
        description="Stanford Cars dataset with 196 car models",
    ),
    "dtd": HFDatasetInfo(
        hf_id="tanganke/dtd",
        num_classes=47,
        description="Describable Textures Dataset with 47 texture classes",
    ),
    "oxford_pets": HFDatasetInfo(
        hf_id="timm/oxford-iiit-pet",
        num_classes=37,
        description="Oxford-IIIT Pet dataset with 37 pet breeds",
    ),
    "caltech101": HFDatasetInfo(
        hf_id="flwrlabs/caltech101",
        num_classes=101,
        train_split="train",
        test_split="train",
        description="Caltech-101 dataset with 101 object categories",
    ),
    "flowers102": HFDatasetInfo(
        hf_id="Donghyun99/Oxford-Flower-102",
        num_classes=102,
        description="Oxford Flowers-102 dataset with 102 flower species",
    ),
}


def get_hf_dataset_info(dataset_name: str) -> HFDatasetInfo:
    """Look up dataset info by short name."""
    if dataset_name not in HF_DATASET_REGISTRY:
        available = ", ".join(sorted(HF_DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown HuggingFace dataset: {dataset_name}. Available: {available}"
        )
    return HF_DATASET_REGISTRY[dataset_name]


def list_hf_datasets() -> List[str]:
    return list(HF_DATASET_REGISTRY.keys())
