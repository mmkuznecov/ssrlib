"""Registry for Hugging Face datasets."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class HFDatasetInfo:
    """Information about a HuggingFace dataset."""

    hf_id: str  # HuggingFace dataset identifier
    num_classes: int
    train_split: str = "train"
    test_split: str = "test"
    image_key: str = "image"
    label_key: str = "label"
    description: str = ""


# Registry of supported HuggingFace datasets
HF_DATASET_REGISTRY: Dict[str, HFDatasetInfo] = {
    "food101": HFDatasetInfo(
        hf_id="ethz/food101",
        num_classes=101,
        train_split="train",
        test_split="validation",  # Food101 uses 'validation' instead of 'test'
        image_key="image",
        label_key="label",
        description="Food-101 dataset with 101 food categories",
    ),
    "cifar10": HFDatasetInfo(
        hf_id="uoft-cs/cifar10",
        num_classes=10,
        train_split="train",
        test_split="test",
        image_key="img",
        label_key="label",
        description="CIFAR-10 dataset with 10 classes",
    ),
    "cifar100": HFDatasetInfo(
        hf_id="uoft-cs/cifar100",
        num_classes=100,
        train_split="train",
        test_split="test",
        image_key="img",
        label_key="fine_label",  # CIFAR-100 uses 'fine_label'
        description="CIFAR-100 dataset with 100 fine-grained classes",
    ),
    "sun397": HFDatasetInfo(
        hf_id="tanganke/sun397",
        num_classes=397,
        train_split="train",
        test_split="test",
        image_key="image",
        label_key="label",
        description="SUN397 scene recognition dataset with 397 categories",
    ),
    "stanford_cars": HFDatasetInfo(
        hf_id="tanganke/stanford_cars",
        num_classes=196,
        train_split="train",
        test_split="test",
        image_key="image",
        label_key="label",
        description="Stanford Cars dataset with 196 car models",
    ),
    "dtd": HFDatasetInfo(
        hf_id="tanganke/dtd",
        num_classes=47,
        train_split="train",
        test_split="test",
        image_key="image",
        label_key="label",
        description="Describable Textures Dataset with 47 texture classes",
    ),
    "oxford_pets": HFDatasetInfo(
        hf_id="timm/oxford-iiit-pet",
        num_classes=37,
        train_split="train",
        test_split="test",
        image_key="image",
        label_key="label",
        description="Oxford-IIIT Pet dataset with 37 pet breeds",
    ),
    "caltech101": HFDatasetInfo(
        hf_id="flwrlabs/caltech101",
        num_classes=101,
        train_split="train",
        test_split="train",  # Only has train split
        image_key="image",
        label_key="label",
        description="Caltech-101 dataset with 101 object categories",
    ),
    "flowers102": HFDatasetInfo(
        hf_id="Donghyun99/Oxford-Flower-102",
        num_classes=102,
        train_split="train",
        test_split="test",
        image_key="image",
        label_key="label",
        description="Oxford Flowers-102 dataset with 102 flower species",
    ),
}


def get_hf_dataset_info(dataset_name: str) -> HFDatasetInfo:
    """
    Get HuggingFace dataset information.

    Args:
        dataset_name: Dataset name

    Returns:
        HFDatasetInfo object

    Raises:
        ValueError: If dataset not in registry
    """
    if dataset_name not in HF_DATASET_REGISTRY:
        available = ", ".join(HF_DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown HuggingFace dataset: {dataset_name}. " f"Available: {available}")

    return HF_DATASET_REGISTRY[dataset_name]


def list_hf_datasets() -> list:
    """Get list of available HuggingFace datasets."""
    return list(HF_DATASET_REGISTRY.keys())
