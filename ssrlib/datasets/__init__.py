"""Datasets for ssrlib with automatic discovery."""

import logging
from pathlib import Path
from typing import Dict, List, Type, Any
import warnings

logger = logging.getLogger(__name__)

# Import base class and registry system
from .base import BaseDataset
from ..core.registry import BaseRegistry, discover_components

# Import HF registry utilities
from .hf_registry import list_hf_datasets, get_hf_dataset_info

# Type alias for clarity
DatasetRegistry = BaseRegistry[BaseDataset]


def discover_dataset_classes() -> DatasetRegistry:
    """Discover all dataset classes in the datasets module."""
    registry = DatasetRegistry("dataset").enable_modalities()

    return discover_components(
        package_path=Path(__file__).parent,
        package_name=__name__,
        base_class=BaseDataset,
        registry=registry,
    )


# Perform discovery at import time
logger.debug("Starting dataset discovery...")
_dataset_registry = discover_dataset_classes()


# Convenience functions
def get_available_datasets() -> Dict[str, Type[BaseDataset]]:
    """Get dictionary of all available datasets."""
    return _dataset_registry._items.copy()


def get_dataset_descriptions() -> Dict[str, str]:
    """Get dictionary of dataset descriptions."""
    return _dataset_registry._descriptions.copy()


def list_datasets(category: str = None, modality: str = None) -> List[str]:
    """List available datasets with optional filtering."""
    if category:
        return _dataset_registry.list_by_category(category).get(category, [])
    elif modality:
        return _dataset_registry.list_by_modality(modality).get(modality, [])
    return _dataset_registry.list_all()


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a dataset."""
    return _dataset_registry.get_info(name)


def print_available_datasets() -> None:
    """Print all available datasets with descriptions."""
    _dataset_registry.print_registry()


def create_dataset(name: str, **kwargs) -> BaseDataset:
    """Create a dataset by name."""
    dataset_class = _dataset_registry.get(name)
    if dataset_class is None:
        available = ", ".join(_dataset_registry.list_all())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return dataset_class(**kwargs)


def get_vision_datasets() -> List[str]:
    """Get list of vision datasets."""
    return list_datasets(modality="vision")


def get_text_datasets() -> List[str]:
    """Get list of text datasets."""
    return list_datasets(modality="text")


def get_audio_datasets() -> List[str]:
    """Get list of audio datasets."""
    return list_datasets(modality="audio")


def get_synthetic_datasets() -> List[str]:
    """Get list of synthetic datasets."""
    return list_datasets(modality="synthetic")


def get_datasets_by_category(category: str) -> List[str]:
    """Get datasets by category."""
    return list_datasets(category=category)


def get_dataset_categories() -> List[str]:
    """Get list of all available categories."""
    return list(_dataset_registry._categories.keys())


def get_dataset_modalities() -> List[str]:
    """Get list of all available modalities."""
    if _dataset_registry._modalities:
        return list(set(_dataset_registry._modalities.values()))
    return []


# HuggingFace dataset utilities
def get_hf_datasets() -> List[str]:
    """Get list of available HuggingFace datasets."""
    return list_hf_datasets()


# Create dynamic exports
_exported_classes = {}
for name, dataset_class in _dataset_registry._items.items():
    _exported_classes[name] = dataset_class

# Update module globals
globals().update(_exported_classes)

# Create __all__ dynamically
__all__ = [
    "BaseDataset",
    "get_available_datasets",
    "get_dataset_descriptions",
    "list_datasets",
    "get_dataset_info",
    "print_available_datasets",
    "create_dataset",
    "get_vision_datasets",
    "get_text_datasets",
    "get_audio_datasets",
    "get_synthetic_datasets",
    "get_datasets_by_category",
    "get_dataset_categories",
    "get_dataset_modalities",
    "get_hf_datasets",
    "get_hf_dataset_info",
    "list_hf_datasets",
    *_dataset_registry.list_all(),
]

# Log results
if logger.isEnabledFor(logging.INFO):
    logger.info(f"Dataset discovery complete: {len(_dataset_registry.list_all())} datasets found")
    for category, datasets in _dataset_registry.list_by_category().items():
        logger.info(f"  {category}: {', '.join(datasets)}")

# Warn about errors
if _dataset_registry._discovery_errors:
    warnings.warn(
        f"Some dataset modules failed to import: {len(_dataset_registry._discovery_errors)} errors. "
        f"Run logging.getLogger('{__name__}').setLevel(logging.DEBUG) for details.",
        ImportWarning,
    )
