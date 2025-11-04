"""Embedder implementations for SSLib with automatic discovery."""

import logging
from pathlib import Path
from typing import Dict, List, Type, Any
import warnings

logger = logging.getLogger(__name__)

# Import base class and registry system
from .base import BaseEmbedder
from ..core.registry import BaseRegistry, discover_components

# Type alias
EmbedderRegistry = BaseRegistry[BaseEmbedder]


def discover_embedder_classes() -> EmbedderRegistry:
    """Discover all embedder classes in the embedders module."""
    registry = EmbedderRegistry("embedder").enable_modalities()

    return discover_components(
        package_path=Path(__file__).parent,
        package_name=__name__,
        base_class=BaseEmbedder,
        registry=registry,
    )


# Perform discovery at import time
logger.debug("Starting embedder discovery...")
_embedder_registry = discover_embedder_classes()


# Convenience functions
def get_available_embedders() -> Dict[str, Type[BaseEmbedder]]:
    """Get dictionary of all available embedders."""
    return _embedder_registry._items.copy()


def get_embedder_descriptions() -> Dict[str, str]:
    """Get dictionary of embedder descriptions."""
    return _embedder_registry._descriptions.copy()


def list_embedders(category: str = None, modality: str = None) -> List[str]:
    """List available embedders with optional filtering."""
    if category:
        return _embedder_registry.list_by_category(category).get(category, [])
    elif modality:
        return _embedder_registry.list_by_modality(modality).get(modality, [])
    return _embedder_registry.list_all()


def get_embedder_info(name: str) -> Dict[str, Any]:
    """Get detailed information about an embedder."""
    return _embedder_registry.get_info(name)


def print_available_embedders() -> None:
    """Print all available embedders with descriptions."""
    _embedder_registry.print_registry()


def create_embedder(name: str, **kwargs) -> BaseEmbedder:
    """Create an embedder by name."""
    embedder_class = _embedder_registry.get(name)
    if embedder_class is None:
        available = ", ".join(_embedder_registry.list_all())
        raise ValueError(f"Unknown embedder '{name}'. Available: {available}")
    return embedder_class(**kwargs)


def get_vision_embedders() -> List[str]:
    """Get list of vision embedders."""
    return list_embedders(modality="vision")


def get_text_embedders() -> List[str]:
    """Get list of text embedders."""
    return list_embedders(modality="text")


def get_audio_embedders() -> List[str]:
    """Get list of audio embedders."""
    return list_embedders(modality="audio")


def get_multimodal_embedders() -> List[str]:
    """Get list of multimodal embedders."""
    return list_embedders(modality="multimodal")


def get_embedders_by_category(category: str) -> List[str]:
    """Get embedders by category."""
    return list_embedders(category=category)


def get_embedder_categories() -> List[str]:
    """Get list of all available categories."""
    return list(_embedder_registry._categories.keys())


def get_embedder_modalities() -> List[str]:
    """Get list of all available modalities."""
    if _embedder_registry._modalities:
        return list(set(_embedder_registry._modalities.values()))
    return []


# Create dynamic exports
_exported_classes = {}
for name, embedder_class in _embedder_registry._items.items():
    _exported_classes[name] = embedder_class

# Update module globals
globals().update(_exported_classes)

# Create __all__ dynamically
__all__ = [
    "BaseEmbedder",
    "get_available_embedders",
    "get_embedder_descriptions",
    "list_embedders",
    "get_embedder_info",
    "print_available_embedders",
    "create_embedder",
    "get_vision_embedders",
    "get_text_embedders",
    "get_audio_embedders",
    "get_multimodal_embedders",
    "get_embedders_by_category",
    "get_embedder_categories",
    "get_embedder_modalities",
    *_embedder_registry.list_all(),
]

# Log results
if logger.isEnabledFor(logging.INFO):
    logger.info(
        f"Embedder discovery complete: {len(_embedder_registry.list_all())} embedders found"
    )
    for category, embedders in _embedder_registry.list_by_category().items():
        logger.info(f"  {category}: {', '.join(embedders)}")

# Warn about errors
if _embedder_registry._discovery_errors:
    warnings.warn(
        f"Some embedder modules failed to import: {len(_embedder_registry._discovery_errors)} errors. "
        f"Run logging.getLogger('{__name__}').setLevel(logging.DEBUG) for details.",
        ImportWarning,
    )
