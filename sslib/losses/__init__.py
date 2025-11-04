"""Loss functions for SSLib with automatic discovery."""

import logging
from pathlib import Path
from typing import Dict, List, Type, Any
import warnings

logger = logging.getLogger(__name__)

# Import base classes and registry system
from .base import BaseLoss, ContrastiveLossBase, DistanceMetric
from ..core.registry import BaseRegistry, discover_components

# Type alias
LossRegistry = BaseRegistry[BaseLoss]


def categorize_loss(cls: Type[BaseLoss]) -> str:
    """Determine category for a loss function."""
    # Check inheritance hierarchy
    if issubclass(cls, ContrastiveLossBase):
        return "contrastive"

    # Check name patterns
    class_name = cls.__name__.lower()
    if "contrastive" in class_name or "triplet" in class_name:
        return "contrastive"
    elif "info" in class_name or "nce" in class_name:
        return "information_theory"
    elif "deepinfomax" in class_name:
        return "mutual_information"
    else:
        return "general"


def discover_loss_classes() -> LossRegistry:
    """Discover all loss classes in the losses module."""
    registry = LossRegistry("loss")

    return discover_components(
        package_path=Path(__file__).parent,
        package_name=__name__,
        base_class=BaseLoss,
        registry=registry,
        get_category_func=categorize_loss,
    )


# Perform discovery at import time
logger.debug("Starting loss function discovery...")
_loss_registry = discover_loss_classes()


# Convenience functions
def get_available_losses() -> Dict[str, Type[BaseLoss]]:
    """Get dictionary of all available loss functions."""
    return _loss_registry._items.copy()


def get_loss_descriptions() -> Dict[str, str]:
    """Get dictionary of loss descriptions."""
    return _loss_registry._descriptions.copy()


def list_losses(category: str = None) -> List[str]:
    """List available loss functions."""
    if category:
        return _loss_registry.list_by_category(category).get(category, [])
    return _loss_registry.list_all()


def get_loss_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a loss function."""
    return _loss_registry.get_info(name)


def print_available_losses() -> None:
    """Print all available loss functions with descriptions."""
    _loss_registry.print_registry()


def create_loss(name: str, **kwargs) -> BaseLoss:
    """Create a loss function by name."""
    loss_class = _loss_registry.get(name)
    if loss_class is None:
        available = ", ".join(_loss_registry.list_all())
        raise ValueError(f"Unknown loss function '{name}'. Available: {available}")
    return loss_class(**kwargs)


# Create dynamic exports
_exported_classes = {}
for name, loss_class in _loss_registry._items.items():
    _exported_classes[name] = loss_class

# Update module globals
globals().update(_exported_classes)

# Create __all__ dynamically
__all__ = [
    "BaseLoss",
    "ContrastiveLossBase",
    "DistanceMetric",
    "get_available_losses",
    "get_loss_descriptions",
    "list_losses",
    "get_loss_info",
    "print_available_losses",
    "create_loss",
    *_loss_registry.list_all(),
]

# Log results
if logger.isEnabledFor(logging.INFO):
    logger.info(
        f"Loss discovery complete: {len(_loss_registry.list_all())} losses found"
    )
    for category, losses in _loss_registry.list_by_category().items():
        logger.info(f"  {category}: {', '.join(losses)}")

# Warn about errors
if _loss_registry._discovery_errors:
    warnings.warn(
        f"Some loss modules failed to import: {len(_loss_registry._discovery_errors)} errors. "
        f"Run logging.getLogger('{__name__}').setLevel(logging.DEBUG) for details.",
        ImportWarning,
    )
