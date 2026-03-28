"""Processing for ssrlib with automatic discovery."""

import logging
from pathlib import Path
from typing import Dict, List, Type, Any
import warnings

logger = logging.getLogger(__name__)

# Import base class and registry system
from .base import BaseProcessor
from ..core.registry import BaseRegistry, discover_components

# Type alias
ProcessorRegistry = BaseRegistry[BaseProcessor]


def discover_processor_classes() -> ProcessorRegistry:
    """Discover all processor classes in the processing module."""
    registry = ProcessorRegistry("processor")

    return discover_components(
        package_path=Path(__file__).parent,
        package_name=__name__,
        base_class=BaseProcessor,
        registry=registry,
    )


# Perform discovery at import time
logger.debug("Starting processor discovery...")
_processor_registry = discover_processor_classes()


# Public API functions
def get_available_processors() -> Dict[str, Type[BaseProcessor]]:
    """Get dictionary of all available processors.

    Returns:
        Dictionary mapping processor names to their classes
    """
    return _processor_registry._items.copy()


def get_processor_descriptions() -> Dict[str, str]:
    """Get dictionary of processor descriptions.

    Returns:
        Dictionary mapping processor names to their descriptions
    """
    return _processor_registry._descriptions.copy()


def list_processors() -> List[str]:
    """List all available processor names.

    Returns:
        List of processor names
    """
    return _processor_registry.list_all()


def get_processor_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a processor.

    Args:
        name: Name of the processor

    Returns:
        Dictionary containing processor information

    Raises:
        ValueError: If processor not found
    """
    return _processor_registry.get_info(name)


def print_available_processors() -> None:
    """Print all available processors with descriptions."""
    _processor_registry.print_registry()


def create_processor(name: str, **kwargs) -> BaseProcessor:
    """Create a processor instance by name.

    Args:
        name: Name of the processor
        **kwargs: Processor-specific initialization arguments

    Returns:
        Initialized processor instance

    Raises:
        ValueError: If processor not found
    """
    processor_class = _processor_registry.get(name)
    if processor_class is None:
        available = ", ".join(_processor_registry.list_all())
        raise ValueError(f"Unknown processor '{name}'. Available: {available}")
    return processor_class(**kwargs)


# Create dynamic exports
_exported_classes = {}
for name, processor_class in _processor_registry._items.items():
    _exported_classes[name] = processor_class

# Update module globals for direct imports
globals().update(_exported_classes)

# Create __all__ dynamically
__all__ = [
    "BaseProcessor",
    "get_available_processors",
    "get_processor_descriptions",
    "list_processors",
    "get_processor_info",
    "print_available_processors",
    "create_processor",
    *_processor_registry.list_all(),
]

# Log results
if logger.isEnabledFor(logging.INFO):
    processors = _processor_registry.list_all()
    logger.info(f"Processor discovery complete: {len(processors)} processors found")
    logger.info(f"  Available: {', '.join(sorted(processors))}")

# Warn about errors
if _processor_registry._discovery_errors:
    warnings.warn(
        f"Some processor modules failed to import: {len(_processor_registry._discovery_errors)} errors. "
        f"Run logging.getLogger('{__name__}').setLevel(logging.DEBUG) for details.",
        ImportWarning,
    )
