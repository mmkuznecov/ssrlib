"""Processing for SSLib with automatic discovery."""

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


def categorize_processor(cls: Type[BaseProcessor]) -> str:
    """Determine category for a processor."""
    class_name = cls.__name__.lower()

    # Check for transformation processors
    if any(term in class_name for term in ["zca", "whiten", "normalize", "transform"]):
        return "transformation"

    # Check for dimensionality processors
    if any(term in class_name for term in ["rank", "dimension", "intrinsic"]):
        return "dimensionality"

    # Check for statistical processors
    if any(
        term in class_name
        for term in ["covariance", "correlation", "cov", "leverage", "score"]
    ):
        return "statistical"

    # Check docstring for hints
    if cls.__doc__:
        doc_lower = cls.__doc__.lower()
        if any(term in doc_lower for term in ["transform", "whiten", "normalize"]):
            return "transformation"
        elif any(term in doc_lower for term in ["rank", "dimension"]):
            return "dimensionality"
        elif any(
            term in doc_lower for term in ["covariance", "statistical", "leverage"]
        ):
            return "statistical"

    return "analysis"


def determine_output_type(cls: Type[BaseProcessor]) -> str:
    """Determine output type for a processor."""
    class_name = cls.__name__.lower()

    # Check class name for output type hints
    if "covariance" in class_name or "cov" in class_name:
        return "matrix"
    elif any(
        term in class_name for term in ["zca", "whiten", "transform", "normalize"]
    ):
        return "embeddings"
    elif any(term in class_name for term in ["rank", "dimension"]):
        return "scalar"
    elif "leverage" in class_name or "score" in class_name:
        return "vector"

    # Check docstring for output type hints
    if cls.__doc__:
        doc_lower = cls.__doc__.lower()
        if "matrix" in doc_lower or "covariance" in doc_lower:
            return "matrix"
        elif (
            "whitened" in doc_lower
            or "transformed" in doc_lower
            or "normalized" in doc_lower
        ):
            return "embeddings"
        elif "scalar" in doc_lower or "single value" in doc_lower:
            return "scalar"
        elif "vector" in doc_lower or "scores" in doc_lower:
            return "vector"

    # Try to inspect metadata if available
    try:
        temp_instance = cls()
        metadata = temp_instance.get_metadata()
        if "output_type" in metadata:
            output_type = metadata["output_type"]
            if "matrix" in output_type:
                return "matrix"
            elif "embeddings" in output_type or "whitened" in output_type:
                return "embeddings"
            elif "scalar" in output_type:
                return "scalar"
            elif "vector" in output_type or "scores" in output_type:
                return "vector"
    except Exception:
        pass

    return "unknown"


def extract_processor_properties(cls: Type[BaseProcessor]) -> Dict[str, Any]:
    """Extract processor-specific properties."""
    properties = {}

    # Check for class-level attributes
    for attr_name in dir(cls):
        if attr_name.startswith("_") and attr_name.endswith("_"):
            continue

        try:
            attr_value = getattr(cls, attr_name)
            if not callable(attr_value):
                if attr_name in [
                    "default_epsilon",
                    "default_center",
                    "requires_centering",
                ]:
                    properties[attr_name] = attr_value
        except Exception:
            continue

    return properties


def discover_processor_classes() -> ProcessorRegistry:
    """Discover all processor classes in the processing module."""
    registry = ProcessorRegistry("processor").enable_output_types()

    return discover_components(
        package_path=Path(__file__).parent,
        package_name=__name__,
        base_class=BaseProcessor,
        registry=registry,
        get_category_func=categorize_processor,
        get_output_type_func=determine_output_type,
        get_properties_func=extract_processor_properties,
    )


# Perform discovery at import time
logger.debug("Starting processor discovery...")
_processor_registry = discover_processor_classes()


# Convenience functions
def get_available_processors() -> Dict[str, Type[BaseProcessor]]:
    """Get dictionary of all available processors."""
    return _processor_registry._items.copy()


def get_processor_descriptions() -> Dict[str, str]:
    """Get dictionary of processor descriptions."""
    return _processor_registry._descriptions.copy()


def list_processors(category: str = None, output_type: str = None) -> List[str]:
    """List available processors with optional filtering."""
    if category:
        return _processor_registry.list_by_category(category).get(category, [])
    elif output_type:
        return _processor_registry.list_by_output_type(output_type).get(output_type, [])
    return _processor_registry.list_all()


def get_processor_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a processor."""
    return _processor_registry.get_info(name)


def print_available_processors() -> None:
    """Print all available processors with descriptions."""
    _processor_registry.print_registry()


def create_processor(name: str, **kwargs) -> BaseProcessor:
    """Create a processor by name."""
    processor_class = _processor_registry.get(name)
    if processor_class is None:
        available = ", ".join(_processor_registry.list_all())
        raise ValueError(f"Unknown processor '{name}'. Available: {available}")
    return processor_class(**kwargs)


def get_transformation_processors() -> List[str]:
    """Get list of transformation processors."""
    return list_processors(category="transformation")


def get_statistical_processors() -> List[str]:
    """Get list of statistical processors."""
    return list_processors(category="statistical")


def get_dimensionality_processors() -> List[str]:
    """Get list of dimensionality processors."""
    return list_processors(category="dimensionality")


def get_analysis_processors() -> List[str]:
    """Get list of analysis processors."""
    return list_processors(category="analysis")


def get_processors_by_output(output_type: str) -> List[str]:
    """Get processors by output type."""
    return list_processors(output_type=output_type)


def get_processor_categories() -> List[str]:
    """Get list of all available categories."""
    return list(_processor_registry._categories.keys())


def get_processor_output_types() -> List[str]:
    """Get list of all available output types."""
    if _processor_registry._output_types:
        return list(set(_processor_registry._output_types.values()))
    return []


# Create dynamic exports
_exported_classes = {}
for name, processor_class in _processor_registry._items.items():
    _exported_classes[name] = processor_class

# Update module globals
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
    "get_transformation_processors",
    "get_statistical_processors",
    "get_dimensionality_processors",
    "get_analysis_processors",
    "get_processors_by_output",
    "get_processor_categories",
    "get_processor_output_types",
    *_processor_registry.list_all(),
]

# Log results
if logger.isEnabledFor(logging.INFO):
    logger.info(
        f"Processor discovery complete: {len(_processor_registry.list_all())} processors found"
    )
    for category, processors in _processor_registry.list_by_category().items():
        logger.info(f"  {category}: {', '.join(processors)}")

# Warn about errors
if _processor_registry._discovery_errors:
    warnings.warn(
        f"Some processor modules failed to import: {len(_processor_registry._discovery_errors)} errors. "
        f"Run logging.getLogger('{__name__}').setLevel(logging.DEBUG) for details.",
        ImportWarning,
    )
