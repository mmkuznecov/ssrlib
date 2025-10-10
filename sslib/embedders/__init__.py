"""Embedder implementations for SSLib with automatic discovery."""

import importlib
import inspect
import pkgutil
import logging
from pathlib import Path
from typing import Dict, List, Type, Any, Optional
import warnings

# Set up logging for discovery process
logger = logging.getLogger(__name__)

# Import base class first
from .base import BaseEmbedder


class EmbedderRegistry:
    """Registry for dynamically discovered embedder classes."""

    def __init__(self):
        self._embedders: Dict[str, Type[BaseEmbedder]] = {}
        self._descriptions: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._modalities: Dict[str, str] = {}
        self._properties: Dict[str, Dict[str, Any]] = {}
        self._discovery_errors: List[str] = []

    def register(
        self,
        name: str,
        embedder_class: Type[BaseEmbedder],
        description: str = "",
        category: str = "general",
        modality: str = "unknown",
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an embedder.

        Args:
            name: Name of the embedder
            embedder_class: Embedder class
            description: Description of the embedder
            category: Category for organization
            modality: Data modality (vision, text, audio, etc.)
            properties: Additional embedder properties
        """
        self._embedders[name] = embedder_class
        self._descriptions[name] = description
        self._modalities[name] = modality
        self._properties[name] = properties or {}

        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

    def get_embedder(self, name: str) -> Optional[Type[BaseEmbedder]]:
        """Get embedder class by name."""
        return self._embedders.get(name)

    def get_description(self, name: str) -> str:
        """Get description for an embedder."""
        return self._descriptions.get(name, "No description available.")

    def get_modality(self, name: str) -> str:
        """Get modality for an embedder."""
        return self._modalities.get(name, "unknown")

    def get_properties(self, name: str) -> Dict[str, Any]:
        """Get properties for an embedder."""
        return self._properties.get(name, {})

    def list_embedders(self) -> List[str]:
        """List all available embedder names."""
        return list(self._embedders.keys())

    def list_by_category(self, category: str = None) -> Dict[str, List[str]]:
        """List embedders by category."""
        if category:
            return {category: self._categories.get(category, [])}
        return self._categories.copy()

    def list_by_modality(self, modality: str = None) -> Dict[str, List[str]]:
        """List embedders by modality."""
        modality_groups = {}
        for name, mod in self._modalities.items():
            if mod not in modality_groups:
                modality_groups[mod] = []
            modality_groups[mod].append(name)

        if modality:
            return {modality: modality_groups.get(modality, [])}
        return modality_groups

    def get_embedder_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive information about an embedder."""
        embedder_class = self.get_embedder(name)
        if not embedder_class:
            return {}

        info = {
            "name": name,
            "class": embedder_class.__name__,
            "module": embedder_class.__module__,
            "description": self.get_description(name),
            "modality": self.get_modality(name),
            "docstring": embedder_class.__doc__ or "",
            "base_classes": [cls.__name__ for cls in embedder_class.__mro__[1:]],
            "is_abstract": inspect.isabstract(embedder_class),
            "properties": self.get_properties(name),
        }

        # Try to get initialization signature
        try:
            sig = inspect.signature(embedder_class.__init__)
            info["parameters"] = {
                name: {
                    "default": param.default if param.default != param.empty else None,
                    "annotation": (
                        str(param.annotation)
                        if param.annotation != param.empty
                        else None
                    ),
                }
                for name, param in sig.parameters.items()
                if name not in ["self", "args", "kwargs"]
            }
        except Exception as e:
            info["parameters"] = f"Error extracting parameters: {e}"

        # Get available models if present
        if hasattr(embedder_class, "AVAILABLE_MODELS"):
            info["available_models"] = list(embedder_class.AVAILABLE_MODELS.keys())

        return info

    def print_registry(self) -> None:
        """Print formatted registry information."""
        print("Available Embedders:")
        print("=" * 50)

        # Print by category
        for category, embedders in self._categories.items():
            print(f"\n{category.upper()}:")
            for embedder_name in sorted(embedders):
                description = self.get_description(embedder_name)
                modality = self.get_modality(embedder_name)
                print(f"  {embedder_name} ({modality}): {description}")

        # Print by modality
        print(f"\nBY MODALITY:")
        modality_groups = self.list_by_modality()
        for modality, embedders in modality_groups.items():
            print(f"  {modality}: {', '.join(sorted(embedders))}")

        if self._discovery_errors:
            print(f"\nDiscovery Errors ({len(self._discovery_errors)}):")
            for error in self._discovery_errors:
                print(f"  - {error}")


def extract_description(cls: Type) -> str:
    """Extract description from class docstring."""
    if not cls.__doc__:
        return "No description available."

    docstring = inspect.cleandoc(cls.__doc__)
    lines = docstring.split("\n")
    if lines:
        first_line = lines[0].strip()
        if first_line:
            return first_line

    for line in lines:
        line = line.strip()
        if line and not line.startswith("Args:") and not line.startswith("Returns:"):
            return line

    return "No description available."


def discover_embedder_classes() -> EmbedderRegistry:
    """Discover all embedder classes in the embedders module.

    Returns:
        EmbedderRegistry with discovered classes
    """
    registry = EmbedderRegistry()

    # Get the current package path
    package_path = Path(__file__).parent
    package_name = __name__

    logger.debug(f"Discovering embedders in {package_path}")

    # Iterate through all modules in the package
    for module_info in pkgutil.walk_packages(
        [str(package_path)], prefix=f"{package_name}."
    ):
        module_name = module_info.name

        # Skip base module
        if module_name.endswith(".base"):
            continue

        try:
            # Import the module
            module = importlib.import_module(module_name)

            logger.debug(f"Scanning module: {module_name}")

            # Find all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Filter for embedder classes
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseEmbedder)
                    and not inspect.isabstract(obj)
                    and obj != BaseEmbedder
                    and obj.__module__ == module_name
                ):

                    # Extract information from class metadata
                    description = extract_description(obj)
                    category = obj.get_embedder_category()
                    modality = obj.get_embedder_modality()
                    properties = obj.get_embedder_properties()

                    # Register the embedder
                    registry.register(
                        name, obj, description, category, modality, properties
                    )

                    logger.debug(f"Registered embedder: {name} ({category}/{modality})")

        except Exception as e:
            error_msg = f"Failed to import {module_name}: {e}"
            logger.warning(error_msg)
            registry._discovery_errors.append(error_msg)
            continue

    logger.info(f"Discovered {len(registry.list_embedders())} embedders")
    return registry


def get_available_embedders() -> Dict[str, Type[BaseEmbedder]]:
    """Get dictionary of all available embedders."""
    return _embedder_registry._embedders.copy()


def get_embedder_descriptions() -> Dict[str, str]:
    """Get dictionary of embedder descriptions."""
    return _embedder_registry._descriptions.copy()


def list_embedders(category: str = None, modality: str = None) -> List[str]:
    """List available embedders with optional filtering."""
    if category:
        return _embedder_registry.list_by_category(category).get(category, [])
    elif modality:
        return _embedder_registry.list_by_modality(modality).get(modality, [])
    return _embedder_registry.list_embedders()


def get_embedder_info(name: str) -> Dict[str, Any]:
    """Get detailed information about an embedder."""
    return _embedder_registry.get_embedder_info(name)


def print_available_embedders() -> None:
    """Print all available embedders with descriptions."""
    _embedder_registry.print_registry()


# Perform discovery at import time
logger.debug("Starting embedder discovery...")
_embedder_registry = discover_embedder_classes()

# Create dynamic exports
_exported_classes = {}
for name, embedder_class in _embedder_registry._embedders.items():
    _exported_classes[name] = embedder_class

# Update module globals with discovered classes
globals().update(_exported_classes)

# Create __all__ list dynamically
__all__ = [
    # Base class
    "BaseEmbedder",
    # Registry functions
    "get_available_embedders",
    "get_embedder_descriptions",
    "list_embedders",
    "get_embedder_info",
    "print_available_embedders",
    # Dynamically discovered classes
    *_embedder_registry.list_embedders(),
]

# Log discovery results
if logger.isEnabledFor(logging.INFO):
    logger.info(
        f"Embedder discovery complete: {len(_embedder_registry.list_embedders())} embedders found"
    )
    for category, embedders in _embedder_registry.list_by_category().items():
        logger.info(f"  {category}: {', '.join(embedders)}")

# Warn about any errors
if _embedder_registry._discovery_errors:
    warnings.warn(
        f"Some embedder modules failed to import: {len(_embedder_registry._discovery_errors)} errors. "
        f"Run logging.getLogger('{__name__}').setLevel(logging.DEBUG) for details.",
        ImportWarning,
    )


# Convenience functions
def create_embedder(name: str, **kwargs) -> BaseEmbedder:
    """Create an embedder by name.

    Args:
        name: Embedder class name
        **kwargs: Arguments to pass to embedder constructor

    Returns:
        Instantiated embedder

    Raises:
        ValueError: If embedder not found
    """
    embedder_class = _embedder_registry.get_embedder(name)
    if embedder_class is None:
        available = ", ".join(_embedder_registry.list_embedders())
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
    return list(set(_embedder_registry._modalities.values()))


# Add convenience functions to exports
__all__.extend(
    [
        "create_embedder",
        "get_vision_embedders",
        "get_text_embedders",
        "get_audio_embedders",
        "get_multimodal_embedders",
        "get_embedders_by_category",
        "get_embedder_categories",
        "get_embedder_modalities",
    ]
)
