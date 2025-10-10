import importlib
import inspect
import pkgutil
import logging
from pathlib import Path
from typing import Dict, List, Type, Any, Optional
import warnings

# Set up logging for discovery process
logger = logging.getLogger(__name__)

# Import base class first (before discovery to avoid circular imports)
from .base import BaseDataset


class DatasetRegistry:
    """Registry for dynamically discovered dataset classes."""

    def __init__(self):
        self._datasets: Dict[str, Type[BaseDataset]] = {}
        self._descriptions: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._modalities: Dict[str, str] = {}
        self._properties: Dict[str, Dict[str, Any]] = {}
        self._discovery_errors: List[str] = []

    def register(
        self,
        name: str,
        dataset_class: Type[BaseDataset],
        description: str = "",
        category: str = "general",
        modality: str = "unknown",
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a dataset.

        Args:
            name: Name of the dataset
            dataset_class: Dataset class
            description: Description of the dataset
            category: Category for organization
            modality: Data modality (vision, text, audio, etc.)
            properties: Additional dataset properties
        """
        self._datasets[name] = dataset_class
        self._descriptions[name] = description
        self._modalities[name] = modality
        self._properties[name] = properties or {}

        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

    def get_dataset(self, name: str) -> Optional[Type[BaseDataset]]:
        """Get dataset class by name."""
        return self._datasets.get(name)

    def get_description(self, name: str) -> str:
        """Get description for a dataset."""
        return self._descriptions.get(name, "No description available.")

    def get_modality(self, name: str) -> str:
        """Get modality for a dataset."""
        return self._modalities.get(name, "unknown")

    def get_properties(self, name: str) -> Dict[str, Any]:
        """Get properties for a dataset."""
        return self._properties.get(name, {})

    def list_datasets(self) -> List[str]:
        """List all available dataset names."""
        return list(self._datasets.keys())

    def list_by_category(self, category: str = None) -> Dict[str, List[str]]:
        """List datasets by category."""
        if category:
            return {category: self._categories.get(category, [])}
        return self._categories.copy()

    def list_by_modality(self, modality: str = None) -> Dict[str, List[str]]:
        """List datasets by modality."""
        modality_groups = {}
        for name, mod in self._modalities.items():
            if mod not in modality_groups:
                modality_groups[mod] = []
            modality_groups[mod].append(name)

        if modality:
            return {modality: modality_groups.get(modality, [])}
        return modality_groups

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive information about a dataset."""
        dataset_class = self.get_dataset(name)
        if not dataset_class:
            return {}

        info = {
            "name": name,
            "class": dataset_class.__name__,
            "module": dataset_class.__module__,
            "description": self.get_description(name),
            "modality": self.get_modality(name),
            "docstring": dataset_class.__doc__ or "",
            "base_classes": [cls.__name__ for cls in dataset_class.__mro__[1:]],
            "is_abstract": inspect.isabstract(dataset_class),
            "properties": self.get_properties(name),
        }

        # Try to get initialization signature
        try:
            sig = inspect.signature(dataset_class.__init__)
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

        return info

    def print_registry(self) -> None:
        """Print formatted registry information."""
        print("Available Datasets:")
        print("=" * 50)

        # Print by category
        for category, datasets in self._categories.items():
            print(f"\n{category.upper()}:")
            for dataset_name in sorted(datasets):
                description = self.get_description(dataset_name)
                modality = self.get_modality(dataset_name)
                print(f"  {dataset_name} ({modality}): {description}")

        # Print by modality
        print(f"\nBY MODALITY:")
        modality_groups = self.list_by_modality()
        for modality, datasets in modality_groups.items():
            print(f"  {modality}: {', '.join(sorted(datasets))}")

        if self._discovery_errors:
            print(f"\nDiscovery Errors ({len(self._discovery_errors)}):")
            for error in self._discovery_errors:
                print(f"  - {error}")


def extract_description(cls: Type) -> str:
    """Extract description from class docstring.

    Args:
        cls: Class to extract description from

    Returns:
        First line or paragraph of docstring
    """
    if not cls.__doc__:
        return "No description available."

    # Clean up docstring
    docstring = inspect.cleandoc(cls.__doc__)

    # Take first line or first paragraph
    lines = docstring.split("\n")
    if lines:
        first_line = lines[0].strip()
        if first_line:
            return first_line

    # Fallback to first non-empty line
    for line in lines:
        line = line.strip()
        if line and not line.startswith("Args:") and not line.startswith("Returns:"):
            return line

    return "No description available."


def discover_dataset_classes() -> DatasetRegistry:
    """Discover all dataset classes in the datasets module.

    Returns:
        DatasetRegistry with discovered classes
    """
    registry = DatasetRegistry()

    # Get the current package path
    package_path = Path(__file__).parent
    package_name = __name__

    logger.debug(f"Discovering datasets in {package_path}")

    # Iterate through all modules in the package
    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name

        # Skip certain modules
        if module_name in ["__init__", "base"]:
            continue

        try:
            # Import the module
            full_module_name = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module_name)

            logger.debug(f"Scanning module: {full_module_name}")

            # Find all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Filter for dataset classes
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseDataset)
                    and not inspect.isabstract(obj)
                    and obj != BaseDataset
                    and obj.__module__ == full_module_name
                ):

                    # Extract information from class metadata
                    description = extract_description(obj)
                    category = obj.get_dataset_category()
                    modality = obj.get_dataset_modality()
                    properties = obj.get_dataset_properties()

                    # Register the dataset
                    registry.register(
                        name, obj, description, category, modality, properties
                    )

                    logger.debug(f"Registered dataset: {name} ({category}/{modality})")

        except Exception as e:
            error_msg = f"Failed to import {module_name}: {e}"
            logger.warning(error_msg)
            registry._discovery_errors.append(error_msg)
            continue

    logger.info(f"Discovered {len(registry.list_datasets())} datasets")
    return registry


def get_available_datasets() -> Dict[str, Type[BaseDataset]]:
    """Get dictionary of all available datasets.

    Returns:
        Dictionary mapping dataset names to dataset classes
    """
    return _dataset_registry._datasets.copy()


def get_dataset_descriptions() -> Dict[str, str]:
    """Get dictionary of dataset descriptions.

    Returns:
        Dictionary mapping dataset names to descriptions
    """
    return _dataset_registry._descriptions.copy()


def list_datasets(category: str = None, modality: str = None) -> List[str]:
    """List available datasets with optional filtering.

    Args:
        category: Optional category filter
        modality: Optional modality filter

    Returns:
        List of dataset names
    """
    if category:
        return _dataset_registry.list_by_category(category).get(category, [])
    elif modality:
        return _dataset_registry.list_by_modality(modality).get(modality, [])
    return _dataset_registry.list_datasets()


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a dataset.

    Args:
        name: Dataset name

    Returns:
        Dictionary with dataset information
    """
    return _dataset_registry.get_dataset_info(name)


def print_available_datasets() -> None:
    """Print all available datasets with descriptions."""
    _dataset_registry.print_registry()


# Perform discovery at import time
logger.debug("Starting dataset discovery...")
_dataset_registry = discover_dataset_classes()

# Create dynamic exports
_exported_classes = {}
for name, dataset_class in _dataset_registry._datasets.items():
    _exported_classes[name] = dataset_class

# Update module globals with discovered classes
globals().update(_exported_classes)

# Create __all__ list dynamically
__all__ = [
    # Base class
    "BaseDataset",
    # Registry functions
    "get_available_datasets",
    "get_dataset_descriptions",
    "list_datasets",
    "get_dataset_info",
    "print_available_datasets",
    # Dynamically discovered classes
    *_dataset_registry.list_datasets(),
]

# Log discovery results
if logger.isEnabledFor(logging.INFO):
    logger.info(
        f"Dataset discovery complete: {len(_dataset_registry.list_datasets())} datasets found"
    )
    for category, datasets in _dataset_registry.list_by_category().items():
        logger.info(f"  {category}: {', '.join(datasets)}")

# Warn about any errors
if _dataset_registry._discovery_errors:
    warnings.warn(
        f"Some dataset modules failed to import: {len(_dataset_registry._discovery_errors)} errors. "
        f"Run logging.getLogger('{__name__}').setLevel(logging.DEBUG) for details.",
        ImportWarning,
    )


# Convenience functions for backward compatibility and ease of use
def create_dataset(name: str, **kwargs) -> BaseDataset:
    """Create a dataset by name.

    Args:
        name: Dataset name
        **kwargs: Arguments to pass to dataset constructor

    Returns:
        Instantiated dataset

    Raises:
        ValueError: If dataset not found
    """
    dataset_class = _dataset_registry.get_dataset(name)
    if dataset_class is None:
        available = ", ".join(_dataset_registry.list_datasets())
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
    """Get datasets by category.

    Args:
        category: Category name (e.g., 'vision', 'synthetic', 'text')

    Returns:
        List of dataset names in category
    """
    return list_datasets(category=category)


def get_dataset_categories() -> List[str]:
    """Get list of all available categories."""
    return list(_dataset_registry._categories.keys())


def get_dataset_modalities() -> List[str]:
    """Get list of all available modalities."""
    return list(set(_dataset_registry._modalities.values()))


# Add convenience functions to exports
__all__.extend(
    [
        "create_dataset",
        "get_vision_datasets",
        "get_text_datasets",
        "get_audio_datasets",
        "get_synthetic_datasets",
        "get_datasets_by_category",
        "get_dataset_categories",
        "get_dataset_modalities",
    ]
)
