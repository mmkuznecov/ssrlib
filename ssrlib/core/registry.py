"""Generic registry system for ssrlib components."""

import importlib
import inspect
import pkgutil
import logging
from pathlib import Path
from typing import Dict, List, Type, Any, Optional, TypeVar, Generic, Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic type for the component class


class BaseRegistry(Generic[T]):
    """Generic registry for dynamically discovered components."""

    def __init__(self, component_type_name: str):
        """
        Args:
            component_type_name: Name of component type (e.g., 'dataset', 'embedder')
        """
        self.component_type_name = component_type_name
        self._items: Dict[str, Type[T]] = {}
        self._descriptions: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._properties: Dict[str, Dict[str, Any]] = {}
        self._discovery_errors: List[str] = []

        # Optional features
        self._modalities: Optional[Dict[str, str]] = None
        self._output_types: Optional[Dict[str, str]] = None

    def enable_modalities(self) -> "BaseRegistry[T]":
        """Enable modality tracking (for datasets, embedders)."""
        self._modalities = {}
        return self

    def enable_output_types(self) -> "BaseRegistry[T]":
        """Enable output type tracking (for processors)."""
        self._output_types = {}
        return self

    def register(
        self,
        name: str,
        item_class: Type[T],
        description: str = "",
        category: str = "general",
        modality: Optional[str] = None,
        output_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a component."""
        self._items[name] = item_class
        self._descriptions[name] = description
        self._properties[name] = properties or {}

        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

        if self._modalities is not None and modality:
            self._modalities[name] = modality

        if self._output_types is not None and output_type:
            self._output_types[name] = output_type

    def get(self, name: str) -> Optional[Type[T]]:
        """Get component class by name."""
        return self._items.get(name)

    def get_description(self, name: str) -> str:
        """Get description for a component."""
        return self._descriptions.get(name, "No description available.")

    def get_properties(self, name: str) -> Dict[str, Any]:
        """Get properties for a component."""
        return self._properties.get(name, {})

    def list_all(self) -> List[str]:
        """List all available component names."""
        return list(self._items.keys())

    def list_by_category(self, category: str = None) -> Dict[str, List[str]]:
        """List components by category."""
        if category:
            return {category: self._categories.get(category, [])}
        return self._categories.copy()

    def list_by_modality(self, modality: str = None) -> Dict[str, List[str]]:
        """List components by modality (if enabled)."""
        if self._modalities is None:
            return {}

        modality_groups = {}
        for name, mod in self._modalities.items():
            if mod not in modality_groups:
                modality_groups[mod] = []
            modality_groups[mod].append(name)

        if modality:
            return {modality: modality_groups.get(modality, [])}
        return modality_groups

    def list_by_output_type(self, output_type: str = None) -> Dict[str, List[str]]:
        """List components by output type (if enabled)."""
        if self._output_types is None:
            return {}

        output_groups = {}
        for name, out_type in self._output_types.items():
            if out_type not in output_groups:
                output_groups[out_type] = []
            output_groups[out_type].append(name)

        if output_type:
            return {output_type: output_groups.get(output_type, [])}
        return output_groups

    def get_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive information about a component."""
        item_class = self.get(name)
        if not item_class:
            return {}

        info = {
            "name": name,
            "class": item_class.__name__,
            "module": item_class.__module__,
            "description": self.get_description(name),
            "docstring": item_class.__doc__ or "",
            "base_classes": [cls.__name__ for cls in item_class.__mro__[1:]],
            "is_abstract": inspect.isabstract(item_class),
            "properties": self.get_properties(name),
        }

        # Add modality if available
        if self._modalities is not None and name in self._modalities:
            info["modality"] = self._modalities[name]

        # Add output type if available
        if self._output_types is not None and name in self._output_types:
            info["output_type"] = self._output_types[name]

        # Try to get initialization signature
        try:
            sig = inspect.signature(item_class.__init__)
            info["parameters"] = {
                param_name: {
                    "default": param.default if param.default != param.empty else None,
                    "annotation": (
                        str(param.annotation) if param.annotation != param.empty else None
                    ),
                }
                for param_name, param in sig.parameters.items()
                if param_name not in ["self", "args", "kwargs"]
            }
        except Exception as e:
            info["parameters"] = f"Error extracting parameters: {e}"

        # Check for AVAILABLE_MODELS
        if hasattr(item_class, "AVAILABLE_MODELS"):
            info["available_models"] = list(item_class.AVAILABLE_MODELS.keys())

        return info

    def print_registry(self) -> None:
        """Print formatted registry information."""
        print(f"Available {self.component_type_name.title()}s:")
        print("=" * 50)

        # Print by category
        for category, items in self._categories.items():
            print(f"\n{category.upper()}:")
            for item_name in sorted(items):
                description = self.get_description(item_name)

                # Add modality or output type if available
                suffix = ""
                if self._modalities is not None and item_name in self._modalities:
                    suffix = f" ({self._modalities[item_name]})"
                elif self._output_types is not None and item_name in self._output_types:
                    suffix = f" [{self._output_types[item_name]}]"

                print(f"  {item_name}{suffix}: {description}")

        # Print by modality if enabled
        if self._modalities is not None:
            print(f"\nBY MODALITY:")
            modality_groups = self.list_by_modality()
            for modality, items in modality_groups.items():
                print(f"  {modality}: {', '.join(sorted(items))}")

        # Print by output type if enabled
        if self._output_types is not None:
            print(f"\nBY OUTPUT TYPE:")
            output_groups = self.list_by_output_type()
            for output_type, items in output_groups.items():
                print(f"  {output_type}: {', '.join(sorted(items))}")

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


def discover_components(
    package_path: Path,
    package_name: str,
    base_class: Type[T],
    registry: BaseRegistry[T],
    get_category_func: Optional[Callable[[Type[T]], str]] = None,
    get_modality_func: Optional[Callable[[Type[T]], str]] = None,
    get_output_type_func: Optional[Callable[[Type[T]], str]] = None,
    get_properties_func: Optional[Callable[[Type[T]], Dict[str, Any]]] = None,
    skip_modules: Optional[List[str]] = None,
) -> BaseRegistry[T]:
    """
    Generic discovery function for any component type.

    Args:
        package_path: Path to the package directory
        package_name: Full package name
        base_class: Base class to filter for
        registry: Registry instance to populate
        get_category_func: Optional function to determine category from class
        get_modality_func: Optional function to determine modality from class
        get_output_type_func: Optional function to determine output type from class
        get_properties_func: Optional function to extract properties from class
        skip_modules: List of module names to skip (default: ['__init__', 'base'])
    """
    if skip_modules is None:
        skip_modules = ["__init__", "base"]

    logger.debug(f"Discovering components in {package_path}")

    # Iterate through all modules in the package
    for module_info in pkgutil.walk_packages([str(package_path)], prefix=f"{package_name}."):
        module_name = module_info.name

        # Skip specified modules
        if any(module_name.endswith(f".{skip}") for skip in skip_modules):
            continue

        try:
            module = importlib.import_module(module_name)
            logger.debug(f"Scanning module: {module_name}")

            # Find all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, base_class)
                    and not inspect.isabstract(obj)
                    and obj != base_class
                    and obj.__module__ == module_name
                ):
                    # Extract information
                    description = extract_description(obj)

                    # Get category
                    category = "general"
                    if get_category_func:
                        category = get_category_func(obj)
                    elif hasattr(obj, f"get_{registry.component_type_name}_category"):
                        category = getattr(obj, f"get_{registry.component_type_name}_category")()

                    # Get modality
                    modality = None
                    if get_modality_func:
                        modality = get_modality_func(obj)
                    elif hasattr(obj, f"get_{registry.component_type_name}_modality"):
                        modality = getattr(obj, f"get_{registry.component_type_name}_modality")()

                    # Get output type
                    output_type = None
                    if get_output_type_func:
                        output_type = get_output_type_func(obj)

                    # Get properties
                    properties = {}
                    if get_properties_func:
                        properties = get_properties_func(obj)
                    elif hasattr(obj, f"get_{registry.component_type_name}_properties"):
                        properties = getattr(
                            obj, f"get_{registry.component_type_name}_properties"
                        )()

                    # Register the component
                    registry.register(
                        name,
                        obj,
                        description,
                        category,
                        modality,
                        output_type,
                        properties,
                    )

                    logger.debug(f"Registered: {name} ({category})")

        except Exception as e:
            error_msg = f"Failed to import {module_name}: {e}"
            logger.warning(error_msg)
            registry._discovery_errors.append(error_msg)
            continue

    logger.info(f"Discovered {len(registry.list_all())} components")
    return registry
