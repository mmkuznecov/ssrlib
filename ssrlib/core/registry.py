"""Generic registry system used by the embedder / dataset / loss auto-discovery.

The processing package no longer depends on this registry — its ``__init__``
uses explicit imports. Other packages still use auto-discovery for now to
preserve their per-class category / modality / property metadata.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """Generic registry for dynamically discovered components."""

    def __init__(self, component_type_name: str):
        self.component_type_name = component_type_name
        self._items: Dict[str, Type[T]] = {}
        self._descriptions: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._properties: Dict[str, Dict[str, Any]] = {}
        self._discovery_errors: List[str] = []
        self._modalities: Optional[Dict[str, str]] = None
        self._output_types: Optional[Dict[str, str]] = None

    def enable_modalities(self) -> "BaseRegistry[T]":
        self._modalities = {}
        return self

    def enable_output_types(self) -> "BaseRegistry[T]":
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
        self._items[name] = item_class
        self._descriptions[name] = description
        self._properties[name] = properties or {}
        self._categories.setdefault(category, []).append(name)
        if self._modalities is not None and modality:
            self._modalities[name] = modality
        if self._output_types is not None and output_type:
            self._output_types[name] = output_type

    def get(self, name: str) -> Optional[Type[T]]:
        return self._items.get(name)

    def get_description(self, name: str) -> str:
        return self._descriptions.get(name, "No description available.")

    def get_properties(self, name: str) -> Dict[str, Any]:
        return self._properties.get(name, {})

    def list_all(self) -> List[str]:
        return list(self._items.keys())

    def list_by_category(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        if category:
            return {category: self._categories.get(category, [])}
        return dict(self._categories)

    def list_by_modality(self, modality: Optional[str] = None) -> Dict[str, List[str]]:
        if self._modalities is None:
            return {}
        groups: Dict[str, List[str]] = {}
        for name, mod in self._modalities.items():
            groups.setdefault(mod, []).append(name)
        if modality:
            return {modality: groups.get(modality, [])}
        return groups

    def get_info(self, name: str) -> Dict[str, Any]:
        item_class = self.get(name)
        if not item_class:
            return {}
        info: Dict[str, Any] = {
            "name": name,
            "class": item_class.__name__,
            "module": item_class.__module__,
            "description": self.get_description(name),
            "docstring": item_class.__doc__ or "",
            "base_classes": [cls.__name__ for cls in item_class.__mro__[1:]],
            "is_abstract": inspect.isabstract(item_class),
            "properties": self.get_properties(name),
        }
        if self._modalities is not None and name in self._modalities:
            info["modality"] = self._modalities[name]
        if self._output_types is not None and name in self._output_types:
            info["output_type"] = self._output_types[name]
        try:
            sig = inspect.signature(item_class.__init__)
            info["parameters"] = {
                pn: {
                    "default": p.default if p.default != p.empty else None,
                    "annotation": (
                        str(p.annotation) if p.annotation != p.empty else None
                    ),
                }
                for pn, p in sig.parameters.items()
                if pn not in ("self", "args", "kwargs")
            }
        except Exception as exc:  # pragma: no cover - signature edge cases
            info["parameters"] = f"Error extracting parameters: {exc}"
        if hasattr(item_class, "AVAILABLE_MODELS"):
            info["available_models"] = list(item_class.AVAILABLE_MODELS.keys())
        return info


def extract_description(cls: Type) -> str:
    """Extract description from class docstring (first non-empty line)."""
    if not cls.__doc__:
        return "No description available."
    docstring = inspect.cleandoc(cls.__doc__)
    for line in docstring.split("\n"):
        line = line.strip()
        if line and not line.startswith(("Args:", "Returns:")):
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
    """Walk a package and register all concrete subclasses of base_class."""
    if skip_modules is None:
        skip_modules = ["__init__", "base"]

    for module_info in pkgutil.walk_packages(
        [str(package_path)], prefix=f"{package_name}."
    ):
        module_name = module_info.name
        if any(module_name.endswith(f".{skip}") for skip in skip_modules):
            continue
        try:
            module = importlib.import_module(module_name)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, base_class)
                    and not inspect.isabstract(obj)
                    and obj is not base_class
                    and obj.__module__ == module_name
                ):
                    description = extract_description(obj)
                    category = "general"
                    if get_category_func:
                        category = get_category_func(obj)
                    elif hasattr(obj, f"get_{registry.component_type_name}_category"):
                        category = getattr(
                            obj, f"get_{registry.component_type_name}_category"
                        )()
                    modality = None
                    if get_modality_func:
                        modality = get_modality_func(obj)
                    elif hasattr(obj, f"get_{registry.component_type_name}_modality"):
                        modality = getattr(
                            obj, f"get_{registry.component_type_name}_modality"
                        )()
                    output_type = None
                    if get_output_type_func:
                        output_type = get_output_type_func(obj)
                    properties: Dict[str, Any] = {}
                    if get_properties_func:
                        properties = get_properties_func(obj)
                    elif hasattr(obj, f"get_{registry.component_type_name}_properties"):
                        properties = getattr(
                            obj, f"get_{registry.component_type_name}_properties"
                        )()
                    registry.register(
                        obj.__name__,
                        obj,
                        description,
                        category,
                        modality,
                        output_type,
                        properties,
                    )
        except Exception as exc:
            registry._discovery_errors.append(f"Failed to import {module_name}: {exc}")
            logger.warning("Failed to import %s: %s", module_name, exc)
    return registry
