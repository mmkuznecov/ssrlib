import importlib
import inspect
import pkgutil
import logging
from pathlib import Path
from typing import Dict, List, Type, Any, Optional, Tuple
import warnings

# Set up logging for discovery process
logger = logging.getLogger(__name__)

# Import base class first (before discovery to avoid circular imports)
from .base import BaseProcessor


class ProcessorRegistry:
    """Registry for dynamically discovered processor classes."""
    
    def __init__(self):
        self._processors: Dict[str, Type[BaseProcessor]] = {}
        self._descriptions: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._output_types: Dict[str, str] = {}
        self._properties: Dict[str, Dict[str, Any]] = {}
        self._discovery_errors: List[str] = []
    
    def register(self, 
                 name: str, 
                 processor_class: Type[BaseProcessor], 
                 description: str = "",
                 category: str = "general",
                 output_type: str = "unknown",
                 properties: Optional[Dict[str, Any]] = None) -> None:
        """Register a processor.
        
        Args:
            name: Name of the processor
            processor_class: Processor class
            description: Description of the processor
            category: Category for organization
            output_type: Type of output (matrix, embeddings, scalar, vector)
            properties: Additional processor properties
        """
        self._processors[name] = processor_class
        self._descriptions[name] = description
        self._output_types[name] = output_type
        self._properties[name] = properties or {}
        
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
    
    def get_processor(self, name: str) -> Optional[Type[BaseProcessor]]:
        """Get processor class by name."""
        return self._processors.get(name)
    
    def get_description(self, name: str) -> str:
        """Get description for a processor."""
        return self._descriptions.get(name, "No description available.")
    
    def get_output_type(self, name: str) -> str:
        """Get output type for a processor."""
        return self._output_types.get(name, "unknown")
    
    def get_properties(self, name: str) -> Dict[str, Any]:
        """Get properties for a processor."""
        return self._properties.get(name, {})
    
    def list_processors(self) -> List[str]:
        """List all available processor names."""
        return list(self._processors.keys())
    
    def list_by_category(self, category: str = None) -> Dict[str, List[str]]:
        """List processors by category.
        
        Args:
            category: Specific category to list, or None for all
            
        Returns:
            Dictionary of category -> processor names
        """
        if category:
            return {category: self._categories.get(category, [])}
        return self._categories.copy()
    
    def list_by_output_type(self, output_type: str = None) -> Dict[str, List[str]]:
        """List processors by output type.
        
        Args:
            output_type: Specific output type to list, or None for all
            
        Returns:
            Dictionary of output_type -> processor names
        """
        output_groups = {}
        for name, out_type in self._output_types.items():
            if out_type not in output_groups:
                output_groups[out_type] = []
            output_groups[out_type].append(name)
        
        if output_type:
            return {output_type: output_groups.get(output_type, [])}
        return output_groups
    
    def get_processor_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive information about a processor.
        
        Args:
            name: Processor name
            
        Returns:
            Dictionary with processor information
        """
        processor_class = self.get_processor(name)
        if not processor_class:
            return {}
        
        # Extract class information
        info = {
            "name": name,
            "class": processor_class.__name__,
            "module": processor_class.__module__,
            "description": self.get_description(name),
            "output_type": self.get_output_type(name),
            "docstring": processor_class.__doc__ or "",
            "base_classes": [cls.__name__ for cls in processor_class.__mro__[1:]],
            "is_abstract": inspect.isabstract(processor_class),
            "properties": self.get_properties(name)
        }
        
        # Try to get initialization signature
        try:
            sig = inspect.signature(processor_class.__init__)
            info["parameters"] = {
                name: {
                    "default": param.default if param.default != param.empty else None,
                    "annotation": str(param.annotation) if param.annotation != param.empty else None
                }
                for name, param in sig.parameters.items()
                if name not in ['self', 'args', 'kwargs']
            }
        except Exception as e:
            info["parameters"] = f"Error extracting parameters: {e}"
        
        return info
    
    def print_registry(self) -> None:
        """Print formatted registry information."""
        print("Available Processors:")
        print("=" * 50)
        
        # Print by category
        for category, processors in self._categories.items():
            print(f"\n{category.upper()}:")
            for processor_name in sorted(processors):
                description = self.get_description(processor_name)
                output_type = self.get_output_type(processor_name)
                print(f"  {processor_name} [{output_type}]: {description}")
        
        # Print by output type
        print(f"\nBY OUTPUT TYPE:")
        output_groups = self.list_by_output_type()
        for output_type, processors in output_groups.items():
            print(f"  {output_type}: {', '.join(sorted(processors))}")
        
        if self._discovery_errors:
            print(f"\nDiscovery Errors ({len(self._discovery_errors)}):")
            for error in self._discovery_errors:
                print(f"  - {error}")


def extract_description(cls: Type) -> str:
    """Extract description from class docstring."""
    if not cls.__doc__:
        return "No description available."
    
    # Clean up docstring
    docstring = inspect.cleandoc(cls.__doc__)
    
    # Take first line or first paragraph
    lines = docstring.split('\n')
    if lines:
        first_line = lines[0].strip()
        if first_line:
            return first_line
    
    # Fallback to first non-empty line
    for line in lines:
        line = line.strip()
        if line and not line.startswith('Args:') and not line.startswith('Returns:'):
            return line
    
    return "No description available."


def categorize_processor(cls: Type[BaseProcessor]) -> str:
    """Determine category for a processor.
    
    Args:
        cls: Processor class
        
    Returns:
        Category name
    """
    class_name = cls.__name__.lower()
    
    # Check for transformation processors
    if any(term in class_name for term in ["zca", "whiten", "normalize", "transform"]):
        return "transformation"
    
    # Check for dimensionality processors
    if any(term in class_name for term in ["rank", "dimension", "intrinsic"]):
        return "dimensionality"
    
    # Check for statistical processors
    if any(term in class_name for term in ["covariance", "correlation", "cov", "leverage", "score"]):
        return "statistical"
    
    # Check docstring for hints
    if cls.__doc__:
        doc_lower = cls.__doc__.lower()
        if any(term in doc_lower for term in ["transform", "whiten", "normalize"]):
            return "transformation"
        elif any(term in doc_lower for term in ["rank", "dimension"]):
            return "dimensionality"
        elif any(term in doc_lower for term in ["covariance", "statistical", "leverage"]):
            return "statistical"
    
    return "analysis"


def determine_output_type(cls: Type[BaseProcessor]) -> str:
    """Determine output type for a processor.
    
    Args:
        cls: Processor class
        
    Returns:
        Output type string
    """
    class_name = cls.__name__.lower()
    
    # Check class name for output type hints
    if "covariance" in class_name or "cov" in class_name:
        return "matrix"
    elif any(term in class_name for term in ["zca", "whiten", "transform", "normalize"]):
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
        elif "whitened" in doc_lower or "transformed" in doc_lower or "normalized" in doc_lower:
            return "embeddings"
        elif "scalar" in doc_lower or "single value" in doc_lower:
            return "scalar"
        elif "vector" in doc_lower or "scores" in doc_lower:
            return "vector"
    
    # Try to inspect metadata if available
    try:
        # Create temporary instance to check metadata
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
        if attr_name.startswith('_') and attr_name.endswith('_'):
            continue
        
        try:
            attr_value = getattr(cls, attr_name)
            if not callable(attr_value) and not inspect.ismethod(attr_value):
                # Include non-callable class attributes
                if attr_name in ['default_epsilon', 'default_center', 'requires_centering']:
                    properties[attr_name] = attr_value
        except Exception:
            continue
    
    # Try to extract info from __init__ signature
    try:
        sig = inspect.signature(cls.__init__)
        defaults = {}
        for name, param in sig.parameters.items():
            if name in ['self', 'args', 'kwargs']:
                continue
            if param.default != param.empty:
                defaults[f"default_{name}"] = param.default
        if defaults:
            properties.update(defaults)
    except Exception:
        pass
    
    return properties


def discover_processor_classes() -> ProcessorRegistry:
    """Discover all processor classes in the processing module.
    
    Returns:
        ProcessorRegistry with discovered classes
    """
    registry = ProcessorRegistry()
    
    # Get the current package path
    package_path = Path(__file__).parent
    package_name = __name__
    
    logger.debug(f"Discovering processors in {package_path}")
    
    # Iterate through all modules in the package
    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name
        
        # Skip certain modules
        if module_name in ['__init__', 'base']:
            continue
        
        try:
            # Import the module
            full_module_name = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module_name)
            
            logger.debug(f"Scanning module: {full_module_name}")
            
            # Find all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Filter for processor classes
                if (inspect.isclass(obj) and
                    issubclass(obj, BaseProcessor) and
                    not inspect.isabstract(obj) and
                    obj != BaseProcessor and
                    obj.__module__ == full_module_name):
                    
                    # Extract information
                    description = extract_description(obj)
                    category = categorize_processor(obj)
                    output_type = determine_output_type(obj)
                    properties = extract_processor_properties(obj)
                    
                    # Register the processor
                    registry.register(name, obj, description, category, output_type, properties)
                    
                    logger.debug(f"Registered processor: {name} ({category}/{output_type})")
        
        except Exception as e:
            error_msg = f"Failed to import {module_name}: {e}"
            logger.warning(error_msg)
            registry._discovery_errors.append(error_msg)
            continue
    
    logger.info(f"Discovered {len(registry.list_processors())} processors")
    return registry


def get_available_processors() -> Dict[str, Type[BaseProcessor]]:
    """Get dictionary of all available processors."""
    return _processor_registry._processors.copy()


def get_processor_descriptions() -> Dict[str, str]:
    """Get dictionary of processor descriptions."""
    return _processor_registry._descriptions.copy()


def list_processors(category: str = None, output_type: str = None) -> List[str]:
    """List available processors with optional filtering.
    
    Args:
        category: Optional category filter
        output_type: Optional output type filter
        
    Returns:
        List of processor names
    """
    if category:
        return _processor_registry.list_by_category(category).get(category, [])
    elif output_type:
        return _processor_registry.list_by_output_type(output_type).get(output_type, [])
    return _processor_registry.list_processors()


def get_processor_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a processor."""
    return _processor_registry.get_processor_info(name)


def print_available_processors() -> None:
    """Print all available processors with descriptions."""
    _processor_registry.print_registry()


# Perform discovery at import time
logger.debug("Starting processor discovery...")
_processor_registry = discover_processor_classes()

# Create dynamic exports
_exported_classes = {}
for name, processor_class in _processor_registry._processors.items():
    _exported_classes[name] = processor_class

# Update module globals with discovered classes
globals().update(_exported_classes)

# Create __all__ list dynamically
__all__ = [
    # Base class
    "BaseProcessor",
    
    # Registry functions
    "get_available_processors",
    "get_processor_descriptions",
    "list_processors",
    "get_processor_info",
    "print_available_processors",
    
    # Dynamically discovered classes
    *_processor_registry.list_processors()
]

# Log discovery results
if logger.isEnabledFor(logging.INFO):
    logger.info(f"Processor discovery complete: {len(_processor_registry.list_processors())} processors found")
    for category, processors in _processor_registry.list_by_category().items():
        logger.info(f"  {category}: {', '.join(processors)}")

# Warn about any errors
if _processor_registry._discovery_errors:
    warnings.warn(
        f"Some processor modules failed to import: {len(_processor_registry._discovery_errors)} errors. "
        f"Run logging.getLogger('{__name__}').setLevel(logging.DEBUG) for details.",
        ImportWarning
    )


# Convenience functions
def create_processor(name: str, **kwargs) -> BaseProcessor:
    """Create a processor by name.
    
    Args:
        name: Processor name
        **kwargs: Arguments to pass to processor constructor
        
    Returns:
        Instantiated processor
        
    Raises:
        ValueError: If processor not found
    """
    processor_class = _processor_registry.get_processor(name)
    if processor_class is None:
        available = ', '.join(_processor_registry.list_processors())
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
    """Get processors by output type.
    
    Args:
        output_type: Output type ('matrix', 'embeddings', 'scalar', 'vector')
        
    Returns:
        List of processor names with that output type
    """
    return list_processors(output_type=output_type)


def get_processor_categories() -> List[str]:
    """Get list of all available categories."""
    return list(_processor_registry._categories.keys())


def get_processor_output_types() -> List[str]:
    """Get list of all available output types."""
    return list(set(_processor_registry._output_types.values()))


# Add convenience functions to exports
__all__.extend([
    "create_processor",
    "get_transformation_processors",
    "get_statistical_processors",
    "get_dimensionality_processors",
    "get_analysis_processors",
    "get_processors_by_output",
    "get_processor_categories",
    "get_processor_output_types"
])