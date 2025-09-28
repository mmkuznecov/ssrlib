"""Loss functions for SSLib with automatic discovery."""

import importlib
import inspect
import pkgutil
import logging
from pathlib import Path
from typing import Dict, List, Type, Any, Optional, Tuple
import warnings

# Set up logging for discovery process
logger = logging.getLogger(__name__)

# Import base classes first (before discovery to avoid circular imports)
from .base import BaseLoss, ContrastiveLossBase, DistanceMetric


class LossRegistry:
    """Registry for dynamically discovered loss functions."""
    
    def __init__(self):
        self._losses: Dict[str, Type[BaseLoss]] = {}
        self._descriptions: Dict[str, str] = {}
        self._categories: Dict[str, List[str]] = {}
        self._discovery_errors: List[str] = []
    
    def register(self, 
                 name: str, 
                 loss_class: Type[BaseLoss], 
                 description: str = "",
                 category: str = "general") -> None:
        """Register a loss function.
        
        Args:
            name: Name of the loss function
            loss_class: Loss class
            description: Description of the loss
            category: Category for organization
        """
        self._losses[name] = loss_class
        self._descriptions[name] = description
        
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
    
    def get_loss(self, name: str) -> Optional[Type[BaseLoss]]:
        """Get loss class by name."""
        return self._losses.get(name)
    
    def get_description(self, name: str) -> str:
        """Get description for a loss function."""
        return self._descriptions.get(name, "No description available.")
    
    def list_losses(self) -> List[str]:
        """List all available loss function names."""
        return list(self._losses.keys())
    
    def list_by_category(self, category: str = None) -> Dict[str, List[str]]:
        """List losses by category.
        
        Args:
            category: Specific category to list, or None for all
            
        Returns:
            Dictionary of category -> loss names
        """
        if category:
            return {category: self._categories.get(category, [])}
        return self._categories.copy()
    
    def get_loss_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive information about a loss function.
        
        Args:
            name: Loss function name
            
        Returns:
            Dictionary with loss information
        """
        loss_class = self.get_loss(name)
        if not loss_class:
            return {}
        
        # Extract class information
        info = {
            "name": name,
            "class": loss_class.__name__,
            "module": loss_class.__module__,
            "description": self.get_description(name),
            "docstring": loss_class.__doc__ or "",
            "base_classes": [cls.__name__ for cls in loss_class.__mro__[1:]],
            "is_abstract": inspect.isabstract(loss_class),
        }
        
        # Try to get initialization signature
        try:
            sig = inspect.signature(loss_class.__init__)
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
        print("Available Loss Functions:")
        print("=" * 50)
        
        for category, losses in self._categories.items():
            print(f"\n{category.upper()}:")
            for loss_name in sorted(losses):
                description = self.get_description(loss_name)
                print(f"  {loss_name}: {description}")
        
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


def categorize_loss(cls: Type[BaseLoss]) -> str:
    """Determine category for a loss function.
    
    Args:
        cls: Loss class
        
    Returns:
        Category name
    """
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
    """Discover all loss classes in the losses module.
    
    Returns:
        LossRegistry with discovered classes
    """
    registry = LossRegistry()
    
    # Get the current package path
    package_path = Path(__file__).parent
    package_name = __name__
    
    logger.debug(f"Discovering losses in {package_path}")
    
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
                # Filter for loss classes
                if (inspect.isclass(obj) and
                    issubclass(obj, BaseLoss) and
                    not inspect.isabstract(obj) and
                    obj not in [BaseLoss, ContrastiveLossBase] and
                    obj.__module__ == full_module_name):
                    
                    # Extract information
                    description = extract_description(obj)
                    category = categorize_loss(obj)
                    
                    # Register the loss
                    registry.register(name, obj, description, category)
                    
                    logger.debug(f"Registered loss: {name} ({category})")
        
        except Exception as e:
            error_msg = f"Failed to import {module_name}: {e}"
            logger.warning(error_msg)
            registry._discovery_errors.append(error_msg)
            continue
    
    logger.info(f"Discovered {len(registry.list_losses())} loss functions")
    return registry


def get_available_losses() -> Dict[str, Type[BaseLoss]]:
    """Get dictionary of all available loss functions.
    
    Returns:
        Dictionary mapping loss names to loss classes
    """
    return _loss_registry._losses.copy()


def get_loss_descriptions() -> Dict[str, str]:
    """Get dictionary of loss descriptions.
    
    Returns:
        Dictionary mapping loss names to descriptions
    """
    return _loss_registry._descriptions.copy()


def list_losses(category: str = None) -> List[str]:
    """List available loss functions.
    
    Args:
        category: Optional category filter
        
    Returns:
        List of loss function names
    """
    if category:
        return _loss_registry.list_by_category(category).get(category, [])
    return _loss_registry.list_losses()


def get_loss_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a loss function.
    
    Args:
        name: Loss function name
        
    Returns:
        Dictionary with loss information
    """
    return _loss_registry.get_loss_info(name)


def print_available_losses() -> None:
    """Print all available loss functions with descriptions."""
    _loss_registry.print_registry()


# Perform discovery at import time
logger.debug("Starting loss function discovery...")
_loss_registry = discover_loss_classes()

# Create dynamic exports
_exported_classes = {}
for name, loss_class in _loss_registry._losses.items():
    _exported_classes[name] = loss_class

# Update module globals with discovered classes
globals().update(_exported_classes)

# Create __all__ list dynamically
__all__ = [
    # Base classes
    "BaseLoss",
    "ContrastiveLossBase", 
    "DistanceMetric",
    
    # Registry functions
    "get_available_losses",
    "get_loss_descriptions", 
    "list_losses",
    "get_loss_info",
    "print_available_losses",
    
    # Dynamically discovered classes
    *_loss_registry.list_losses()
]

# Log discovery results
if logger.isEnabledFor(logging.INFO):
    logger.info(f"Loss discovery complete: {len(_loss_registry.list_losses())} losses found")
    for category, losses in _loss_registry.list_by_category().items():
        logger.info(f"  {category}: {', '.join(losses)}")

# Warn about any errors
if _loss_registry._discovery_errors:
    warnings.warn(
        f"Some loss modules failed to import: {len(_loss_registry._discovery_errors)} errors. "
        f"Run logging.getLogger('{__name__}').setLevel(logging.DEBUG) for details.",
        ImportWarning
    )


# Convenience function for backward compatibility
def create_loss(name: str, **kwargs) -> BaseLoss:
    """Create a loss function by name.
    
    Args:
        name: Loss function name
        **kwargs: Arguments to pass to loss constructor
        
    Returns:
        Instantiated loss function
        
    Raises:
        ValueError: If loss function not found
    """
    loss_class = _loss_registry.get_loss(name)
    if loss_class is None:
        available = ', '.join(_loss_registry.list_losses())
        raise ValueError(f"Unknown loss function '{name}'. Available: {available}")
    
    return loss_class(**kwargs)


# Add create_loss to exports
__all__.append("create_loss")