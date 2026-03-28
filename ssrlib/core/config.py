from typing import Dict, Any
import yaml
import json


class Config:
    """Configuration management class for ssrlib."""

    def __init__(self, config_dict: Dict = None):
        """Initialize configuration.

        Args:
            config_dict: Dictionary containing configuration
        """
        self._config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (supports dot notation like 'model.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from file.

        Args:
            config_path: Path to configuration file (.yaml or .json)

        Returns:
            Config instance
        """
        with open(config_path, "r") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                config_dict = yaml.safe_load(f)
            elif config_path.endswith(".json"):
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")

        return cls(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
