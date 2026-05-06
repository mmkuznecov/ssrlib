"""Configuration management."""

from __future__ import annotations

import json
from typing import Any, Dict

import yaml


class Config:
    """Lightweight dotted-key config wrapper."""

    def __init__(self, config_dict: Dict | None = None):
        self._config: Dict[str, Any] = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value, supporting dotted access like 'model.batch_size'."""
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a config value, supporting dotted access."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load config from a YAML or JSON file."""
        with open(config_path, "r") as f:
            if config_path.endswith((".yaml", ".yml")):
                config_dict = yaml.safe_load(f)
            elif config_path.endswith(".json"):
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        return cls(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._config)
