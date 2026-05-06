"""Core orchestration: Pipeline, PipelineResults, Config, BaseRegistry."""

from .config import Config
from .pipeline import Pipeline, PipelineResults
from .registry import BaseRegistry, discover_components, extract_description

__all__ = [
    "Pipeline",
    "PipelineResults",
    "Config",
    "BaseRegistry",
    "discover_components",
    "extract_description",
]
