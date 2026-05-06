"""ssrlib: a modular framework for self-supervised representation analysis.

Public API:
    - ``Pipeline``, ``PipelineResults``, ``Config`` from ``ssrlib.core``
    - ``EmbeddingProbe``, ``embedding_probe`` from ``ssrlib.analysis``
    - Processors from ``ssrlib.processing``
    - Datasets from ``ssrlib.datasets``
    - Embedders from ``ssrlib.embedders``
    - Losses from ``ssrlib.losses``
"""

from __future__ import annotations

import logging

from .analysis import EmbeddingProbe, embedding_probe
from .core import Config, Pipeline, PipelineResults

# A library should not configure handlers on its parent loggers; just attach
# a NullHandler so users see nothing by default and configure logging in
# their application as they wish.
logging.getLogger("ssrlib").addHandler(logging.NullHandler())

__version__ = "0.2.0"

__all__ = [
    "Pipeline",
    "PipelineResults",
    "Config",
    "EmbeddingProbe",
    "embedding_probe",
    "__version__",
]
