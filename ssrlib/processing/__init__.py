"""Processors for ssrlib.

Public API:
    - Processor classes (imported explicitly so IDEs see them)
    - ``list_processors()``: names of registered processors
    - ``create_processor(name, **kwargs)``: factory by string name
    - ``get_available_processors()``: dict of name -> class

Adding a new processor: implement ``BaseProcessor.process``, then register
the class by appending it to ``_PROCESSOR_CLASSES`` below.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from .base import BaseProcessor
from .covariance import CovarianceProcessor
from .effective_rank import EffectiveRankProcessor
from .leverage_scores import LeverageScoresProcessor
from .map_reduce import MapReduceMixin
from .neural_collapse import NeuralCollapseProcessor
from .pairwise_stats import PairwiseDistanceStatsProcessor
from .spectral_quality import (
    AlphaReQProcessor,
    CoherenceProcessor,
    ConditionNumberProcessor,
    EntropyDecompositionProcessor,
    NESumProcessor,
    ParticipationRatioProcessor,
    RankMeProcessor,
)
from .spectrum import SpectrumProcessor
from .stable_rank import StableRankProcessor
from .zca import ZCAProcessor

logger = logging.getLogger(__name__)

_PROCESSOR_CLASSES: List[Type[BaseProcessor]] = [
    # core
    CovarianceProcessor,
    ZCAProcessor,
    SpectrumProcessor,
    EffectiveRankProcessor,
    StableRankProcessor,
    LeverageScoresProcessor,
    PairwiseDistanceStatsProcessor,
    # spectral quality
    NESumProcessor,
    RankMeProcessor,
    AlphaReQProcessor,
    ParticipationRatioProcessor,
    CoherenceProcessor,
    ConditionNumberProcessor,
    EntropyDecompositionProcessor,
    # supervised metrics
    NeuralCollapseProcessor,
]

_REGISTRY: Dict[str, Type[BaseProcessor]] = {
    cls.__name__: cls for cls in _PROCESSOR_CLASSES
}


def list_processors() -> List[str]:
    """Return the names of all registered processors."""
    return list(_REGISTRY.keys())


def get_available_processors() -> Dict[str, Type[BaseProcessor]]:
    """Return a dict mapping processor names to their classes."""
    return dict(_REGISTRY)


def create_processor(name: str, **kwargs) -> BaseProcessor:
    """Instantiate a processor by class name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown processor '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name](**kwargs)


__all__ = [
    "BaseProcessor",
    "MapReduceMixin",
    # processors
    "CovarianceProcessor",
    "ZCAProcessor",
    "SpectrumProcessor",
    "EffectiveRankProcessor",
    "StableRankProcessor",
    "LeverageScoresProcessor",
    "PairwiseDistanceStatsProcessor",
    "NESumProcessor",
    "RankMeProcessor",
    "AlphaReQProcessor",
    "ParticipationRatioProcessor",
    "CoherenceProcessor",
    "ConditionNumberProcessor",
    "EntropyDecompositionProcessor",
    "NeuralCollapseProcessor",
    # registry helpers
    "list_processors",
    "create_processor",
    "get_available_processors",
]
