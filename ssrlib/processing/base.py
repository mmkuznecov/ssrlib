from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseProcessor(ABC):
    """Base class for all processors in SSLib."""

    def __init__(self, name: str, **kwargs):
        """Initialize processor.

        Args:
            name: Name of the processor
            **kwargs: Additional processor-specific parameters
        """
        self.name = name
        self._metadata = {}

    @abstractmethod
    def process(self, embeddings: np.ndarray) -> np.ndarray:
        """Process embeddings.

        Args:
            embeddings: Input embeddings of shape (n_vectors, n_features)

        Returns:
            Processed embeddings or computed features
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get processor metadata."""
        return {"name": self.name, **self._metadata}
