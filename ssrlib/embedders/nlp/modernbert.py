"""ModernBERT embedder implementation."""

from typing import Dict, Any, ClassVar

from .bert_base import TransformerEmbedderBase


class ModernBERTEmbedder(TransformerEmbedderBase):
    """ModernBERT embedder for natural language processing."""

    # Class-level metadata
    _embedder_category: ClassVar[str] = "nlp"
    _embedder_modality: ClassVar[str] = "text"
    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "model_family": "ModernBERT",
        "source": "Answer.AI",
        "modernized_architecture": True,
        "rotary_embeddings": True,
        "max_sequence_length": 8192,
        "pooling_strategy": "cls",
    }

    MODEL_FAMILY_NAME: ClassVar[str] = "ModernBERT"
    DEFAULT_MODEL: ClassVar[str] = "modernbert-base"

    AVAILABLE_MODELS = {
        "modernbert-base": {
            "embedding_dim": 768,
            "hf_name": "answerdotai/ModernBERT-base",
            "num_layers": 22,
        },
        "modernbert-large": {
            "embedding_dim": 1024,
            "hf_name": "answerdotai/ModernBERT-large",
            "num_layers": 28,
        },
    }

    def __init__(
        self,
        model_name: str = "modernbert-base",
        pooling: str = "cls",
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize ModernBERT embedder.

        Args:
            model_name: Name of the ModernBERT model to use
            pooling: Pooling strategy ('cls' or 'mean')
            device: Device to run on ('cpu' or 'cuda')
            **kwargs: Additional arguments
        """
        super().__init__(model_name, pooling, device, **kwargs)

    def _get_model_metadata(self) -> Dict[str, Any]:
        """Get ModernBERT-specific metadata."""
        return {
            "model_name": self.model_name,
            "hf_name": self.hf_name,
            "embedding_dim": self.embedding_dim,
            "pooling": self.pooling,
            "model_family": "ModernBERT",
            "num_layers": self.AVAILABLE_MODELS[self.model_name]["num_layers"],
        }

    def _get_default_max_length(self) -> int:
        """ModernBERT default max length is 512 (same as BERT)."""
        return 512

    def _clamp_max_length(self, max_length: int) -> int:
        """ModernBERT supports up to 8192 tokens."""
        return min(max_length, 8192)
