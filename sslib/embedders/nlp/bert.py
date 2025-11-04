"""BERT embedder implementation."""

from typing import Dict, Any, ClassVar

from .bert_base import TransformerEmbedderBase


class BERTEmbedder(TransformerEmbedderBase):
    """BERT embedder for natural language processing."""

    # Class-level metadata
    _embedder_category: ClassVar[str] = "nlp"
    _embedder_modality: ClassVar[str] = "text"
    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "model_family": "BERT",
        "source": "Google",
        "masked_language_model": True,
        "bidirectional": True,
        "architecture": "Transformer",
        "max_sequence_length": 512,
        "pooling_strategy": "cls",
    }

    MODEL_FAMILY_NAME: ClassVar[str] = "BERT"
    DEFAULT_MODEL: ClassVar[str] = "bert-base-uncased"

    AVAILABLE_MODELS = {
        "bert-base-uncased": {
            "embedding_dim": 768,
            "hf_name": "bert-base-uncased",
            "languages": ["en"],
        },
        "bert-base-cased": {
            "embedding_dim": 768,
            "hf_name": "bert-base-cased",
            "languages": ["en"],
        },
        "bert-large-uncased": {
            "embedding_dim": 1024,
            "hf_name": "bert-large-uncased",
            "languages": ["en"],
        },
        "bert-large-cased": {
            "embedding_dim": 1024,
            "hf_name": "bert-large-cased",
            "languages": ["en"],
        },
        "bert-base-multilingual-cased": {
            "embedding_dim": 768,
            "hf_name": "bert-base-multilingual-cased",
            "languages": ["multilingual"],
        },
    }

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling: str = "cls",
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize BERT embedder.

        Args:
            model_name: Name of the BERT model to use
            pooling: Pooling strategy ('cls' or 'mean')
            device: Device to run on ('cpu' or 'cuda')
            **kwargs: Additional arguments
        """
        super().__init__(model_name, pooling, device, **kwargs)

    def _get_model_metadata(self) -> Dict[str, Any]:
        """Get BERT-specific metadata."""
        return {
            "model_name": self.model_name,
            "hf_name": self.hf_name,
            "embedding_dim": self.embedding_dim,
            "pooling": self.pooling,
            "model_family": "BERT",
            "languages": self.AVAILABLE_MODELS[self.model_name]["languages"],
        }
