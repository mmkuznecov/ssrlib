"""Base class for BERT-style transformer embedders."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, ClassVar, Optional
from abc import ABC, abstractmethod

from ..base import BaseEmbedder


class TransformerEmbedderBase(BaseEmbedder, ABC):
    """Base class for BERT-style transformer embedders (BERT, ModernBERT, etc.)."""

    # Subclasses must define these
    AVAILABLE_MODELS: ClassVar[Dict[str, Dict[str, Any]]] = {}
    DEFAULT_MODEL: ClassVar[str] = ""
    MODEL_FAMILY_NAME: ClassVar[str] = ""

    def __init__(
        self,
        model_name: str,
        pooling: str = "cls",
        device: str = "cpu",
        **kwargs,
    ):
        """Initialize transformer embedder.

        Args:
            model_name: Name of the model to use
            pooling: Pooling strategy ('cls' or 'mean')
            device: Device to run on ('cpu' or 'cuda')
            **kwargs: Additional arguments
        """
        # Validate model name
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model {model_name}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        # Validate pooling
        if pooling not in ["cls", "mean"]:
            raise ValueError(f"Pooling must be 'cls' or 'mean', got {pooling}")

        # Initialize base with family-prefixed name
        super().__init__(f"{self.MODEL_FAMILY_NAME}_{model_name}", device, **kwargs)

        self.model_name = model_name
        self.hf_name = self.AVAILABLE_MODELS[model_name]["hf_name"]
        self.embedding_dim = self.AVAILABLE_MODELS[model_name]["embedding_dim"]
        self.pooling = pooling
        self.tokenizer = None

        # Update metadata with model-specific info
        self._metadata.update(self._get_model_metadata())

    @abstractmethod
    def _get_model_metadata(self) -> Dict[str, Any]:
        """Get model-specific metadata to update.

        Subclasses should override this to add model-specific metadata fields.

        Returns:
            Dictionary with metadata to update
        """
        pass

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def load_model(self) -> None:
        """Load model from Hugging Face."""
        if self._loaded:
            return

        print(f"Loading {self.MODEL_FAMILY_NAME} model: {self.hf_name}")
        try:
            self.model = AutoModel.from_pretrained(self.hf_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {str(e)}")

    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Perform mean pooling on token embeddings.

        Args:
            token_embeddings: Token embeddings from model
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _apply_pooling(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply the configured pooling strategy.

        Args:
            last_hidden_state: Model output hidden states
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        if self.pooling == "cls":
            # Use [CLS] token embedding (first token)
            return last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            # Use mean pooling over all tokens
            return self._mean_pooling(last_hidden_state, attention_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through model.

        Note: This expects pre-tokenized input_ids as tensors.
        For text input, use embed_texts() method instead.

        Args:
            batch: Input batch of token IDs of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        if not self._loaded:
            self.load_model()

        self.model.eval()
        with torch.no_grad():
            # Create attention mask (assuming padding token is 0)
            attention_mask = (batch != 0).long()

            outputs = self.model(input_ids=batch, attention_mask=attention_mask)
            embeddings = self._apply_pooling(outputs.last_hidden_state, attention_mask)

        return embeddings

    def embed_texts(
        self, texts: list, max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Embed a list of texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length (uses model default if None)

        Returns:
            Embeddings tensor of shape (len(texts), embedding_dim)
        """
        if not self._loaded:
            self.load_model()

        # Get max length for this model
        if max_length is None:
            max_length = self._get_default_max_length()
        else:
            max_length = self._clamp_max_length(max_length)

        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self._apply_pooling(outputs.last_hidden_state, attention_mask)

        return embeddings

    def _get_default_max_length(self) -> int:
        """Get default max length for this model. Override if needed."""
        return 512

    def _clamp_max_length(self, max_length: int) -> int:
        """Clamp max length to model's maximum. Override if needed."""
        return max_length
