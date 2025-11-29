"""E5 Multilingual embedder implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, ClassVar, Optional

from ..base import BaseEmbedder


class E5Embedder(BaseEmbedder):
    """E5 Multilingual embedder for natural language processing."""

    # Class-level metadata
    _embedder_category: ClassVar[str] = "nlp"
    _embedder_modality: ClassVar[str] = "text"
    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "model_family": "E5",
        "source": "Microsoft",
        "multilingual": True,
        "instruction_following": True,
        "contrastive_learning": True,
        "max_sequence_length": 512,
        "pooling_strategy": "mean",
        "normalize_embeddings": True,
        "supports_94_languages": True,
    }

    AVAILABLE_MODELS = {
        "e5-small": {
            "embedding_dim": 384,
            "hf_name": "intfloat/e5-small-v2",
            "multilingual": False,
        },
        "e5-base": {
            "embedding_dim": 768,
            "hf_name": "intfloat/e5-base-v2",
            "multilingual": False,
        },
        "e5-large": {
            "embedding_dim": 1024,
            "hf_name": "intfloat/e5-large-v2",
            "multilingual": False,
        },
        "multilingual-e5-small": {
            "embedding_dim": 384,
            "hf_name": "intfloat/multilingual-e5-small",
            "multilingual": True,
        },
        "multilingual-e5-base": {
            "embedding_dim": 768,
            "hf_name": "intfloat/multilingual-e5-base",
            "multilingual": True,
        },
        "multilingual-e5-large": {
            "embedding_dim": 1024,
            "hf_name": "intfloat/multilingual-e5-large",
            "multilingual": True,
        },
        "multilingual-e5-large-instruct": {
            "embedding_dim": 1024,
            "hf_name": "intfloat/multilingual-e5-large-instruct",
            "multilingual": True,
            "instruction_following": True,
        },
    }

    def __init__(
        self,
        model_name: str = "multilingual-e5-base",
        device: str = "cpu",
        normalize: bool = True,
        **kwargs,
    ):
        """Initialize E5 embedder.

        Args:
            model_name: Name of the E5 model to use
            device: Device to run on ('cpu' or 'cuda')
            normalize: Whether to normalize embeddings
            **kwargs: Additional arguments
        """
        super().__init__(f"E5_{model_name}", device, **kwargs)

        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model {model_name}. " f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_name = model_name
        self.hf_name = self.AVAILABLE_MODELS[model_name]["hf_name"]
        self.embedding_dim = self.AVAILABLE_MODELS[model_name]["embedding_dim"]
        self.normalize = normalize
        self.tokenizer = None
        self.is_instruct = self.AVAILABLE_MODELS[model_name].get("instruction_following", False)

        # Update metadata
        self._metadata.update(
            {
                "model_name": model_name,
                "hf_name": self.hf_name,
                "embedding_dim": self.embedding_dim,
                "normalize": normalize,
                "model_family": "E5",
                "multilingual": self.AVAILABLE_MODELS[model_name]["multilingual"],
                "instruction_following": self.is_instruct,
            }
        )

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def load_model(self) -> None:
        """Load E5 model from Hugging Face."""
        if self._loaded:
            return

        print(f"Loading E5 model: {self.hf_name}")
        try:
            self.model = AutoModel.from_pretrained(self.hf_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {str(e)}")

    def _average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Perform average pooling on token embeddings.

        This is the official pooling method for E5 models.

        Args:
            last_hidden_states: Token embeddings from model
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through E5 model.

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

            # Use average pooling (E5's official method)
            embeddings = self._average_pool(outputs.last_hidden_state, attention_mask)

            # Normalize if requested
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def embed_texts(
        self, texts: list, max_length: int = 512, task_instruction: Optional[str] = None
    ) -> torch.Tensor:
        """Embed a list of texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            task_instruction: Optional task instruction (for instruct models)
                Example: "Given a web search query, retrieve relevant passages"

        Returns:
            Embeddings tensor of shape (len(texts), embedding_dim)
        """
        if not self._loaded:
            self.load_model()

        # Add instruction prefix if using instruct model
        if self.is_instruct and task_instruction:
            texts = [f"Instruct: {task_instruction}\nQuery: {text}" for text in texts]

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

            # Use average pooling (E5's official method)
            embeddings = self._average_pool(outputs.last_hidden_state, attention_mask)

            # Normalize if requested
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def embed_queries_and_documents(
        self, queries: list, documents: list, task_instruction: Optional[str] = None
    ) -> tuple:
        """Embed queries and documents separately (for retrieval tasks).

        For instruct models, queries get the instruction prefix but documents don't.

        Args:
            queries: List of query strings
            documents: List of document strings
            task_instruction: Task instruction for queries (instruct models only)

        Returns:
            Tuple of (query_embeddings, document_embeddings)
        """
        # Embed queries with instruction
        query_embeddings = self.embed_texts(queries, task_instruction=task_instruction)

        # Embed documents without instruction
        document_embeddings = self.embed_texts(documents, task_instruction=None)

        return query_embeddings, document_embeddings
