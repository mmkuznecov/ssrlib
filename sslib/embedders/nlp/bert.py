"""BERT embedder implementation."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, ClassVar, Optional

from ..base import BaseEmbedder


class BERTEmbedder(BaseEmbedder):
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
        "pooling_strategy": "cls"
    }
    
    AVAILABLE_MODELS = {
        "bert-base-uncased": {
            "embedding_dim": 768,
            "hf_name": "bert-base-uncased",
            "languages": ["en"]
        },
        "bert-base-cased": {
            "embedding_dim": 768,
            "hf_name": "bert-base-cased",
            "languages": ["en"]
        },
        "bert-large-uncased": {
            "embedding_dim": 1024,
            "hf_name": "bert-large-uncased",
            "languages": ["en"]
        },
        "bert-large-cased": {
            "embedding_dim": 1024,
            "hf_name": "bert-large-cased",
            "languages": ["en"]
        },
        "bert-base-multilingual-cased": {
            "embedding_dim": 768,
            "hf_name": "bert-base-multilingual-cased",
            "languages": ["multilingual"]
        },
    }
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 pooling: str = "cls", device: str = "cpu", **kwargs):
        """Initialize BERT embedder.
        
        Args:
            model_name: Name of the BERT model to use
            pooling: Pooling strategy ('cls' or 'mean')
            device: Device to run on ('cpu' or 'cuda')
            **kwargs: Additional arguments
        """
        super().__init__(f"BERT_{model_name}", device, **kwargs)
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model {model_name}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        if pooling not in ["cls", "mean"]:
            raise ValueError(f"Pooling must be 'cls' or 'mean', got {pooling}")
            
        self.model_name = model_name
        self.hf_name = self.AVAILABLE_MODELS[model_name]["hf_name"]
        self.embedding_dim = self.AVAILABLE_MODELS[model_name]["embedding_dim"]
        self.pooling = pooling
        self.tokenizer = None
        
        # Update metadata
        self._metadata.update({
            "model_name": model_name,
            "hf_name": self.hf_name,
            "embedding_dim": self.embedding_dim,
            "pooling": pooling,
            "model_family": "BERT",
            "languages": self.AVAILABLE_MODELS[model_name]["languages"]
        })
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
        
    def load_model(self) -> None:
        """Load BERT model from Hugging Face."""
        if self._loaded:
            return
            
        print(f"Loading BERT model: {self.hf_name}")
        try:
            self.model = AutoModel.from_pretrained(self.hf_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {str(e)}")
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, 
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on token embeddings.
        
        Args:
            token_embeddings: Token embeddings from model
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
            
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through BERT model.
        
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
            
            if self.pooling == "cls":
                # Use [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif self.pooling == "mean":
                # Use mean pooling over all tokens
                embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")
                
        return embeddings
    
    def embed_texts(self, texts: list, max_length: int = 512) -> torch.Tensor:
        """Embed a list of texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Embeddings tensor of shape (len(texts), embedding_dim)
        """
        if not self._loaded:
            self.load_model()
        
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            if self.pooling == "cls":
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif self.pooling == "mean":
                embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        return embeddings
