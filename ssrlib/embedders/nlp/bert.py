"""HuggingFace BERT-family text embedder.

Generic wrapper around any BERT-style transformer (BERT, DistilBERT, RoBERTa,
ModernBERT, …). For specialised retrieval models like E5 use the dedicated
class in ``e5.py`` so the prompt prefixes are applied correctly.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List

import numpy as np
import torch

from ..base import BaseEmbedder

logger = logging.getLogger(__name__)


class BERTEmbedder(BaseEmbedder):
    """BERT-family text embedder.

    Dataset samples are expected to be either strings or already-tokenised
    tensors. By default we use the ``[CLS]`` token's last hidden state.

    Args:
        model_name: HuggingFace model id (``bert-base-uncased``, etc.).
        pooling: ``"cls"`` or ``"mean"``.
        max_length: max sequence length passed to the tokenizer.
    """

    _embedder_category: ClassVar[str] = "nlp"
    _embedder_modality: ClassVar[str] = "text"
    _embedder_properties: ClassVar[Dict[str, Any]] = {"framework": "transformers"}

    AVAILABLE_MODELS: ClassVar[Dict[str, int]] = {
        "bert-base-uncased": 768,
        "bert-large-uncased": 1024,
        "distilbert-base-uncased": 768,
        "roberta-base": 768,
        "roberta-large": 1024,
    }

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling: str = "cls",
        max_length: int = 128,
        device: str = "cpu",
        **kwargs,
    ):
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")
        super().__init__(f"BERT-{model_name.split('/')[-1]}", device=device, **kwargs)
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = int(max_length)
        self._embedding_dim = self.AVAILABLE_MODELS.get(model_name)
        self.tokenizer = None
        self._metadata.update(
            {"model_name": model_name, "pooling": pooling, "max_length": max_length}
        )

    def load_model(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading BERT model %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval().to(self.device)
        if self._embedding_dim is None:
            self._embedding_dim = int(self.model.config.hidden_size)
        self._loaded = True

    def get_embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self.load_model()
        return int(self._embedding_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            self.load_model()
        outputs = self.model(input_ids=batch.to(self.device))
        return self._pool(outputs.last_hidden_state)

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        return hidden.mean(dim=1)

    @torch.no_grad()
    def embed_dataset(self, dataset, batch_size: int = 32) -> np.ndarray:
        """Specialised path that tokenises strings on the fly."""
        if not self._loaded:
            self.load_model()

        all_embeddings: List[np.ndarray] = []
        text_batch: List[str] = []
        tensor_batch: List[torch.Tensor] = []

        for sample in dataset:
            if isinstance(sample, str):
                text_batch.append(sample)
                if len(text_batch) >= batch_size:
                    all_embeddings.append(self._embed_text_batch(text_batch))
                    text_batch = []
            elif isinstance(sample, torch.Tensor):
                tensor_batch.append(sample)
                if len(tensor_batch) >= batch_size:
                    out = self.forward(torch.stack(tensor_batch))
                    all_embeddings.append(out.detach().cpu().numpy())
                    tensor_batch = []
            else:
                raise TypeError(
                    f"BERTEmbedder requires str or Tensor inputs, got {type(sample)}"
                )

        if text_batch:
            all_embeddings.append(self._embed_text_batch(text_batch))
        if tensor_batch:
            out = self.forward(torch.stack(tensor_batch))
            all_embeddings.append(out.detach().cpu().numpy())

        if not all_embeddings:
            return np.empty((0, self.get_embedding_dim()), dtype=np.float32)
        return np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)

    def _embed_text_batch(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**enc)
        pooled = self._pool(outputs.last_hidden_state)
        return pooled.detach().cpu().numpy()
