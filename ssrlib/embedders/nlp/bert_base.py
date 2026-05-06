"""Convenience subclass of BERTEmbedder pinned to bert-base-uncased."""

from __future__ import annotations

from .bert import BERTEmbedder


class BERTBaseEmbedder(BERTEmbedder):
    """BERTEmbedder pinned to ``bert-base-uncased`` for backward compatibility."""

    def __init__(self, **kwargs):
        kwargs.setdefault("model_name", "bert-base-uncased")
        super().__init__(**kwargs)
