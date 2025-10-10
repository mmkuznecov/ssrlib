import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional

from .base import BaseLoss


class InfoNCE(BaseLoss):
    """
    InfoNCE (Information Noise Contrastive Estimation) loss for self-supervised learning.

    This contrastive loss enforces the embeddings of similar (positive) samples to be close
    and those of different (negative) samples to be distant. A query embedding is compared
    with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
        normalize: Whether to normalize embeddings before computing similarities.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

    Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = "mean",
        negative_mode: str = "unpaired",
        normalize: bool = True,
    ):
        """Initialize InfoNCE loss.

        Args:
            temperature: Temperature scaling parameter
            reduction: Loss reduction method
            negative_mode: How to handle negative keys ('paired' or 'unpaired')
            normalize: Whether to normalize embeddings
        """
        super().__init__(
            reduction=reduction,
            normalize=normalize,
            temperature=temperature,
            negative_mode=negative_mode,
        )

        assert negative_mode in [
            "paired",
            "unpaired",
        ], f"Invalid negative_mode: {negative_mode}"

        self.negative_mode = negative_mode

    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for InfoNCE loss.

        Args:
            query: Query embeddings (N, D)
            positive_key: Positive key embeddings (N, D)
            negative_keys: Optional negative key embeddings

        Returns:
            InfoNCE loss
        """
        return info_nce(
            query=query,
            positive_key=positive_key,
            negative_keys=negative_keys,
            temperature=self.temperature,
            reduction=self.reduction,
            negative_mode=self.negative_mode,
            normalize=self.normalize,
        )


def info_nce(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    negative_keys: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
    reduction: str = "mean",
    negative_mode: str = "unpaired",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Functional interface for InfoNCE loss.

    Args:
        query: Query embeddings (N, D)
        positive_key: Positive key embeddings (N, D)
        negative_keys: Optional negative key embeddings
        temperature: Temperature scaling parameter
        reduction: Loss reduction method
        negative_mode: How to handle negative keys ('paired' or 'unpaired')
        normalize: Whether to normalize embeddings

    Returns:
        InfoNCE loss
    """
    # Input validation
    if query.dim() != 2:
        raise ValueError("<query> must have 2 dimensions.")
    if positive_key.dim() != 2:
        raise ValueError("<positive_key> must have 2 dimensions.")
    if negative_keys is not None:
        if negative_mode == "unpaired" and negative_keys.dim() != 2:
            raise ValueError(
                "<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'."
            )
        if negative_mode == "paired" and negative_keys.dim() != 3:
            raise ValueError(
                "<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'."
            )

    # Check matching number of samples
    if len(query) != len(positive_key):
        raise ValueError(
            "<query> and <positive_key> must have the same number of samples."
        )
    if negative_keys is not None:
        if negative_mode == "paired" and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>."
            )

    # Check embedding dimensions match
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError(
            "Vectors of <query> and <positive_key> should have the same number of components."
        )
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError(
                "Vectors of <query> and <negative_keys> should have the same number of components."
            )

    # Normalize to unit vectors if requested
    if normalize:
        query, positive_key, negative_keys = _normalize(
            query, positive_key, negative_keys
        )

    if negative_keys is not None:
        # Explicit negative keys provided

        # Cosine similarity between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == "unpaired":
            # Cosine similarity between all query-negative combinations
            negative_logits = query @ _transpose(negative_keys)

        elif negative_mode == "paired":
            # Each query paired with its corresponding negative keys
            query_expanded = query.unsqueeze(1)  # (N, 1, D)
            negative_logits = query_expanded @ _transpose(
                negative_keys
            )  # (N, 1, D) @ (N, D, M) -> (N, 1, M)
            negative_logits = negative_logits.squeeze(1)  # (N, M)

        # Concatenate positive and negative logits
        # First column contains positive logits, rest are negative
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly other positive keys in the batch

        # Cosine similarity between all query-positive_key combinations
        logits = query @ _transpose(positive_key)

        # Positive keys are on the diagonal
        labels = torch.arange(len(query), device=query.device)

    # Apply temperature scaling and compute cross-entropy loss
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def _transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose last two dimensions."""
    return x.transpose(-2, -1)


def _normalize(*xs: torch.Tensor) -> tuple:
    """Normalize tensors to unit vectors."""
    return tuple(None if x is None else F.normalize(x, dim=-1) for x in xs)
