import torch
import torch.nn.functional as F
from typing import Optional

from .base import ContrastiveLossBase, DistanceMetric


class ContrastiveLoss(ContrastiveLossBase):
    """
    Contrastive Loss for learning embeddings.

    The contrastive loss pulls together embeddings of similar samples (positive pairs)
    and pushes apart embeddings of dissimilar samples (negative pairs).

    For positive pairs (label=0): minimize distance
    For negative pairs (label=1): maximize distance up to margin

    Loss = (1-label) * d^2 + label * max(0, margin - d)^2

    Args:
        margin: Minimum margin for negative pairs
        distance_metric: Distance metric to use
        reduction: Loss reduction method
        normalize: Whether to L2 normalize embeddings

    References:
        Hadsell et al. "Dimensionality Reduction by Learning an Invariant Mapping"
    """

    def __init__(
        self,
        margin: float = 2.0,
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        reduction: str = "mean",
        normalize: bool = False,
    ):
        """Initialize contrastive loss.

        Args:
            margin: Margin for negative pairs (default: 2.0)
            distance_metric: Distance metric to use
            reduction: Reduction method ('mean', 'sum', 'none')
            normalize: Whether to L2 normalize embeddings
        """
        super().__init__(
            margin=margin,
            distance_metric=distance_metric,
            reduction=reduction,
            normalize=normalize,
        )

    def forward(
        self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for contrastive loss.

        Args:
            output1: First set of embeddings (N, D)
            output2: Second set of embeddings (N, D)
            label: Binary labels (N,) where 0=similar, 1=dissimilar

        Returns:
            Contrastive loss
        """
        # Apply normalization if enabled
        output1, output2 = self.apply_normalization(output1, output2)

        # Compute distance
        distance = self.compute_distance(output1, output2, self.distance_metric)

        # Contrastive loss computation
        # For similar pairs (label=0): penalize large distances
        pos_loss = (1 - label) * torch.pow(distance, 2)

        # For dissimilar pairs (label=1): penalize small distances (below margin)
        neg_loss = label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        # Combined loss
        loss = pos_loss + neg_loss

        return self.apply_reduction(loss)


# Alternative implementation with squared euclidean distance (matches user's original)
class ContrastiveLossOriginal(torch.nn.Module):
    """
    Original contrastive loss implementation (for compatibility).

    This matches the user's original implementation exactly.
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLossOriginal, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1 - label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive
