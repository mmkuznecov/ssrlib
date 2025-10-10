import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from .base import ContrastiveLossBase, DistanceMetric


class TripletLoss(ContrastiveLossBase):
    """
    Triplet Loss implementation with various distance metrics.

    The triplet loss encourages embeddings where the distance between anchor
    and positive is smaller than the distance between anchor and negative by
    at least a margin.

    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Args:
        margin: Minimum margin between positive and negative pairs
        distance_metric: Distance metric ('euclidean', 'cosine', 'squared_euclidean', 'manhattan')
        reduction: Loss reduction ('mean', 'sum', 'none')
        normalize: Whether to L2 normalize embeddings

    References:
        Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering"
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        reduction: str = "mean",
        normalize: bool = False,
    ):
        """Initialize triplet loss.

        Args:
            margin: Minimum margin between positive and negative pairs
            distance_metric: Distance metric to use
            reduction: Loss reduction method
            normalize: Whether to L2 normalize embeddings
        """
        super().__init__(
            margin=margin,
            distance_metric=distance_metric,
            reduction=reduction,
            normalize=normalize,
        )

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for triplet loss.

        Args:
            anchor: Anchor embeddings (N, D)
            positive: Positive embeddings (N, D)
            negative: Negative embeddings (N, D)

        Returns:
            Triplet loss
        """
        # Apply normalization if enabled
        anchor, positive, negative = self.apply_normalization(
            anchor, positive, negative
        )

        # Compute distances
        pos_dist = self.compute_distance(anchor, positive, self.distance_metric)
        neg_dist = self.compute_distance(anchor, negative, self.distance_metric)

        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)

        return self.apply_reduction(loss)


class TripletLossWithMining(TripletLoss):
    """
    Triplet Loss with hard negative mining.

    This extends the basic triplet loss by automatically selecting
    hard negative examples within each batch.

    Args:
        margin: Minimum margin between positive and negative pairs
        distance_metric: Distance metric to use
        reduction: Loss reduction method
        normalize: Whether to L2 normalize embeddings
        mining_strategy: Strategy for selecting hard negatives ('hardest', 'semi_hard', 'all')
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        reduction: str = "mean",
        normalize: bool = False,
        mining_strategy: str = "hardest",
    ):
        """Initialize triplet loss with mining.

        Args:
            mining_strategy: Mining strategy ('hardest', 'semi_hard', 'all')
        """
        super().__init__(margin, distance_metric, reduction, normalize)

        assert mining_strategy in [
            "hardest",
            "semi_hard",
            "all",
        ], f"Invalid mining strategy: {mining_strategy}"
        self.mining_strategy = mining_strategy

    def mine_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Mine triplets from a batch of embeddings and labels.

        Args:
            embeddings: Batch of embeddings (N, D)
            labels: Corresponding labels (N,)

        Returns:
            Tuple of (anchor_idx, positive_idx, negative_idx)
        """
        # Compute pairwise distances
        pairwise_dist = self.compute_pairwise_distance(embeddings, self.distance_metric)

        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal

        # Remove diagonal (self-comparisons)
        labels_equal.fill_diagonal_(False)

        anchors, positives, negatives = [], [], []

        for i in range(len(embeddings)):
            # Find positive examples (same label, not self)
            positive_mask = labels_equal[i]
            if not positive_mask.any():
                continue  # Skip if no positives available

            # Find negative examples (different label)
            negative_mask = labels_not_equal[i]
            if not negative_mask.any():
                continue  # Skip if no negatives available

            # Get distances for this anchor
            pos_dists = pairwise_dist[i][positive_mask]
            neg_dists = pairwise_dist[i][negative_mask]

            pos_indices = torch.where(positive_mask)[0]
            neg_indices = torch.where(negative_mask)[0]

            if self.mining_strategy == "hardest":
                # Hardest positive (farthest positive)
                hardest_pos_idx = pos_indices[torch.argmax(pos_dists)]
                # Hardest negative (closest negative)
                hardest_neg_idx = neg_indices[torch.argmin(neg_dists)]

                anchors.append(i)
                positives.append(hardest_pos_idx)
                negatives.append(hardest_neg_idx)

            elif self.mining_strategy == "semi_hard":
                # Semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin
                hardest_pos_dist = torch.max(pos_dists)
                semi_hard_mask = (neg_dists > hardest_pos_dist) & (
                    neg_dists < hardest_pos_dist + self.margin
                )

                if semi_hard_mask.any():
                    # Choose random semi-hard negative
                    semi_hard_negs = neg_indices[semi_hard_mask]
                    chosen_neg = semi_hard_negs[
                        torch.randint(len(semi_hard_negs), (1,))
                    ]

                    anchors.append(i)
                    positives.append(pos_indices[torch.argmax(pos_dists)])
                    negatives.append(chosen_neg)

            elif self.mining_strategy == "all":
                # All valid combinations
                for pos_idx in pos_indices:
                    for neg_idx in neg_indices:
                        anchors.append(i)
                        positives.append(pos_idx)
                        negatives.append(neg_idx)

        if not anchors:
            # Return empty tensors if no valid triplets found
            return (
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
            )

        return torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic triplet mining.

        Args:
            embeddings: Batch of embeddings (N, D)
            labels: Corresponding labels (N,)

        Returns:
            Triplet loss
        """
        # Apply normalization if enabled
        embeddings = self.apply_normalization(embeddings)[0]

        # Mine triplets
        anchor_idx, pos_idx, neg_idx = self.mine_triplets(embeddings, labels)

        if len(anchor_idx) == 0:
            # Return zero loss if no valid triplets found
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Extract triplets
        anchors = embeddings[anchor_idx]
        positives = embeddings[pos_idx]
        negatives = embeddings[neg_idx]

        # Compute triplet loss
        return super().forward(anchors, positives, negatives)

    def get_config(self) -> dict:
        """Get configuration including mining strategy."""
        config = super().get_config()
        config["mining_strategy"] = self.mining_strategy
        return config
