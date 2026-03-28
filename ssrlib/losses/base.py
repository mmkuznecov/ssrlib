import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum


class DistanceMetric(Enum):
    """Distance metrics for loss functions."""

    EUCLIDEAN = "euclidean"
    SQUARED_EUCLIDEAN = "squared_euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"


class BaseLoss(nn.Module, ABC):
    """Base class for ssrlib loss functions with common functionality."""

    def __init__(
        self,
        reduction: str = "mean",
        normalize: bool = False,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """Initialize base loss.

        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
            normalize: Whether to L2 normalize embeddings
            temperature: Temperature scaling factor
            **kwargs: Additional loss-specific parameters
        """
        super().__init__()

        assert reduction in ["mean", "sum", "none"], f"Invalid reduction: {reduction}"

        self.reduction = reduction
        self.normalize = normalize
        self.temperature = temperature
        self.loss_params = kwargs

    def apply_normalization(self, *tensors: torch.Tensor) -> tuple:
        """Apply L2 normalization to tensors if enabled.

        Args:
            *tensors: Input tensors to normalize

        Returns:
            Tuple of normalized tensors
        """
        if self.normalize:
            return tuple(F.normalize(t, p=2, dim=-1) for t in tensors)
        return tensors

    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Input logits

        Returns:
            Temperature-scaled logits
        """
        if self.temperature is not None:
            return logits / self.temperature
        return logits

    def apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor.

        Args:
            loss: Loss tensor

        Returns:
            Reduced loss
        """
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def compute_distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    ) -> torch.Tensor:
        """Compute distance between tensors using specified metric.

        Args:
            x1, x2: Input tensors of shape (..., D)
            metric: Distance metric to use

        Returns:
            Distance tensor
        """
        if metric == DistanceMetric.EUCLIDEAN:
            return torch.norm(x1 - x2, p=2, dim=-1)
        elif metric == DistanceMetric.SQUARED_EUCLIDEAN:
            return torch.sum((x1 - x2) ** 2, dim=-1)
        elif metric == DistanceMetric.MANHATTAN:
            return torch.norm(x1 - x2, p=1, dim=-1)
        elif metric == DistanceMetric.COSINE:
            # Ensure normalized for cosine distance
            x1_norm = F.normalize(x1, p=2, dim=-1)
            x2_norm = F.normalize(x2, p=2, dim=-1)
            return 1 - torch.sum(x1_norm * x2_norm, dim=-1)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def compute_pairwise_distance(
        self, x: torch.Tensor, metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    ) -> torch.Tensor:
        """Compute pairwise distances within a batch.

        Args:
            x: Input tensor of shape (N, D)
            metric: Distance metric to use

        Returns:
            Pairwise distance matrix of shape (N, N)
        """
        if metric == DistanceMetric.EUCLIDEAN:
            return torch.cdist(x, x, p=2)
        elif metric == DistanceMetric.SQUARED_EUCLIDEAN:
            return torch.cdist(x, x, p=2) ** 2
        elif metric == DistanceMetric.MANHATTAN:
            return torch.cdist(x, x, p=1)
        elif metric == DistanceMetric.COSINE:
            x_norm = F.normalize(x, p=2, dim=1)
            cosine_sim = torch.mm(x_norm, x_norm.t())
            return 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the loss function."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get loss function configuration.

        Returns:
            Configuration dictionary
        """
        return {
            "reduction": self.reduction,
            "normalize": self.normalize,
            "temperature": self.temperature,
            **self.loss_params,
        }

    def __repr__(self) -> str:
        """String representation of the loss function."""
        config_str = ", ".join(f"{k}={v}" for k, v in self.get_config().items() if v is not None)
        return f"{self.__class__.__name__}({config_str})"


class ContrastiveLossBase(BaseLoss):
    """Base class for contrastive-style losses."""

    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        **kwargs,
    ):
        """Initialize contrastive loss base.

        Args:
            margin: Margin for contrastive learning
            distance_metric: Distance metric to use
            **kwargs: Additional parameters for BaseLoss
        """
        super().__init__(**kwargs)
        self.margin = margin
        self.distance_metric = distance_metric

    def get_config(self) -> Dict[str, Any]:
        """Get configuration including contrastive-specific parameters."""
        config = super().get_config()
        config.update({"margin": self.margin, "distance_metric": self.distance_metric.value})
        return config
