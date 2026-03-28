import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseLoss


# Mock discriminator classes for testing and default usage
class MockGlobalDiscriminator(nn.Module):
    """Mock global discriminator for testing purposes."""

    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(1, output_dim)  # Minimal implementation

    def forward(self, y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Forward pass for global discriminator.

        Args:
            y: Encoded representations [batch_size, encoding_dim]
            M: Feature maps [batch_size, channels, height, width]

        Returns:
            Discriminator output [batch_size, output_dim]
        """
        batch_size = y.shape[0]
        return torch.randn(batch_size, 1, device=y.device)


class MockLocalDiscriminator(nn.Module):
    """Mock local discriminator for testing purposes."""

    def __init__(self, output_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(1, output_channels, kernel_size=1)  # Minimal implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for local discriminator.

        Args:
            x: Combined feature maps [batch_size, channels, height, width]

        Returns:
            Discriminator output [batch_size, output_channels, height, width]
        """
        batch_size, channels, height, width = x.shape
        return torch.randn(batch_size, 1, height, width, device=x.device)


class MockPriorDiscriminator(nn.Module):
    """Mock prior discriminator for testing purposes."""

    def __init__(self, input_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for prior discriminator.

        Args:
            x: Input representations [batch_size, input_dim]

        Returns:
            Discriminator output [batch_size, output_dim] with sigmoid activation
        """
        batch_size = x.shape[0]
        return torch.sigmoid(torch.randn(batch_size, 1, device=x.device))


class DeepInfoMaxLoss(BaseLoss):
    """
    DeepInfoMax Loss for self-supervised representation learning.

    The DeepInfoMaxLoss maximizes mutual information between input and learned
    representations using adversarial training with three discriminators:
    - Global: Captures global mutual information
    - Local: Captures local mutual information
    - Prior: Matches representations to a prior distribution

    References:
        Hjelm et al. "Learning deep representations by mutual information estimation and maximization"
        https://arxiv.org/pdf/1808.06670.pdf

    Args:
        global_discriminator: Global discriminator instance
        local_discriminator: Local discriminator instance
        prior_discriminator: Prior discriminator instance
        alpha: Weight for global loss term
        beta: Weight for local loss term
        gamma: Weight for prior loss term
        reduction: Loss reduction method
    """

    def __init__(
        self,
        global_discriminator: Optional[nn.Module] = None,
        local_discriminator: Optional[nn.Module] = None,
        prior_discriminator: Optional[nn.Module] = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        gamma: float = 0.1,
        reduction: str = "mean",
    ):
        """Initialize DeepInfoMax loss.

        Args:
            global_discriminator: Global discriminator (uses mock if None)
            local_discriminator: Local discriminator (uses mock if None)
            prior_discriminator: Prior discriminator (uses mock if None)
            alpha: Weight for global loss term
            beta: Weight for local loss term
            gamma: Weight for prior loss term
            reduction: Loss reduction method
        """
        super().__init__(reduction=reduction, alpha=alpha, beta=beta, gamma=gamma)

        # Use provided discriminators or create default ones
        if global_discriminator is not None:
            self.global_d = global_discriminator
        else:
            try:
                # Try to import actual discriminators
                from models import GlobalDiscriminator

                self.global_d = GlobalDiscriminator()
            except ImportError:
                # Fall back to mock discriminator
                self.global_d = MockGlobalDiscriminator()

        if local_discriminator is not None:
            self.local_d = local_discriminator
        else:
            try:
                from models import LocalDiscriminator

                self.local_d = LocalDiscriminator()
            except ImportError:
                self.local_d = MockLocalDiscriminator()

        if prior_discriminator is not None:
            self.prior_d = prior_discriminator
        else:
            try:
                from models import PriorDiscriminator

                self.prior_d = PriorDiscriminator()
            except ImportError:
                self.prior_d = MockPriorDiscriminator()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y: torch.Tensor, M: torch.Tensor, M_prime: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DeepInfoMaxLoss.

        Args:
            y: Encoded representations [batch_size, encoding_dim]
            M: Feature maps [batch_size, channels, height, width]
            M_prime: Shuffled/rotated feature maps [batch_size, channels, height, width]

        Returns:
            Combined loss (LOCAL + GLOBAL + PRIOR)
        """
        # Expand y to match spatial dimensions of feature maps
        # See appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        y_exp = y.unsqueeze(-1).unsqueeze(-1)  # [batch_size, encoding_dim, 1, 1]

        # Get spatial dimensions from feature maps
        _, _, height, width = M.shape
        y_exp = y_exp.expand(-1, -1, height, width)  # [batch_size, encoding_dim, height, width]

        # Concatenate feature maps with expanded representations
        y_M = torch.cat((M, y_exp), dim=1)  # [batch_size, channels + encoding_dim, height, width]
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        # Local discriminator loss
        # E_j = E[log(T(y, x_j))] where (y, x_j) are joint samples
        # E_m = E[log(T(y, x_m))] where (y, x_m) are marginal samples
        Ej_local = -F.softplus(-self.local_d(y_M)).mean()
        Em_local = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em_local - Ej_local) * self.beta

        # Global discriminator loss
        Ej_global = -F.softplus(-self.global_d(y, M)).mean()
        Em_global = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em_global - Ej_global) * self.alpha

        # Prior discriminator loss
        # Encourages representations to match a prior distribution
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = -(term_a + term_b) * self.gamma

        # Combined loss
        total_loss = LOCAL + GLOBAL + PRIOR

        return self.apply_reduction(total_loss.unsqueeze(0))

    def get_config(self) -> dict:
        """Get configuration for the loss function."""
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "global_discriminator": type(self.global_d).__name__,
                "local_discriminator": type(self.local_d).__name__,
                "prior_discriminator": type(self.prior_d).__name__,
            }
        )
        return config


# Convenience function for creating shuffled/rotated versions of feature maps
def create_negative_samples(M: torch.Tensor, method: str = "rotate") -> torch.Tensor:
    """
    Create negative samples from feature maps for DeepInfoMax.

    Args:
        M: Feature maps [batch_size, channels, height, width]
        method: Method for creating negatives ('rotate', 'shuffle')

    Returns:
        Negative feature maps with same shape as M
    """
    if method == "rotate":
        # Rotate batch dimension (each sample paired with different sample's features)
        return torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
    elif method == "shuffle":
        # Shuffle along batch dimension
        indices = torch.randperm(M.shape[0])
        return M[indices]
    else:
        raise ValueError(f"Unknown method: {method}")


# Example usage function
def example_usage():
    """Example of how to use DeepInfoMaxLoss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create loss with mock discriminators
    loss_fn = DeepInfoMaxLoss(
        global_discriminator=MockGlobalDiscriminator(),
        local_discriminator=MockLocalDiscriminator(),
        prior_discriminator=MockPriorDiscriminator(),
        alpha=0.5,
        beta=1.0,
        gamma=0.1,
    )

    # Example data
    batch_size = 4
    encoding_dim = 64
    feature_channels = 128
    feature_size = 26

    y = torch.randn(batch_size, encoding_dim, device=device)
    M = torch.randn(batch_size, feature_channels, feature_size, feature_size, device=device)
    M_prime = create_negative_samples(M, method="rotate")

    # Compute loss
    try:
        loss = loss_fn(y, M, M_prime)
        print(f"DeepInfoMax loss: {loss.item():.4f}")
        return loss
    except Exception as e:
        print(f"Error computing loss: {e}")
        return None


if __name__ == "__main__":
    example_usage()
