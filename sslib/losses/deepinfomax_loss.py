import torch
import torch.nn as nn
import torch.nn.functional as F


# Mock discriminator classes for testing
class MockGlobalDiscriminator(nn.Module):
    """Mock global discriminator for testing purposes"""
    def __init__(self, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(1, output_dim)  # Minimal implementation
        
    def forward(self, y, M):
        # Create a mock output with the right batch size
        batch_size = y.shape[0]
        # Return random values for testing
        return torch.randn(batch_size, 1, device=y.device)


class MockLocalDiscriminator(nn.Module):
    """Mock local discriminator for testing purposes"""
    def __init__(self, output_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(1, output_channels, kernel_size=1)  # Minimal implementation
        
    def forward(self, x):
        # Return tensor with same spatial dimensions but different channels
        batch_size, channels, height, width = x.shape
        return torch.randn(batch_size, 1, height, width, device=x.device)


class MockPriorDiscriminator(nn.Module):
    """Mock prior discriminator for testing purposes"""
    def __init__(self, input_dim=64, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # Return values between 0 and 1 (since original uses sigmoid)
        batch_size = x.shape[0]
        return torch.sigmoid(torch.randn(batch_size, 1, device=x.device))


class DeepInfoMaxLoss(nn.Module):
    """
    Parametrizable DeepInfoMaxLoss that accepts discriminators as arguments.
    
    Args:
        global_discriminator: Global discriminator instance (default: GlobalDiscriminator())
        local_discriminator: Local discriminator instance (default: LocalDiscriminator())
        prior_discriminator: Prior discriminator instance (default: PriorDiscriminator())
        alpha: Weight for global loss term (default: 0.5)
        beta: Weight for local loss term (default: 1.0)
        gamma: Weight for prior loss term (default: 0.1)
    """
    
    def __init__(self, 
                 global_discriminator=None, 
                 local_discriminator=None, 
                 prior_discriminator=None,
                 alpha=0.5, 
                 beta=1.0, 
                 gamma=0.1):
        super().__init__()
        
        # Use provided discriminators or create default ones
        if global_discriminator is not None:
            self.global_d = global_discriminator
        else:
            # Import here to avoid circular imports if this is in a separate file
            from models import GlobalDiscriminator
            self.global_d = GlobalDiscriminator()
            
        if local_discriminator is not None:
            self.local_d = local_discriminator
        else:
            from models import LocalDiscriminator
            self.local_d = LocalDiscriminator()
            
        if prior_discriminator is not None:
            self.prior_d = prior_discriminator
        else:
            from models import PriorDiscriminator
            self.prior_d = PriorDiscriminator()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):
        """
        Forward pass of DeepInfoMaxLoss
        
        Args:
            y: Encoded representations [batch_size, encoding_dim]
            M: Feature maps [batch_size, channels, height, width]
            M_prime: Shuffled/rotated feature maps [batch_size, channels, height, width]
            
        Returns:
            Combined loss (LOCAL + GLOBAL + PRIOR)
        """
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 26, 26)

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        # Local discriminator loss
        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        # Global discriminator loss
        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        # Prior discriminator loss
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR


# Example usage demonstrations
def example_usage():
    """Examples of how to use the parametrizable DeepInfoMaxLoss"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example 1: Using default discriminators (same as original behavior)
    print("Example 1: Default discriminators")
    try:
        loss_fn_default = DeepInfoMaxLoss()
        print("✓ Created loss function with default discriminators")
    except ImportError:
        print("⚠ Cannot import original discriminators, using mocks instead")
        loss_fn_default = DeepInfoMaxLoss(
            global_discriminator=MockGlobalDiscriminator(),
            local_discriminator=MockLocalDiscriminator(),
            prior_discriminator=MockPriorDiscriminator()
        )
    
    # Example 2: Using mock discriminators for testing
    print("\nExample 2: Mock discriminators")
    loss_fn_mock = DeepInfoMaxLoss(
        global_discriminator=MockGlobalDiscriminator(),
        local_discriminator=MockLocalDiscriminator(),
        prior_discriminator=MockPriorDiscriminator(),
        alpha=0.3,  # Custom hyperparameters
        beta=1.2,
        gamma=0.15
    )
    print("✓ Created loss function with mock discriminators")
    
    # Example 3: Mixed approach - some custom, some default
    print("\nExample 3: Mixed discriminators")
    loss_fn_mixed = DeepInfoMaxLoss(
        global_discriminator=MockGlobalDiscriminator(),  # Custom
        local_discriminator=None,  # Will use default
        prior_discriminator=MockPriorDiscriminator()  # Custom
    )
    print("✓ Created loss function with mixed discriminators")
    
    # Test with dummy data
    print("\nTesting with dummy data:")
    batch_size = 4
    encoding_dim = 64
    feature_channels = 128
    feature_size = 26
    
    y = torch.randn(batch_size, encoding_dim)
    M = torch.randn(batch_size, feature_channels, feature_size, feature_size)
    M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)  # Rotate as in original
    
    try:
        loss = loss_fn_mock(y, M, M_prime)
        print(f"✓ Loss computed successfully: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Error computing loss: {e}")


if __name__ == "__main__":
    example_usage()