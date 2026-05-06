"""Deep InfoMax loss.

Reference: Hjelm et al. 2019.

Note:
    Earlier versions of ssrlib provided "mock" discriminators that returned
    random tensors when real models weren't available. That silently produced
    garbage gradients. This version raises ``ValueError`` at construction
    when the discriminators are missing, surfacing the misconfiguration
    immediately.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss


class DeepInfoMaxLoss(BaseLoss):
    """Deep InfoMax (DIM) loss combining global, local, and prior terms.

    Args:
        global_discriminator: nn.Module mapping (M, encoded) -> scores.
        local_discriminator: nn.Module mapping (M, encoded) -> per-spatial scores.
        prior_discriminator: nn.Module mapping (encoded,) -> {real/fake} score.
        alpha: weight for the global MI term.
        beta: weight for the local MI term.
        gamma: weight for the prior matching term.
    """

    _loss_category: ClassVar[str] = "mutual_information"
    _loss_modality: ClassVar[str] = "vision"
    _loss_properties: ClassVar[Dict[str, Any]] = {
        "requires_discriminators": True,
    }

    def __init__(
        self,
        global_discriminator: Optional[nn.Module] = None,
        local_discriminator: Optional[nn.Module] = None,
        prior_discriminator: Optional[nn.Module] = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        gamma: float = 0.1,
        **kwargs,
    ):
        super().__init__("DeepInfoMaxLoss", **kwargs)

        missing = [
            name
            for name, mod in [
                ("global_discriminator", global_discriminator),
                ("local_discriminator", local_discriminator),
                ("prior_discriminator", prior_discriminator),
            ]
            if mod is None
        ]
        if missing:
            raise ValueError(
                f"DeepInfoMaxLoss requires the following discriminator modules "
                f"to be supplied at construction: {', '.join(missing)}. "
                "Earlier versions silently substituted mock discriminators "
                "that returned random tensors; this is no longer permitted."
            )

        self.global_d = global_discriminator
        self.local_d = local_discriminator
        self.prior_d = prior_discriminator
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self._metadata.update(
            {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}
        )

    def forward(self, M: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        """Compute the DIM loss.

        Args:
            M: feature map of shape (B, C, H, W).
            encoded: global feature vector of shape (B, F).
        """
        # Shuffle the batch to provide a "fake" pair
        perm = torch.randperm(M.size(0), device=M.device)
        M_prime = M[perm]

        # Global MI
        Ej_g = self.global_d(M, encoded)
        Em_g = self.global_d(M_prime, encoded)
        global_loss = -(F.softplus(-Ej_g).mean() + F.softplus(Em_g).mean())

        # Local MI
        Ej_l = self.local_d(M, encoded)
        Em_l = self.local_d(M_prime, encoded)
        local_loss = -(F.softplus(-Ej_l).mean() + F.softplus(Em_l).mean())

        # Prior matching
        prior = torch.rand_like(encoded)
        prior_loss = (
            torch.log(self.prior_d(prior) + 1e-8).mean()
            + torch.log(1.0 - self.prior_d(encoded) + 1e-8).mean()
        )

        return (
            self.alpha * global_loss + self.beta * local_loss + self.gamma * prior_loss
        )
