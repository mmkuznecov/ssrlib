"""Train a small autoencoder while monitoring its latent representations with ssrlib.

Demonstrates the ``EmbeddingProbe``: at the end of every Nth epoch the script
encodes a held-out validation set and runs a list of spectral processors on
the latent matrix. Metrics are streamed to a ``sink`` callable (here a plain
list, but in practice this would be ``wandb.log``, a CSV writer, a
TensorBoard logger, etc.).

What you should see when training runs cleanly:

    - reconstruction loss decreases monotonically
    - effective_rank rises, then stabilizes (the encoder learns to spread its
      output across multiple useful directions)
    - nesum tracks closely with effective_rank (both measure spectrum flatness
      in different ways)
    - condition_number rises early (encoder differentiates features) then
      either plateaus or drifts up if the spectrum becomes too peaked
    - alpha_req captures the power-law decay of the latent spectrum

Run:
    python examples/ae_with_probe.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ssrlib.analysis import EmbeddingProbe
from ssrlib.processing import (
    AlphaReQProcessor,
    ConditionNumberProcessor,
    EffectiveRankProcessor,
    NESumProcessor,
    ParticipationRatioProcessor,
)


# ----------------------------------------------------------------- 1. Data
def make_powerlaw_data(
    n: int,
    d: int,
    signal_dim: int = 12,
    alpha: float = 0.8,
    noise: float = 0.02,
    seed: int = 0,
) -> torch.Tensor:
    """Continuous data with a power-law covariance spectrum.

    The first ``signal_dim`` axes (in a random orthonormal basis) carry signal
    with eigenvalues ``1 / i^alpha``; the remaining axes carry small white
    noise. Designed to give an autoencoder something gradually-decaying to
    learn so that the spectral metrics keep evolving across epochs.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigvals = np.full(d, noise**2, dtype=np.float64)
    eigvals[:signal_dim] = 1.0 / np.arange(1, signal_dim + 1) ** alpha
    L = Q @ np.diag(np.sqrt(eigvals))
    Z = rng.standard_normal((n, d))
    return torch.from_numpy((Z @ L.T).astype(np.float32))


# ---------------------------------------------------------- 2. Autoencoder
class Autoencoder(nn.Module):
    """Tiny MLP autoencoder; encoder output is what we monitor."""

    def __init__(self, in_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decoder(z), z


# ------------------------------------------------------------- 3. Training
def train_ae(
    epochs: int = 30,
    batch_size: int = 128,
    in_dim: int = 64,
    latent_dim: int = 16,
    every_n_epochs: int = 2,
    csv_out: str = "ae_metrics.csv",
) -> List[dict]:
    """Train the AE and return the list of probed metrics dicts."""
    logging.basicConfig(level=logging.WARNING)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    X_train = make_powerlaw_data(2048, in_dim, seed=0)
    X_val = make_powerlaw_data(512, in_dim, seed=1).to(device)
    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=batch_size, shuffle=True
    )

    # Model
    model = Autoencoder(in_dim, latent_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- Probe ----
    # The sink is any callable that takes a flat metrics dict. Pluggable:
    # `wandb.log`, a CSV writer, a TensorBoard scalar dispatcher, etc.
    metrics_log: List[dict] = []
    probe = EmbeddingProbe(
        processors=[
            EffectiveRankProcessor(),
            NESumProcessor(),
            ConditionNumberProcessor(),
            ParticipationRatioProcessor(),
            AlphaReQProcessor(min_eigvals=4),
        ],
        sink=metrics_log.append,
        every_n_epochs=every_n_epochs,
    )

    # ---- Training loop ----
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for (x,) in train_loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
        recon = running / max(n_seen, 1)

        if probe.should_run(epoch=epoch):
            model.eval()
            with torch.no_grad():
                z = model.encode(X_val)
            metrics = probe(z, epoch=epoch)
            # Decorate with reconstruction loss for the table
            metrics["recon_loss"] = recon

    # ---- Report ----
    print(f"\nFinal reconstruction loss: {recon:.4f}\n")
    header = (
        f"{'epoch':>5}  {'loss':>7}  {'erank':>6}  {'nesum':>6}  "
        f"{'cond#':>10}  {'pr':>6}  {'alpha':>6}  {'r2':>5}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics_log:
        print(
            f"{m['epoch']:>5}  "
            f"{m['recon_loss']:>7.4f}  "
            f"{m['EffectiveRank']:>6.2f}  "
            f"{m['NESum']:>6.2f}  "
            f"{m['ConditionNumber']:>10.2f}  "
            f"{m['ParticipationRatio']:>6.2f}  "
            f"{m['AlphaReQ.0']:>6.2f}  "
            f"{m['AlphaReQ.1']:>5.2f}"
        )

    # Save to CSV for downstream plotting
    cols = [
        "epoch",
        "recon_loss",
        "EffectiveRank",
        "NESum",
        "ConditionNumber",
        "ParticipationRatio",
        "AlphaReQ.0",
        "AlphaReQ.1",
    ]
    csv_path = Path(csv_out)
    rows = [",".join(cols)]
    for m in metrics_log:
        rows.append(
            ",".join(
                f"{m[c]:.6f}" if isinstance(m[c], float) else str(m[c]) for c in cols
            )
        )
    csv_path.write_text("\n".join(rows) + "\n")
    print(f"\nMetrics saved to {csv_path.resolve()}")

    return metrics_log


if __name__ == "__main__":
    train_ae()
