"""Train a classifier and watch Neural Collapse emerge during the terminal phase.

Reproduces the qualitative experiment of Papyan, Han, Donoho (2020) — but with
a tiny model on synthetic data so it runs in ~30 seconds on CPU. The same code
works on real datasets: swap ``make_classification_data`` for any DataLoader
that yields ``(image_tensor, label)`` pairs.

What you should see in the printed table as training progresses past zero
training error:

    train_acc  → 1.0           # zero-error reached
    nc1        → 0             # within-class variance vanishes
    nc2_eqnorm → 0             # class means become equinorm
    nc2_ang    → 0             # class means become equiangular
    nc2_max    → 0             # angles converge to −1/(C−1) ↦ Simplex ETF
    nc3        → 0             # classifier weights match class means
    nc4        → 0             # linear classifier == nearest class-center

Run:
    python examples/classifier_neural_collapse.py
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
from ssrlib.processing import NeuralCollapseProcessor


# ---------------------------------------------------------------- 1. Data
def make_classification_data(
    n_per_class: int = 200,
    n_classes: int = 8,
    in_dim: int = 64,
    separation: float = 2.5,
    noise: float = 1.5,
    centers_seed: int = 0,
    noise_seed: int = 0,
):
    """Noisy Gaussian-mixture classification data.

    ``separation`` controls inter-class distance; ``noise`` controls
    within-class spread. With ``separation ≈ noise`` clusters overlap
    significantly, so the classifier must do real work — and Neural Collapse
    has a chance to develop visibly across epochs rather than instantly.

    Train and validation sets must share ``centers_seed`` so they sample from
    the same class distribution; only ``noise_seed`` should differ.
    """
    centers_rng = np.random.default_rng(centers_seed)
    noise_rng = np.random.default_rng(noise_seed)

    centers = centers_rng.standard_normal((n_classes, in_dim)) * separation
    X_blocks, y_blocks = [], []
    for c in range(n_classes):
        X_blocks.append(
            centers[c] + noise_rng.standard_normal((n_per_class, in_dim)) * noise
        )
        y_blocks.append(np.full(n_per_class, c))
    X = np.concatenate(X_blocks).astype(np.float32)
    y = np.concatenate(y_blocks).astype(np.int64)
    perm = noise_rng.permutation(len(X))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


# ----------------------------------------------------------- 2. Classifier
class Classifier(nn.Module):
    """Small MLP: input → encoder features → linear classifier head."""

    def __init__(self, in_dim: int = 64, feature_dim: int = 32, n_classes: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
        self.head = nn.Linear(feature_dim, n_classes, bias=True)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        h = self.features(x)
        return self.head(h), h


# ------------------------------------------------------------- 3. Training
def train(
    epochs: int = 200,
    batch_size: int = 256,
    n_classes: int = 8,
    every_n_epochs: int = 20,
    csv_out: str = "classifier_nc_metrics.csv",
) -> List[dict]:
    logging.basicConfig(level=logging.WARNING)
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data — train and val share centers so the classifier can actually generalise.
    # noise ≥ separation forces real overlap; the network has to do work to
    # disentangle clusters and we get a wider initial-vs-final NC range.
    X_train, y_train = make_classification_data(
        n_classes=n_classes,
        separation=2.0,
        noise=3.0,
        centers_seed=0,
        noise_seed=0,
    )
    X_val, y_val = make_classification_data(
        n_classes=n_classes,
        separation=2.0,
        noise=3.0,
        centers_seed=0,
        noise_seed=42,
    )
    X_val_dev, y_val_dev = X_val.to(device), y_val.to(device)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    # Model
    model = Classifier(in_dim=64, feature_dim=32, n_classes=n_classes).to(device)
    optim = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    # ---- Probe with NC processor (full 6 components: NC1 + 3×NC2 + NC3 + NC4)
    metrics_log: List[dict] = []
    probe = EmbeddingProbe(
        processors=[NeuralCollapseProcessor()],
        sink=metrics_log.append,
        every_n_epochs=every_n_epochs,
    )

    # ---- Training loop ----
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        n_seen = 0
        for x, lbl in train_loader:
            x, lbl = x.to(device), lbl.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, lbl)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == lbl).sum().item()
            n_seen += x.size(0)
        sched.step()
        train_loss /= n_seen
        train_acc = train_correct / n_seen

        if probe.should_run(epoch=epoch):
            model.eval()
            with torch.no_grad():
                logits_val, h_val = model(X_val_dev)
                test_acc = (logits_val.argmax(1) == y_val_dev).float().mean().item()

            W = model.head.weight.detach().cpu().numpy()  # shape (C, D)
            b = model.head.bias.detach().cpu().numpy()  # shape (C,)

            metrics = probe(
                h_val.cpu().numpy(),
                labels=y_val_dev.cpu().numpy(),
                classifier_weights=W,
                classifier_bias=b,
                epoch=epoch,
            )
            metrics["train_loss"] = train_loss
            metrics["train_acc"] = train_acc
            metrics["test_acc"] = test_acc

    # ---- Report ----
    print(
        f"\nFinal train acc: {train_acc:.3f}, "
        f"final test acc: {metrics_log[-1]['test_acc']:.3f}\n"
    )
    header = (
        f"{'epoch':>5}  {'loss':>7}  {'train':>5}  {'test':>5}  "
        f"{'NC1':>9}  {'NC2eqn':>8}  {'NC2ang':>8}  {'NC2max':>8}  "
        f"{'NC3':>8}  {'NC4':>5}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics_log:
        print(
            f"{m['epoch']:>5}  "
            f"{m['train_loss']:>7.4f}  "
            f"{m['train_acc']:>5.3f}  "
            f"{m['test_acc']:>5.3f}  "
            f"{m['NeuralCollapse.0']:>9.4f}  "  # NC1
            f"{m['NeuralCollapse.1']:>8.4f}  "  # NC2_equinorm
            f"{m['NeuralCollapse.2']:>8.4f}  "  # NC2_equiangle
            f"{m['NeuralCollapse.3']:>8.4f}  "  # NC2_max_equiangle
            f"{m['NeuralCollapse.4']:>8.4f}  "  # NC3
            f"{m['NeuralCollapse.5']:>5.3f}"  # NC4
        )

    # CSV for downstream plotting
    cols = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_acc",
        "NeuralCollapse.0",
        "NeuralCollapse.1",
        "NeuralCollapse.2",
        "NeuralCollapse.3",
        "NeuralCollapse.4",
        "NeuralCollapse.5",
    ]
    pretty_cols = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_acc",
        "nc1",
        "nc2_equinorm",
        "nc2_equiangle",
        "nc2_max_equiangle",
        "nc3_selfdual",
        "nc4_ncc_mismatch",
    ]
    csv_path = Path(csv_out)
    rows = [",".join(pretty_cols)]
    for m in metrics_log:
        rows.append(
            ",".join(
                f"{m[c]:.6f}" if isinstance(m[c], float) else str(m[c]) for c in cols
            )
        )
    csv_path.write_text("\n".join(rows) + "\n")
    print(f"\nNC metrics saved to {csv_path.resolve()}")
    return metrics_log


if __name__ == "__main__":
    train()
