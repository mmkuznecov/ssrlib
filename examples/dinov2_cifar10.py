"""Analyse DINOv2 features on CIFAR-10 with spectral + Neural Collapse metrics.

Demonstrates how to combine ssrlib's pipeline (for embedding extraction) with
the EmbeddingProbe (for label-aware metrics like Neural Collapse) on a real
dataset.

Run:
    pip install -e ".[hf]"          # installs `datasets` and `transformers`
    python examples/dinov2_cifar10.py

Expected output (qualitative — exact numbers will vary):

    Foundation model features (no fine-tuning) typically show:
        - Effective rank: 100-300 out of 384 dims (DINOv2-small)
        - NC1 (within/between cov): ~5-20  (much higher than a trained classifier)
        - NC2 metrics: > 0  (no Simplex ETF emerges without classification training)
        - Alpha-ReQ exponent: ~0.7-1.2 for natural-image features
"""

from __future__ import annotations

import argparse
import logging
from typing import Tuple

import numpy as np
import torch

from ssrlib.analysis import EmbeddingProbe
from ssrlib.datasets import HFVisionDataset
from ssrlib.embedders.cv import DINOv2Embedder
from ssrlib.processing import (
    AlphaReQProcessor,
    ConditionNumberProcessor,
    EffectiveRankProcessor,
    NESumProcessor,
    NeuralCollapseProcessor,
    ParticipationRatioProcessor,
)


def extract_features_and_labels(
    dataset: HFVisionDataset,
    embedder: DINOv2Embedder,
    n_samples: int,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings + labels for the first ``n_samples`` items of a dataset.

    Uses the labelled ``__getitem__`` path (yields ``(image, label)``), unlike
    the pipeline's iterator which yields image-only tensors. NC metrics need
    labels, so we go through this path explicitly.
    """
    embedder.load_model()
    n_samples = min(n_samples, len(dataset))

    feats: list[np.ndarray] = []
    lbls: list[int] = []
    img_buf: list[torch.Tensor] = []
    lbl_buf: list[int] = []

    for idx in range(n_samples):
        img, lbl = dataset[idx]
        img_buf.append(img)
        lbl_buf.append(int(lbl))
        if len(img_buf) >= batch_size:
            with torch.no_grad():
                emb = embedder.forward(torch.stack(img_buf))
            feats.append(emb.detach().cpu().numpy())
            lbls.extend(lbl_buf)
            img_buf, lbl_buf = [], []

    if img_buf:
        with torch.no_grad():
            emb = embedder.forward(torch.stack(img_buf))
        feats.append(emb.detach().cpu().numpy())
        lbls.extend(lbl_buf)

    return (
        np.concatenate(feats, axis=0).astype(np.float64),
        np.asarray(lbls, dtype=np.int64),
    )


def main(n_samples: int = 2000, model_size: str = "vits14") -> dict:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    print(f"Loading CIFAR-10 (test split) and DINOv2-{model_size} ...")
    dataset = HFVisionDataset(dataset_name="cifar10", split="test")
    embedder = DINOv2Embedder(model_size=model_size, device="cpu")

    print(f"Extracting features for {min(n_samples, len(dataset))} images ...")
    X, y = extract_features_and_labels(dataset, embedder, n_samples=n_samples)
    print(f"  X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")

    probe = EmbeddingProbe(
        processors=[
            EffectiveRankProcessor(),
            NESumProcessor(),
            ParticipationRatioProcessor(),
            ConditionNumberProcessor(),
            AlphaReQProcessor(),
            NeuralCollapseProcessor(),
        ]
    )
    metrics = probe(X, labels=y)

    print()
    print("---- Spectral metrics ----")
    print(f"  Effective Rank:        {metrics['EffectiveRank']:>9.2f}  / {X.shape[1]}")
    print(f"  NESum:                 {metrics['NESum']:>9.2f}")
    print(f"  Participation Ratio:   {metrics['ParticipationRatio']:>9.2f}")
    print(f"  Condition Number:      {metrics['ConditionNumber']:>9.2e}")
    print(
        f"  AlphaReQ:  α={metrics['AlphaReQ.0']:.2f},  R²={metrics['AlphaReQ.1']:.2f}"
    )

    print()
    print("---- Neural Collapse metrics (no classifier head) ----")
    print(f"  NC1 within/between cov:          {metrics['NeuralCollapse.0']:>8.4f}")
    print(f"  NC2 equinorm CV:                 {metrics['NeuralCollapse.1']:>8.4f}")
    print(f"  NC2 equiangle std:               {metrics['NeuralCollapse.2']:>8.4f}")
    print(f"  NC2 max-equiangle deviation:     {metrics['NeuralCollapse.3']:>8.4f}")
    print()
    print(
        "Note: Foundation models like DINOv2 are trained without labels, so they "
        "don't display strict Neural Collapse — but their features are often "
        "well-organised enough that NC metrics already start small. Comparing "
        "these numbers across pretrained models gives a sense of how class-"
        "discriminative each model's representation is on CIFAR-10."
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of CIFAR-10 images to embed (default 2000).",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="vits14",
        choices=["vits14", "vitb14", "vitl14"],
        help="DINOv2 model size (default vits14, smallest).",
    )
    args = parser.parse_args()
    main(n_samples=args.n_samples, model_size=args.model_size)
