"""Compare DINOv2 and CLIP feature quality on CIFAR-10.

Uses the ssrlib Pipeline to extract embeddings for the cartesian product of
{datasets} × {embedders} × {processors}, then prints a side-by-side
comparison table of spectral metrics.

This script downloads model weights and the CIFAR-10 test split on first run.
With ``--n_samples 1000`` it takes ~2 minutes on CPU and ~10 seconds on a
single GPU.

Run:
    pip install -e ".[hf]"
    python examples/embedder_comparison.py --n_samples 1000

Sample output (numbers will vary):

    Dataset        Embedder          ERank    NESum    PR    Alpha    Cond#
    CIFAR10[1000]  DINOv2-vits14     142.31   23.4     85    0.92     1.2e+04
    CIFAR10[1000]  CLIP-base-p32      94.50   18.1     56    1.14     8.7e+03
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict

from ssrlib import Pipeline
from ssrlib.datasets import HFVisionDataset
from ssrlib.embedders.cv import CLIPEmbedder, DINOv2Embedder
from ssrlib.processing import (
    AlphaReQProcessor,
    ConditionNumberProcessor,
    EffectiveRankProcessor,
    NESumProcessor,
    ParticipationRatioProcessor,
)


class _SubsetWrapper:
    """Wrap an HF dataset to expose only the first ``n_samples`` items.

    Used so the comparison script can run quickly without downloading or
    embedding the entire 10k-image CIFAR-10 test set.
    """

    def __init__(self, base, n_samples: int):
        self._base = base
        self.n = min(n_samples, len(base))
        # Mimic just enough of BaseDataset's interface for the pipeline:
        self.name = f"{base.name}[{self.n}]"
        self._downloaded = True
        self._metadata = dict(base._metadata)
        self._metadata["subset_size"] = self.n
        self._dataset_category = getattr(base, "_dataset_category", "vision")
        self._dataset_modality = getattr(base, "_dataset_modality", "vision")

    def download(self):
        self._base.download()

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            img, _ = self._base[i]
            yield img

    def get_metadata(self) -> Dict[str, Any]:
        meta = self._base.get_metadata().copy()
        meta.update({"subset_size": self.n, "name": self.name})
        return meta


def main(n_samples: int = 1000):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    base = HFVisionDataset(dataset_name="cifar10", split="test")
    dataset = _SubsetWrapper(base, n_samples=n_samples)

    pipeline = Pipeline(
        [
            ("dataset", dataset),
            (
                "embedders",
                [
                    DINOv2Embedder(model_size="vits14", device="cpu"),
                    CLIPEmbedder(
                        model_name="openai/clip-vit-base-patch32", device="cpu"
                    ),
                ],
            ),
            (
                "processors",
                [
                    EffectiveRankProcessor(),
                    NESumProcessor(),
                    ParticipationRatioProcessor(),
                    AlphaReQProcessor(),
                    ConditionNumberProcessor(),
                ],
            ),
        ]
    )

    results = pipeline.execute()

    # Pretty-print results
    embedder_names = [e.name for e in pipeline.embedders]
    print()
    print(
        f"{'Embedder':>22}  {'ERank':>7}  {'NESum':>7}  "
        f"{'PR':>7}  {'α':>5}  {'R²':>5}  {'Cond#':>9}"
    )
    print("-" * 75)
    for emb_name in embedder_names:
        key = (dataset.name, emb_name)
        erank = float(results.processed[(*key, "EffectiveRank")][0])
        nesum = float(results.processed[(*key, "NESum")][0])
        pr = float(results.processed[(*key, "ParticipationRatio")][0])
        alpha, r2 = results.processed[(*key, "AlphaReQ")]
        cond = float(results.processed[(*key, "ConditionNumber")][0])
        print(
            f"{emb_name:>22}  {erank:>7.2f}  {nesum:>7.2f}  "
            f"{pr:>7.2f}  {alpha:>5.2f}  {r2:>5.2f}  {cond:>9.2e}"
        )

    print()
    print("Reading the table:")
    print("  ERank — effective rank (Roy & Vetterli 2007); higher = more spread")
    print("  NESum — Σ λi / λ_max (He & Ozay 2022); higher = flatter spectrum")
    print("  PR    — participation ratio = (Σλ)² / Σλ²; higher = more active dims")
    print("  α     — power-law decay exponent of the spectrum (Stringer et al.)")
    print("  Cond# — λ_max / λ_min; high values flag near-degenerate directions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of CIFAR-10 test images to embed.",
    )
    args = parser.parse_args()
    main(n_samples=args.n_samples)
