"""Minimal end-to-end example for ssrlib.

Uses the network-free synthetic dataset + IdentityEmbedder so this script
runs without downloading any models or data. For a real example replace
``IdentityEmbedder`` with e.g. ``DINOv2Embedder()``.
"""

from __future__ import annotations

import logging

from ssrlib import EmbeddingProbe, Pipeline
from ssrlib.datasets import SynthTestDataset
from ssrlib.embedders import IdentityEmbedder
from ssrlib.processing import (
    ConditionNumberProcessor,
    CovarianceProcessor,
    EffectiveRankProcessor,
    NESumProcessor,
    SpectrumProcessor,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    pipeline = Pipeline(
        [
            ("dataset", SynthTestDataset(tensors_num=64, seed=42)),
            ("embedder", IdentityEmbedder(output_dim=32, seed=0)),
            (
                "processors",
                [
                    CovarianceProcessor(),
                    SpectrumProcessor(normalize=True),
                    EffectiveRankProcessor(),
                    NESumProcessor(),
                    ConditionNumberProcessor(),
                ],
            ),
        ]
    )

    results = pipeline.execute()

    cov = results.processed[("SynthTest", "Identity", "Covariance")]
    nesum = results.processed[("SynthTest", "Identity", "NESum")][0]
    erank = results.processed[("SynthTest", "Identity", "EffectiveRank")][0]
    cond = results.processed[("SynthTest", "Identity", "ConditionNumber")][0]

    print(f"Covariance: shape={cov.shape}, ||C||_F = {(cov * cov).sum() ** 0.5:.3f}")
    print(f"NESum:           {nesum:.3f}")
    print(f"EffectiveRank:   {erank:.3f}")
    print(f"ConditionNumber: {cond:.3f}")

    # The same processors as a probe — useful inside training loops.
    probe = EmbeddingProbe.from_pipeline(pipeline)
    emb = list(results.embeddings.values())[0]
    print("\nProbe metrics:")
    for k, v in sorted(probe(emb, epoch=0).items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
