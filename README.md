# ssrlib

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular Python framework for **self-supervised representation analysis**:
plug datasets into embedders, run spectral processors over the resulting
embeddings, and monitor representations during training. Originally built as
a **scikit-learn-inspired pipeline** with automatic component discovery; v0.2
trades the now-deprecated caching layer for an `EmbeddingProbe` that
streams metrics out of training loops.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Framework Architecture](#framework-architecture)
- [Core Components](#core-components)
- [Embedding Probe — monitoring during training](#embedding-probe--monitoring-during-training)
- [Streaming processors](#streaming-processors)
- [Module Discovery System](#module-discovery-system)
- [Adding New Components](#adding-new-components)
- [Usage Examples](#usage-examples)
- [Migration from v0.1](#migration-from-v01)
- [Best Practices](#best-practices)
- [Development](#development)
- [License](#license)
- [Citation](#citation)

---

## Overview

**ssrlib** provides a thin orchestration layer over four reusable component
families:

- **Datasets** — synthetic (`SynthTestDataset`), HuggingFace (`HFVisionDataset`,
  CIFAR/Food101/Caltech101/…), Kaggle (CelebA, ImageNet-100).
- **Embedders** — DINOv2, DINO, CLIP, VICReg (vision); BERT, ModernBERT, E5
  (text); plus a network-free `IdentityEmbedder` for testing.
- **Processors** — covariance, ZCA, spectrum, effective rank, leverage scores,
  pairwise stats; plus the new spectral-quality bundle (NESum, RankMe,
  AlphaReQ, ParticipationRatio, Coherence, ConditionNumber,
  EntropyDecomposition).
- **Losses** — InfoNCE, NT-Xent, Triplet, DeepInfoMax.

### What's new in v0.2

- **`EmbeddingProbe`** — drop-in monitoring for training loops; runs any list
  of processors on encoder outputs and emits a flat metrics dict. Now also
  forwards `labels`, `classifier_weights`, etc. to processors that accept
  them via signature inspection.
- **Spectral-quality processors** — NESum, RankMe, AlphaReQ,
  ParticipationRatio, Coherence, ConditionNumber, EntropyDecomposition.
- **Neural Collapse processor** — `NeuralCollapseProcessor` computes NC1
  (variability collapse), NC2 (Simplex ETF: equinorm + equiangle + max
  equiangle), and optionally NC3 (self-duality) and NC4 (NCC mismatch),
  following Papyan, Han, Donoho 2020.
- **Streaming covariance** via the new `MapReduceMixin`. Set
  `pipeline.execute(streaming=True)` to incrementalise processors that support it.
- **Smaller HF dataset stack.** One `HFVisionDataset` class plus a registry
  replaces the four-file `hf_mixin / hf_vision / cifar10 / food101` layout
  (with backward-compat shims for the old class names).
- **Storage layer removed.** The `TensorStorage` caching subsystem has been
  removed; the pipeline is now ~80 LOC instead of ~250.

See the [migration guide](#migration-from-v01) for breaking changes.

### Key Features

- **Composable pipeline**: any number of (datasets × embedders × processors).
- **Self-describing components**: class-level `_*_category`, `_*_modality`,
  `_*_properties` metadata is auto-discovered.
- **Hookable monitoring** via `EmbeddingProbe` with a `sink` callable
  (`wandb.log`, CSV writer, TensorBoard, …).
- **Streaming support** via opt-in `MapReduceMixin` for processors where
  embeddings don't fit in memory.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip
- CUDA-capable GPU (optional, but recommended for embedding extraction)

### Basic install (from source)

```bash
git clone https://github.com/mmkuznecov/ssrlib.git
cd ssrlib
pip install -e .
```

### With optional dependencies

```bash
# HuggingFace datasets + transformers (needed for CIFAR-10 / Food-101 / BERT / CLIP)
pip install -e ".[hf]"

# Development tools (pytest, coverage, formatters, type-checker)
pip install -e ".[dev]"

# Example notebooks and visualisation
pip install -e ".[examples]"

# Everything
pip install -e ".[all]"
```

### Verify

```python
import ssrlib
from ssrlib.processing import list_processors
from ssrlib.datasets   import list_datasets
from ssrlib.embedders  import list_embedders

print(f"ssrlib  : {ssrlib.__version__}")
print(f"procs   : {len(list_processors())}")
print(f"datasets: {len(list_datasets())}")
print(f"embedders: {len(list_embedders())}")
```

### GPU support

ssrlib auto-detects CUDA. To verify:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

PyTorch with a specific CUDA build:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Quick start (no network required)

```python
from ssrlib import Pipeline
from ssrlib.datasets import SynthTestDataset
from ssrlib.embedders import IdentityEmbedder
from ssrlib.processing import (
    CovarianceProcessor, EffectiveRankProcessor, NESumProcessor,
)

pipeline = Pipeline([
    ("dataset",    SynthTestDataset(tensors_num=64, seed=42)),
    ("embedder",   IdentityEmbedder(output_dim=32, seed=0)),
    ("processors", [CovarianceProcessor(), EffectiveRankProcessor(), NESumProcessor()]),
])

results = pipeline.execute()
print(results.processed[("SynthTest", "Identity", "EffectiveRank")])
```

---

## Framework Architecture

```
ssrlib/
├── core/                    # Pipeline + Config + generic Registry
│   ├── pipeline.py          # ~80-LOC orchestrator (no storage layer)
│   ├── config.py            # dotted-key Config wrapper
│   └── registry.py          # discovery for embedders / datasets / losses
│
├── analysis/                # NEW: monitoring during training
│   └── embedding_probe.py   # EmbeddingProbe + @embedding_probe decorator
│
├── datasets/                # Datasets
│   ├── base.py              # BaseDataset
│   ├── synthtest_dataset.py # Synthetic (no network)
│   ├── hf_registry.py       # HFDatasetInfo + HF_DATASET_REGISTRY
│   ├── hf_vision.py         # ONE class for all HF vision datasets
│   ├── kaggle_mixin.py      # zip-download / extract for Kaggle datasets
│   ├── celeba.py            # CelebA (Kaggle)
│   └── imagenet100.py       # ImageNet-100 (Kaggle)
│
├── embedders/               # Models
│   ├── base.py
│   ├── mock.py              # NEW: IdentityEmbedder (no network)
│   ├── cv/                  # DINOv2, DINO, CLIP, VICReg
│   └── nlp/                 # BERT, ModernBERT, E5
│
├── processing/              # Spectral analysis
│   ├── base.py              # BaseProcessor
│   ├── _spectral.py         # NEW: shared centering / SVD / eigvals
│   ├── map_reduce.py        # NEW: MapReduceMixin for streaming
│   ├── covariance.py        # implements MapReduceMixin
│   ├── zca.py               # bug fix: condition_number guard
│   ├── spectrum.py
│   ├── effective_rank.py
│   ├── stable_rank.py       # default flipped to center=False
│   ├── leverage_scores.py
│   ├── pairwise_stats.py    # comments translated to English
│   ├── spectral_quality.py  # NEW: NESum, RankMe, AlphaReQ, …
│   └── neural_collapse.py   # NEW: NC1, NC2, NC3, NC4 (Papyan et al. 2020)
│
└── losses/                  # SSL losses
    ├── base.py
    ├── infonce_loss.py
    ├── contrastive_loss.py
    ├── triplet_loss.py
    └── deepinfomax_loss.py  # raises if discriminators missing (v0.1 silently mocked)
```

---

## Core Components

### 1. Pipeline

Orchestrates datasets → embedders → processors. Executes the cartesian
product unless explicitly told otherwise.

```python
from ssrlib import Pipeline, Config
from ssrlib.datasets   import SynthTestDataset
from ssrlib.embedders  import IdentityEmbedder
from ssrlib.processing import CovarianceProcessor, NESumProcessor

pipeline = Pipeline([
    ("dataset",    SynthTestDataset(tensors_num=100, seed=42)),
    ("embedder",   IdentityEmbedder(output_dim=32)),
    ("processors", [CovarianceProcessor(), NESumProcessor()]),
])
results = pipeline.execute()
```

### 2. Config

Dotted-key wrapper around dict / YAML / JSON.

```python
from ssrlib import Config

config = Config({"batch_size": 64, "model": {"name": "dinov2_vitb14"}})
config.get("model.name")           # "dinov2_vitb14"
config.set("batch_size", 128)
config = Config.from_file("config.yaml")
```

### 3. PipelineResults

```python
results = pipeline.execute()

emb  = results.get_embeddings("SynthTest", "Identity")
cov  = results.get_processed("SynthTest", "Identity", "Covariance")

results.metadata          # dict: datasets / embedders / processors / config
results.timing            # dict: embedding_time, processing_time, total_time
results.list_dataset_keys()
```

---

## Embedding Probe — monitoring during training

The `EmbeddingProbe` runs any list of processors on encoder outputs and
emits a flat metrics dict. Use it inside a training loop to track how
representations evolve epoch by epoch.

```python
from ssrlib.analysis import EmbeddingProbe
from ssrlib.processing import (
    EffectiveRankProcessor, NESumProcessor, ConditionNumberProcessor,
)

probe = EmbeddingProbe(
    processors=[
        EffectiveRankProcessor(),
        NESumProcessor(),
        ConditionNumberProcessor(),
    ],
    sink=lambda metrics: wandb.log(metrics),  # any callable
    every_n_epochs=5,
)

for epoch in range(epochs):
    train_one_epoch(...)
    if probe.should_run(epoch=epoch):
        emb = encode_validation_set(model)   # numpy or torch tensor
        probe(emb, epoch=epoch)
```

If you want to reuse the processors already configured on a pipeline:

```python
probe = EmbeddingProbe.from_pipeline(pipeline, every_n_epochs=5)
```

For zero-infrastructure use, the `@embedding_probe` decorator wraps a
function that returns embeddings:

```python
from ssrlib.analysis import embedding_probe
from ssrlib.processing import NESumProcessor, ConditionNumberProcessor

@embedding_probe(processors=[NESumProcessor(), ConditionNumberProcessor()])
def encode(model, x):
    return model(x)

emb, metrics = encode(model, x)
```

A complete autoencoder example is in `examples/ae_with_probe.py`. Sample
output (training a small MLP autoencoder on power-law-spectrum data):

```
epoch     loss   erank   nesum       cond#      pr   alpha     r2
-----------------------------------------------------------------
    0   0.0593    1.71    1.12     2638.66    1.25    2.49   0.97
    4   0.0202    4.96    2.39     3964.27    3.75    3.19   0.85
   10   0.0057    6.49    2.70     3255.40    4.80    2.58   0.75
   20   0.0007    6.71    2.82     2984.49    4.91    2.26   0.76
   28   0.0006    6.81    2.84     2844.31    5.00    2.24   0.75
```

Effective rank rises from ~1.7 to ~6.8 as the encoder learns to spread
information across multiple latent axes, while the alpha-ReQ exponent
converges toward the true power-law decay of the data.

---

## Streaming processors

Processors that implement `MapReduceMixin` can be fed batches incrementally.
This is opt-in; processors that don't support it fall through to the
whole-array `process(...)` path automatically.

```python
results = pipeline.execute(streaming=True)
```

`CovarianceProcessor` ships with a streaming implementation that accumulates
`Σ x`, `Σ xxᵀ`, `n`, and finalizes to an unbiased covariance matching
`np.cov` (verified to floating-point precision in the test suite).

You'll only feel the win when embeddings don't fit in memory — for typical
workloads (50k × 1024 floats ≈ 200 MB) the whole-array path is fine.

Adding streaming to a custom processor:

```python
from ssrlib.processing.base import BaseProcessor
from ssrlib.processing.map_reduce import MapReduceMixin

class StreamingMean(BaseProcessor, MapReduceMixin):
    def __init__(self, **kwargs):
        super().__init__("StreamingMean", **kwargs)
        self.reset()

    def process(self, X):
        return X.mean(axis=0)

    def reset(self):
        self._sum = None; self._n = 0

    def partial_fit(self, batch):
        if self._sum is None:
            self._sum = batch.sum(axis=0)
        else:
            self._sum += batch.sum(axis=0)
        self._n += batch.shape[0]

    def finalize(self):
        return self._sum / self._n
```

---

## Neural Collapse metrics

`NeuralCollapseProcessor` computes the four metrics from
**Papyan, Han, Donoho (2020)**, _"Prevalence of Neural Collapse during the
terminal phase of deep learning training"_ (PNAS):

- **NC1**: variability collapse, `Tr(ΣW · ΣB⁺) / C` — within-class variance
  shrinks relative to between-class variance.
- **NC2**: Simplex ETF — three sub-metrics:
  - `nc2_equinorm`: coefficient of variation of `‖μc − μG‖` (→ 0 when all
    class means have equal length)
  - `nc2_equiangle`: standard deviation of pairwise cosines (→ 0 when all
    class-mean angles are equal)
  - `nc2_max_equiangle`: deviation of pairwise cosines from `−1/(C−1)`
    (→ 0 when class means form a Simplex ETF)
- **NC3**: self-duality — `‖Wᵀ/‖W‖_F − Ṁ/‖Ṁ‖_F‖_F²` → 0 when classifier
  weights match centered class means.
- **NC4**: NCC equivalence — fraction of points where the linear classifier
  disagrees with nearest-class-center.

NC1 and NC2 only need embeddings + labels. NC3 and NC4 also require classifier
weights (and optionally bias):

```python
from ssrlib.analysis import EmbeddingProbe
from ssrlib.processing import NeuralCollapseProcessor

probe = EmbeddingProbe(processors=[NeuralCollapseProcessor()])

# In a training-loop validation step:
W = model.head.weight.detach().cpu().numpy()  # (C, D)
b = model.head.bias.detach().cpu().numpy()    # (C,)
metrics = probe(
    features,                      # (N, D) embeddings
    labels=labels,                 # (N,) class indices
    classifier_weights=W,
    classifier_bias=b,
    epoch=epoch,
)
# metrics has 'NeuralCollapse.0' .. 'NeuralCollapse.5' for NC1, NC2*, NC3, NC4
```

The processor's metadata exposes each component by name:

```python
proc = NeuralCollapseProcessor()
proc.process(X, labels=y, classifier_weights=W, classifier_bias=b)
proc.get_metadata()
# {..., 'nc1': 0.18, 'nc2_equinorm': 0.10, 'nc2_equiangle': 0.21,
#       'nc2_max_equiangle': 0.18, 'nc3_selfdual': 0.14, 'nc4_ncc_mismatch': 0.003,
#       'components_order': ['nc1', 'nc2_equinorm', ..., 'nc4_ncc_mismatch']}
```

A complete training example with NC monitoring is in
`examples/classifier_neural_collapse.py`. A typical output table on synthetic
data shows NC1 dropping ~5×, NC4 dropping ~50×, and NC3 dropping ~3× during
the terminal phase.

### How `EmbeddingProbe` routes labels and classifier weights

The probe inspects each processor's `process` signature and forwards only the
kwargs the processor declares. So you can call `probe(emb, labels=y,
classifier_weights=W)` with a list of processors mixing `NESumProcessor`
(takes only embeddings), `EffectiveRankProcessor` (also embeddings only), and
`NeuralCollapseProcessor` (takes labels + classifier kwargs) — the routing
just works:

```python
probe = EmbeddingProbe(processors=[
    NESumProcessor(),
    EffectiveRankProcessor(),
    NeuralCollapseProcessor(),
])
metrics = probe(features, labels=labels, classifier_weights=W, classifier_bias=b)
```

---



ssrlib v0.2 mixes two registration approaches:

- **`processing/`** uses **explicit imports + a manual `_REGISTRY` dict.**
  This gives full IDE / linter / type-checker support for processor classes
  while still exposing `list_processors()` / `create_processor(name, **kw)`
  for runtime lookup. To add a processor, append it to the
  `_PROCESSOR_CLASSES` list in `processing/__init__.py`.

- **`datasets/`, `embedders/`, `losses/`** still use **auto-discovery** via
  `core/registry.py`. The registry walks the package, extracts class-level
  metadata (`_*_category`, `_*_modality`, `_*_properties`), and registers
  every concrete subclass of the base class. Convenient for projects with
  many third-party model wrappers, but each wrapper must declare its
  metadata explicitly.

Discovery API:

```python
from ssrlib.datasets   import list_datasets,   create_dataset,   get_available_datasets
from ssrlib.embedders  import list_embedders,  create_embedder,  get_available_embedders
from ssrlib.processing import list_processors, create_processor, get_available_processors
from ssrlib.losses     import list_losses,     create_loss,      get_available_losses

# Create by string name
proc = create_processor("NESumProcessor", center=True)
ds   = create_dataset("SynthTestDataset", tensors_num=100, seed=42)
```

---

## Adding New Components

### Adding a Dataset

Drop a new module in `ssrlib/datasets/` (e.g. `my_dataset.py`):

```python
from typing import Any, ClassVar, Dict, Iterator
import torch
from .base import BaseDataset


class MyDataset(BaseDataset):
    """My custom dataset."""

    _dataset_category:   ClassVar[str] = "vision"
    _dataset_modality:   ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "image_size": (224, 224), "num_classes": 10,
    }

    def __init__(self, root: str = "data", split: str = "train", **kwargs):
        super().__init__("MyDataset", **kwargs)
        self.root, self.split = root, split

    def download(self):    self._downloaded = True
    def __len__(self):     return self._num_samples
    def __iter__(self) -> Iterator[torch.Tensor]:
        for idx in range(len(self)):
            yield self._load_sample(idx)
```

It's auto-discovered. Then add it to the explicit list in
`datasets/__init__.py` so IDEs see it directly.

### Adding an Embedder

Drop a new module in `ssrlib/embedders/cv/` or `nlp/`:

```python
from typing import Any, ClassVar, Dict
import torch
from ..base import BaseEmbedder


class MyEmbedder(BaseEmbedder):
    """My embedder."""

    _embedder_category:   ClassVar[str] = "vision"
    _embedder_modality:   ClassVar[str] = "vision"
    _embedder_properties: ClassVar[Dict[str, Any]] = {"source": "MyOrg"}

    AVAILABLE_MODELS: ClassVar[Dict[str, int]] = {"my_small": 384, "my_large": 768}

    def __init__(self, model_name="my_small", device="cpu", **kw):
        super().__init__(f"MyEmbedder_{model_name}", device, **kw)
        self.model_name = model_name
        self._embedding_dim = self.AVAILABLE_MODELS[model_name]

    def get_embedding_dim(self): return self._embedding_dim
    def load_model(self):
        if self._loaded: return
        # … load weights …
        self._loaded = True
    def forward(self, batch): return self.model(batch.to(self.device))
```

### Adding a Processor

Drop a new module in `ssrlib/processing/` and register it in
`processing/__init__.py` (one line in `_PROCESSOR_CLASSES`):

```python
import numpy as np
from .base import BaseProcessor

class MyProcessor(BaseProcessor):
    def __init__(self, alpha: float = 1.0, **kw):
        super().__init__("MyProcessor", **kw)
        self.alpha = alpha

    def process(self, X):
        out = X * self.alpha
        self._metadata.update({"input_shape": X.shape})
        return out
```

To support streaming, add `MapReduceMixin` and implement `partial_fit` /
`finalize` / `reset` (see `StreamingMean` example above).

---

## Usage Examples

The `examples/` directory contains four runnable scripts ranging from
network-free demos to real-dataset analyses:

| Script | Network? | Runtime | What it shows |
|---|---|---|---|
| `basic_pipeline.py` | no | <1 s | Smallest possible end-to-end demo |
| `ae_with_probe.py` | no | ~10 s | Train an AE, monitor spectral metrics with `EmbeddingProbe` |
| `classifier_neural_collapse.py` | no | ~30 s | Train a classifier, watch all 6 NC metrics evolve |
| `dinov2_cifar10.py` | yes | ~1-2 min CPU | DINOv2 features on CIFAR-10 + spectral + NC metrics |
| `embedder_comparison.py` | yes | ~2 min CPU | Side-by-side comparison of DINOv2 vs CLIP on CIFAR-10 |

### Basic single pipeline

```python
from ssrlib import Pipeline
from ssrlib.datasets   import SynthTestDataset
from ssrlib.embedders.cv import DINOv2Embedder
from ssrlib.processing import CovarianceProcessor, NESumProcessor

pipeline = Pipeline([
    ("dataset",    SynthTestDataset(tensors_num=50, seed=42)),
    ("embedder",   DINOv2Embedder("vitb14")),
    ("processors", [CovarianceProcessor(), NESumProcessor()]),
])
results = pipeline.execute()

emb  = results.get_embeddings("SynthTest", "DINOv2-vitb14")
nesm = results.get_processed("SynthTest", "DINOv2-vitb14", "NESum")[0]
print(f"emb shape={emb.shape}, NESum={nesm:.3f}")
```

### Multi-component sweep

Computes every (dataset × embedder × processor) combination:

```python
from ssrlib.embedders.cv import DINOv2Embedder, CLIPEmbedder
from ssrlib.processing  import CovarianceProcessor, ZCAProcessor, NESumProcessor

pipeline = Pipeline([
    ("datasets",  [SynthTestDataset(tensors_num=100, seed=1),
                   SynthTestDataset(tensors_num=100, seed=2)]),
    ("embedders", [DINOv2Embedder("vitb14"), CLIPEmbedder()]),
    ("processors",[CovarianceProcessor(), ZCAProcessor(epsilon=1e-6), NESumProcessor()]),
])
results = pipeline.execute()
print(f"{len(results.embeddings)} embedding sets, {len(results.processed)} processed outputs")
```

Duplicate dataset names get unique pipeline keys (`SynthTest`, `SynthTest[1]`).

### Real dataset (HuggingFace)

```python
from ssrlib.datasets   import HFVisionDataset, CIFAR10Dataset
from ssrlib.embedders.cv import DINOv2Embedder
from ssrlib.processing import (
    CovarianceProcessor, EffectiveRankProcessor, AlphaReQProcessor,
)

pipeline = Pipeline([
    # Either form is equivalent:
    ("dataset",    HFVisionDataset(dataset_name="cifar10", split="train")),
    # ("dataset",  CIFAR10Dataset(split="train")),

    ("embedder",   DINOv2Embedder("vits14")),
    ("processors", [CovarianceProcessor(), EffectiveRankProcessor(), AlphaReQProcessor()]),
])
results = pipeline.execute()
```

### Configuration-driven

```python
from ssrlib import Config

config = Config.from_file("config.yaml")
pipeline = Pipeline([...], config=config)
results = pipeline.execute(config_override={"batch_size": 32})
```

### Training with monitoring

The autoencoder example in `examples/ae_with_probe.py` produces this kind of
output (training a small MLP autoencoder on power-law-spectrum data):

```
epoch     loss   erank   nesum       cond#      pr   alpha     r2
-----------------------------------------------------------------
    0   0.0593    1.71    1.12     2638.66    1.25    2.49   0.97
    4   0.0202    4.96    2.39     3964.27    3.75    3.19   0.85
   10   0.0057    6.49    2.70     3255.40    4.80    2.58   0.75
   28   0.0006    6.81    2.84     2844.31    5.00    2.24   0.75
```

Effective rank rises from ~1.7 to ~6.8 as the encoder spreads information
across multiple latent axes; the alpha-ReQ exponent converges toward the
true power-law decay of the data.

### Training a classifier with Neural Collapse monitoring

The `examples/classifier_neural_collapse.py` script trains a small MLP
classifier on synthetic Gaussian-mixture data and prints all six NC metrics
every 20 epochs:

```
epoch     loss  train   test        NC1    NC2eqn    NC2ang    NC2max       NC3    NC4
--------------------------------------------------------------------------------------
    0   1.9681  0.356  0.842     0.8468    0.1903    0.2871    0.2389    0.4143  0.149
   20   0.0002  1.000  0.992     0.1969    0.2954    0.2309    0.1946    0.1690  0.007
   60   0.0002  1.000  0.993     0.1916    0.2812    0.2215    0.1892    0.1571  0.006
  120   0.0003  1.000  0.994     0.1854    0.2647    0.2101    0.1811    0.1437  0.003
  180   0.0003  1.000  0.994     0.1835    0.2600    0.2067    0.1784    0.1396  0.003
```

`NC1` (within/between covariance ratio) drops from 0.85 to 0.18, `NC4`
(linear classifier vs nearest-class-center disagreement) drops from 0.149 to
0.003, and `NC3` (self-duality) drops from 0.41 to 0.14 — exactly the
qualitative trends the original Papyan et al. paper observed on real
classification benchmarks during the terminal phase of training.


---

## Migration from v0.1

### Breaking changes

1. **`TensorStorage` removed.** `Pipeline.execute` no longer accepts
   `use_storage`, `storage_dir`, `force_recompute`, or `storage_description`.

   ```python
   # v0.1
   results = pipeline.execute(use_storage=True, storage_dir="./cache/exp1")
   # v0.2 — caching is gone; orchestrate it externally if needed
   results = pipeline.execute()
   ```

2. **`StableRankProcessor` default flipped** from `center=True` to
   `center=False`. The new default matches the textbook definition
   `‖X‖_F² / ‖X‖₂²`. To keep v0.1 behaviour:

   ```python
   StableRankProcessor(center=True)
   ```

3. **HF dataset module layout.** `hf_mixin.py`, the standalone `cifar10.py`,
   and `food101.py` no longer exist. `CIFAR10Dataset` and `Food101Dataset`
   remain importable as one-line shims around `HFVisionDataset`.

4. **`DeepInfoMaxLoss`** raises `ValueError` at construction when
   `global_discriminator` / `local_discriminator` / `prior_discriminator`
   aren't provided. v0.1 silently substituted random-output mocks.

5. **Processor outputs are now uniformly shaped.** Scalars are shape `(1,)`,
   vectors are shape `(k,)`, matrices are `(D, D)`. If you wrote code that
   handled scalar processors as Python floats, change to `result[0]`.

### Non-breaking improvements

- `np.cov` ddof=1 convention is now used consistently across all spectral
  computations (matches `np.cov` default).
- `ZCAProcessor` no longer divides by zero on rank-deficient inputs.
- `pairwise_stats.py` no longer has Russian comments.
- `pyproject.toml` replaces `setup.py`.

---

## Best Practices

### 1. Define class-level metadata

```python
class MyDataset(BaseDataset):
    _dataset_category:   ClassVar[str] = "vision"
    _dataset_modality:   ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {...}
```

### 2. Yield tensors only from `__iter__`

Datasets in pipeline contexts yield single tensors of identical shape
(images only, no labels). For labelled datasets, use `__getitem__` for
training and `__iter__` for embedding extraction.

### 3. Use `EmbeddingProbe` for monitoring; pipelines for evaluation

The pipeline assumes batch-level analysis on a fixed embedding set; the
probe is designed to be fast enough to run inside training loops at every
Nth epoch.

### 4. Stream only when you have to

`MapReduceMixin` only earns its keep when embeddings don't fit in memory.
For 50k × 1024 floats (~200 MB), the whole-array path is faster.

### 5. Pin processor outputs in tests

The processors are deterministic given seeded inputs; in CI, pin the
expected output for a fixed-seed `SynthTestDataset` to detect regressions.

---

## Development

### Running tests

```bash
pytest -q                    # ~5s, network-free
pytest -v --cov=ssrlib       # with coverage
```

The full suite (113 tests) runs without network using `SynthTestDataset`
and `IdentityEmbedder`.

### Code quality

```bash
black ssrlib tests
isort ssrlib tests
pylint ssrlib
mypy ssrlib
```

### Project layout

```
.
├── ssrlib/                  # the library (see Architecture above)
├── tests/                   # 113 pytest tests, all network-free
├── examples/
│   ├── ae_with_probe.py                 # AE training + spectral monitoring (no network)
│   ├── classifier_neural_collapse.py    # classifier + 6 NC metrics (no network)
│   ├── dinov2_cifar10.py                # DINOv2 features on CIFAR-10 + spectral + NC
│   └── embedder_comparison.py           # DINOv2 vs CLIP on CIFAR-10
├── basic_pipeline.py        # smallest end-to-end demo
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## License

MIT License — see LICENSE file for details.

## Citation

If you use ssrlib in your research, please cite:

```bibtex
@software{ssrlib2026,
  author = {Mikhail Kuznetov},
  title = {ssrlib: A Modular Framework for Self-Supervised Representation Analysis},
  year = {2026},
  version = {0.2.0},
  url = {https://github.com/mmkuznecov/ssrlib}
}
```
