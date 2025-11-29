# SSLib Framework Documentation

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Framework Architecture](#framework-architecture)
- [Core Components](#core-components)
- [Module Discovery System](#module-discovery-system)
- [Adding New Components](#adding-new-components)
- [Usage Examples](#usage-examples)
- [Storage & Caching](#storage--caching)
- [Best Practices](#best-practices)
- [Development](#development)
- [Summary](#summary)

---

## Overview

**SSLib** (Self-Supervised Learning Library) is a modular Python framework for self-supervised representation learning. It provides a **scikit-learn-inspired pipeline architecture** with automatic component discovery, intelligent caching, and extensible base classes.

### Key Features

- **Automatic Discovery**: Datasets, embedders, and processors are discovered dynamically at import time
- **Metadata-Driven Design**: Components self-describe their capabilities and properties
- **Intelligent Caching**: Built-in storage system for computed embeddings
- **Modular Architecture**: Easy to extend with new datasets, embedders, and processors
- **Pipeline Orchestration**: Compose multiple datasets × embedders × processors in one execution

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-capable GPU (optional, but recommended for faster embedding extraction)

### Basic Installation

Install SSLib using pip:

```bash
pip install ssllib
```

Or install from source:

```bash
git clone https://github.com/mmkuznecov/ssllib.git
cd ssllib
pip install -e .
```

### Installation with Optional Dependencies

For development work with all features:

```bash
pip install -e ".[dev,examples,all]"
```

Or install specific feature sets:

```bash
# Development tools only
pip install -e ".[dev]"

# Example notebooks and visualization
pip install -e ".[examples]"

# All optional features
pip install -e ".[all]"
```

### Using requirements.txt

Alternatively, install all dependencies using requirements.txt:

```bash
# Basic dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

### Verify Installation

Test your installation:

```python
import ssllib
from ssllib.datasets import list_datasets
from ssllib.embedders import list_embedders

print(f"SSLib version: {ssllib.__version__}")
print(f"Available datasets: {len(list_datasets())}")
print(f"Available embedders: {len(list_embedders())}")
```

### GPU Support

SSLib automatically detects and uses CUDA if available. To verify GPU support:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

To install PyTorch with specific CUDA version:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Quick Start

After installation, try this minimal example:

```python
from ssllib import Pipeline
from ssllib.datasets import SynthTestDataset
from ssllib.embedders.cv import DINOv2Embedder
from ssllib.processing import CovarianceProcessor

# Create a simple pipeline
pipeline = Pipeline([
    ('dataset', SynthTestDataset(tensors_num=100, seed=42)),
    ('embedder', DINOv2Embedder('dinov2_vits14')),
    ('processor', CovarianceProcessor())
])

# Execute
results = pipeline.execute()

# Inspect results
print(f"Embeddings shape: {results.get_embeddings('SynthTest', 'DINOv2_dinov2_vits14').shape}")
print(f"Execution time: {results.timing['total_time']:.2f}s")
```

### Troubleshooting

**Problem: ModuleNotFoundError for transformers or torch**

Solution: Ensure all dependencies are installed:
```bash
pip install torch transformers huggingface-hub
```

**Problem: CUDA out of memory**

Solution: Reduce batch size or use CPU:
```python
pipeline = Pipeline([...], config=Config({'batch_size': 16, 'device': 'cpu'}))
```

**Problem: Dataset download fails**

Solution: Check internet connection and try manual download. For CelebA:
1. Download from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. Extract to `data/CelebA/`

**Problem: Import errors after installation**

Solution: Reinstall in editable mode:
```bash
pip uninstall ssllib
pip install -e .
```

---

## Framework Architecture

```
SSLib/
├── core/                    # Core pipeline and configuration
│   ├── pipeline.py          # Pipeline orchestration
│   ├── config.py            # Configuration management
│   └── registry.py          # Generic discovery/registry
│
├── datasets/                # Dataset implementations
│   ├── __init__.py          # Auto-discovery registry
│   ├── base.py              # BaseDataset
│   ├── celeba.py            # CelebA dataset
│   ├── cifar10.py           # CIFAR-10
│   ├── food101.py           # Food-101
│   ├── imagenet100.py       # ImageNet-100
│   ├── synthtest_dataset.py # Synthetic test data
│   ├── hf_mixin.py          # HuggingFace helpers
│   └── kaggle_mixin.py      # Kaggle helpers
│
├── embedders/               # Embedding models
│   ├── __init__.py          # Auto-discovery registry
│   ├── base.py              # BaseEmbedder
│   ├── cv/                  # Computer vision embedders
│   │   ├── __init__.py
│   │   ├── dinov2.py        # DINOv2 models
│   │   ├── dino.py          # DINO (original)
│   │   ├── clip.py          # CLIP models
│   │   └── vicreg.py        # VICReg
│   └── nlp/                 # NLP embedders
│       ├── __init__.py
│       ├── bert.py          # BERT variants
│       ├── bert_base.py     # Base class for BERT-like models
│       ├── e5.py            # E5 multilingual
│       └── modernbert.py    # ModernBERT
│
├── processing/              # Post-processing & analysis
│   ├── __init__.py          # Auto-discovery registry for processors
│   ├── base.py              # BaseProcessor
│   ├── covariance.py        # Covariance computation
│   ├── zca.py               # ZCA whitening
│   ├── effective_rank.py    # Effective rank metric
│   ├── leverage_scores.py   # Row leverage scores
│   ├── stable_rank.py       # Stable rank metric
│   ├── spectrum.py          # Covariance spectrum
│   └── pairwise_stats.py    # Pairwise distance statistics
│
├── losses/                  # Loss functions (for training)
│   ├── __init__.py
│   ├── base.py              # BaseLoss
│   ├── infonce_loss.py      # InfoNCE (SimCLR)
│   ├── contrastive_loss.py  # Standard contrastive
│   ├── triplet_loss.py      # Triplet loss
│   └── deepinfomax_loss.py  # Deep InfoMax
│
└── storage/                 # Caching & persistence
    ├── __init__.py
    └── tensor_storage.py    # TensorStorage for embeddings
```

---

## Core Components

### 1. Pipeline

The `Pipeline` class orchestrates the entire workflow: datasets → embedders → processors.

**Key Capabilities:**
- Execute multiple dataset × embedder × processor combinations
- Automatic caching of computed embeddings
- Configuration management
- Timing and metadata tracking

```python
from ssllib import Pipeline, Config
from ssllib.datasets import SynthTestDataset
from ssllib.embedders.cv import DINOv2Embedder
from ssllib.processing import CovarianceProcessor

pipeline = Pipeline([
    ('dataset', SynthTestDataset(tensors_num=100)),
    ('embedder', DINOv2Embedder('dinov2_vitb14')),
    ('processor', CovarianceProcessor())
])

results = pipeline.execute()
```

### 2. Config

Manages configuration with dot-notation access and file loading.

```python
from ssllib import Config

# From dictionary
config = Config({
    'batch_size': 64,
    'device': 'cuda',
    'model': {
        'name': 'dinov2_vitb14'
    }
})

# Access with dot notation
batch_size = config.get('batch_size')
model_name = config.get('model.name')

# From file
config = Config.from_file('config.yaml')  # or .json
```

### 3. PipelineResults

Container for all pipeline outputs with convenient access methods.

```python
results = pipeline.execute()

# Access embeddings
embeddings = results.get_embeddings('SynthTest', 'DINOv2_dinov2_vitb14')

# Access processed outputs
covariance = results.get_processed('SynthTest', 'DINOv2_dinov2_vitb14', 'Covariance')

# Inspect metadata
print(results.metadata)
print(results.timing)
print(results.list_dataset_keys())
```

---

## Module Discovery System

SSLib uses **automatic component discovery** inspired by plugin architectures. At import time, the framework scans module directories and registers all valid components.

### How Discovery Works

#### 1. Registry Pattern

Each component type has a registry (`DatasetRegistry`, `EmbedderRegistry`):

```python
# In datasets/__init__.py
_dataset_registry = discover_dataset_classes()

# This populates:
# - _dataset_registry._datasets: Dict[str, Type[BaseDataset]]
# - _dataset_registry._descriptions: Dict[str, str]
# - _dataset_registry._categories: Dict[str, List[str]]
# - _dataset_registry._modalities: Dict[str, str]
```

#### 2. Discovery Process

```python
def discover_dataset_classes() -> DatasetRegistry:
    """Discover all dataset classes in the datasets module."""
    registry = DatasetRegistry()
    
    # Iterate through all modules
    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name
        
        # Skip special modules
        if module_name in ['__init__', 'base']:
            continue
        
        # Import the module
        module = importlib.import_module(f"{package_name}.{module_name}")
        
        # Find all classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if it's a valid dataset
            if (issubclass(obj, BaseDataset) and
                not inspect.isabstract(obj) and
                obj != BaseDataset and
                obj.__module__ == full_module_name):
                
                # Extract metadata from class
                description = extract_description(obj)
                category = obj.get_dataset_category()
                modality = obj.get_dataset_modality()
                properties = obj.get_dataset_properties()
                
                # Register it
                registry.register(name, obj, description, category, modality, properties)
    
    return registry
```

#### 3. Dynamic Export

All discovered classes are automatically exported:

```python
# Update module globals
globals().update(_exported_classes)

# Dynamic __all__
__all__ = [
    "BaseDataset",
    "get_available_datasets",
    "list_datasets",
    *_dataset_registry.list_datasets()  # All discovered classes!
]
```

### Benefits

- **Zero boilerplate**: Just create a class, it's automatically available
- **Self-documenting**: Metadata extracted from class attributes and docstrings
- **Type-safe**: Registry ensures type correctness
- **Inspectable**: Query available components programmatically

### Discovery API

```python
from ssllib.datasets import (
    list_datasets,
    get_dataset_info,
    print_available_datasets,
    get_vision_datasets,
    create_dataset
)

# List all datasets
all_datasets = list_datasets()  # ['CelebADataset', 'SynthTestDataset', ...]

# Filter by modality
vision_datasets = list_datasets(modality='vision')

# Get detailed info
info = get_dataset_info('CelebADataset')
# Returns: {
#   'name': 'CelebADataset',
#   'class': 'CelebADataset',
#   'description': 'CelebA Dataset for SSLib framework.',
#   'modality': 'vision',
#   'properties': {'num_attributes': 40, ...}
# }

# Pretty print all
print_available_datasets()

# Create by name
dataset = create_dataset('CelebADataset', split='train')
```

---

## Adding New Components

### Adding a New Dataset

#### Step 1: Create Dataset Class

Create a new file in `ssllib/datasets/` (e.g., `my_dataset.py`):

```python
from .base import BaseDataset
import torch
from typing import Iterator, Dict, Any, ClassVar

class MyDataset(BaseDataset):
    """My custom dataset for amazing things."""
    
    # Class-level metadata for discovery
    _dataset_category: ClassVar[str] = "vision"
    _dataset_modality: ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "image_size": (224, 224),
        "num_classes": 1000,
        "total_samples": 50000
    }
    
    def __init__(self, root: str = "data", split: str = "train", **kwargs):
        """Initialize dataset."""
        super().__init__("MyDataset", **kwargs)
        
        self.root = root
        self.split = split
        # ... your initialization
        
    def download(self) -> None:
        """Download dataset if needed."""
        if self._downloaded:
            return
        
        # Your download logic
        # ...
        
        self._downloaded = True
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over dataset."""
        for idx in range(len(self)):
            # Yield tensor (for pipeline compatibility)
            yield self._load_sample(idx)
            
    def __len__(self) -> int:
        """Return dataset size."""
        return self._num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get single item."""
        return self._load_sample(idx)
```

#### Step 2: That's It! 

The dataset is **automatically discovered** and available:

```python
from ssllib.datasets import MyDataset  # Auto-imported!

# Or create by name
from ssllib.datasets import create_dataset
dataset = create_dataset('MyDataset', split='train')

# Check it's registered
from ssllib.datasets import list_datasets
print('MyDataset' in list_datasets())  # True
```

### Adding a New Embedder

#### Step 1: Create Embedder Class

Create file in `ssllib/embedders/cv/` (or `nlp/`, `audio/`):

```python
from ..base import BaseEmbedder
import torch
from typing import Dict, Any, ClassVar

class MyEmbedder(BaseEmbedder):
    """My awesome embedder."""
    
    # Class-level metadata
    _embedder_category: ClassVar[str] = "vision"
    _embedder_modality: ClassVar[str] = "vision"
    _embedder_properties: ClassVar[Dict[str, Any]] = {
        "model_family": "MyFamily",
        "source": "MyOrg",
        "pretrained": True
    }
    
    AVAILABLE_MODELS = {
        "my_model_small": {
            "embedding_dim": 384,
            "hf_name": "myorg/my-model-small"
        },
        "my_model_large": {
            "embedding_dim": 768,
            "hf_name": "myorg/my-model-large"
        }
    }
    
    def __init__(self, model_name: str = "my_model_small", 
                 device: str = "cpu", **kwargs):
        """Initialize embedder."""
        super().__init__(f"MyEmbedder_{model_name}", device, **kwargs)
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
        self.embedding_dim = self.AVAILABLE_MODELS[model_name]["embedding_dim"]
        
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
        
    def load_model(self) -> None:
        """Load model."""
        if self._loaded:
            return
        
        # Your model loading logic
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            self.AVAILABLE_MODELS[self.model_name]["hf_name"]
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if not self._loaded:
            self.load_model()
            
        with torch.no_grad():
            outputs = self.model(pixel_values=batch)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
```

#### Step 2: Automatically Available!

```python
from ssllib.embedders.cv import MyEmbedder  # Auto-imported!

# Or by name
from ssllib.embedders import create_embedder
embedder = create_embedder('MyEmbedder', model_name='my_model_large')

# List all vision embedders
from ssllib.embedders import get_vision_embedders
print(get_vision_embedders())  # Includes 'MyEmbedder'
```

### Adding a New Processor

Processors are simpler (no discovery system yet):

```python
from .base import BaseProcessor
import numpy as np

class MyProcessor(BaseProcessor):
    """My custom processor."""
    
    def __init__(self, param: float = 1.0, **kwargs):
        super().__init__("MyProcessor", **kwargs)
        self.param = param
        
    def process(self, embeddings: np.ndarray) -> np.ndarray:
        """Process embeddings."""
        # Your processing logic
        result = embeddings * self.param
        
        # Update metadata
        self._metadata.update({
            "input_shape": embeddings.shape,
            "output_shape": result.shape
        })
        
        return result
```

Then import in `ssllib/processing/__init__.py`:

```python
from .my_processor import MyProcessor

__all__ = [..., "MyProcessor"]
```

---

## Usage Examples

### Basic Single Pipeline

```python
from ssllib import Pipeline, Config
from ssllib.datasets import SynthTestDataset
from ssllib.embedders.cv import DINOv2Embedder
from ssllib.processing import CovarianceProcessor

# Create pipeline
pipeline = Pipeline([
    ('dataset', SynthTestDataset(tensors_num=50, seed=42)),
    ('embedder', DINOv2Embedder('dinov2_vitb14')),
    ('processor', CovarianceProcessor())
])

# Execute
results = pipeline.execute()

# Access results
embeddings = results.get_embeddings('SynthTest', 'DINOv2_dinov2_vitb14')
covariance = results.get_processed('SynthTest', 'DINOv2_dinov2_vitb14', 'Covariance')

print(f"Embeddings shape: {embeddings.shape}")
print(f"Covariance shape: {covariance.shape}")
```

### Multi-Component Pipeline

Compute **all combinations** of datasets × embedders × processors:

```python
from ssllib import Pipeline
from ssllib.datasets import SynthTestDataset
from ssllib.embedders.cv import DINOv2Embedder, CLIPEmbedder
from ssllib.processing import CovarianceProcessor, ZCAProcessor

pipeline = Pipeline([
    ('datasets', [
        SynthTestDataset(tensors_num=100, seed=1),
        SynthTestDataset(tensors_num=100, seed=2)
    ]),
    ('embedders', [
        DINOv2Embedder('dinov2_vitb14'),
        CLIPEmbedder('clip-vit-large-patch14')
    ]),
    ('processors', [
        CovarianceProcessor(),
        ZCAProcessor(epsilon=1e-6)
    ])
])

# This executes: 2 datasets × 2 embedders × 2 processors = 8 combinations
results = pipeline.execute()

print(f"Total embeddings: {len(results.embeddings)}")  # 4 (2×2)
print(f"Total processed: {len(results.processed)}")    # 8 (2×2×2)

# Access specific combination
dataset_key = 'SynthTest'  # Or 'SynthTest[1]' for second instance
emb = results.get_embeddings(dataset_key, 'DINOv2_dinov2_vitb14')
zca = results.get_processed(dataset_key, 'DINOv2_dinov2_vitb14', 'ZCA')
```

### Configuration-Driven Pipeline

```python
from ssllib import Pipeline, Config
from ssllib.datasets import SynthTestDataset
from ssllib.embedders.cv import DINOv2Embedder, CLIPEmbedder
from ssllib.processing import CovarianceProcessor, ZCAProcessor

# Load configuration
config = Config.from_file('config.yaml')
# Or create programmatically
config = Config({
    'device': 'cuda',
    'batch_size': 64,
    'models': {
        'dinov2': 'dinov2_vitb14',
        'clip': 'clip-vit-large-patch14'
    },
    'processing': {
        'zca_epsilon': 1e-9
    }
})

# Build pipeline from config
pipeline = Pipeline([
    ('dataset', SynthTestDataset(tensors_num=200)),
    ('embedders', [
        DINOv2Embedder(config.get('models.dinov2')),
        CLIPEmbedder(config.get('models.clip'))
    ]),
    ('processors', [
        CovarianceProcessor(),
        ZCAProcessor(epsilon=config.get('processing.zca_epsilon'))
    ])
], config=config)

# Execute with runtime overrides
results = pipeline.execute(config_override={'batch_size': 32})
```

### Real Dataset Example

```python
from ssllib import Pipeline
from ssllib.datasets import CelebADataset
from ssllib.embedders.cv import DINOv2Embedder
from ssllib.processing import CovarianceProcessor, EffectiveRankProcessor

# Create pipeline with CelebA
pipeline = Pipeline([
    ('dataset', CelebADataset(split='train', task_name='Attractive')),
    ('embedder', DINOv2Embedder('dinov2_vits14')),  # Smaller for speed
    ('processors', [
        CovarianceProcessor(),
        EffectiveRankProcessor()
    ])
])

# Execute
results = pipeline.execute()

# Analyze embeddings
embeddings = results.get_embeddings('CelebA', 'DINOv2_dinov2_vits14')
covariance = results.get_processed('CelebA', 'DINOv2_dinov2_vits14', 'Covariance')
eff_rank = results.get_processed('CelebA', 'DINOv2_dinov2_vits14', 'EffectiveRank')

print(f"Dataset size: {len(embeddings)}")
print(f"Embedding dim: {embeddings.shape[1]}")
print(f"Effective rank: {eff_rank}")
```

---

## Storage & Caching

SSLib includes a built-in caching system to avoid recomputing expensive embeddings.

### Basic Caching

```python
from ssllib import Pipeline
from ssllib.datasets import SynthTestDataset
from ssllib.embedders.cv import DINOv2Embedder, CLIPEmbedder
from ssllib.processing import CovarianceProcessor

pipeline = Pipeline([
    ('datasets', [
        SynthTestDataset(tensors_num=100, seed=1),
        SynthTestDataset(tensors_num=100, seed=2)
    ]),
    ('embedders', [
        DINOv2Embedder('dinov2_vitb14'),
        CLIPEmbedder('clip-vit-large-patch14')
    ]),
    ('processor', CovarianceProcessor())
])

# First run: Compute and cache all embeddings
results1 = pipeline.execute(
    use_storage=True,
    storage_dir="./cache/my_experiment",
    storage_description="Testing caching system"
)

print(f"Cache hit rate: {results1.metadata['cache_hit_rate']:.2%}")  # 0%
print(f"Total time: {results1.timing['total_time']:.2f}s")

# Second run: Load from cache
results2 = pipeline.execute(
    use_storage=True,
    storage_dir="./cache/my_experiment"
)

print(f"Cache hit rate: {results2.metadata['cache_hit_rate']:.2%}")  # 100%
print(f"Total time: {results2.timing['total_time']:.2f}s")  # Much faster!
print(f"Speedup: {results1.timing['total_time'] / results2.timing['total_time']:.1f}x")
```

### Force Recompute

```python
# Ignore cache and recompute
results = pipeline.execute(
    use_storage=True,
    storage_dir="./cache/my_experiment",
    force_recompute=True  # Ignores cached embeddings
)
```

### Cache Key Generation

Caches are invalidated automatically when:
- Dataset configuration changes
- Dataset size changes  
- Dataset metadata changes
- Embedder changes

This is done via MD5 hashing of configuration:

```python
def _create_storage_key(self, dataset_key: str, embedder_name: str, 
                       dataset: BaseDataset) -> str:
    """Create unique storage key for dataset-embedder combination."""
    dataset_config = {
        'name': dataset.name,
        'size': len(dataset),
        'metadata': dataset.get_metadata()
    }
    
    config_str = json.dumps(dataset_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    return f"{dataset_key}_{embedder_name}_{config_hash}"
```

### Storage Structure

```
./cache/my_experiment/
├── data/
│   ├── chunk_0.npy
│   ├── chunk_1.npy
│   └── ...
└── metadata/
    └── metadata.json
```

Each cached embedding includes metadata:
- `dataset_key`: Unique dataset identifier
- `dataset_name`: Original dataset name
- `embedder_name`: Embedder used
- `embeddings_shape`: Shape of embeddings
- `timestamp`: When computed
- `storage_key`: Cache key for lookup

---

## Best Practices

### 1. Use Metadata Classes

Always define class-level metadata for discovery:

```python
class MyDataset(BaseDataset):
    _dataset_category: ClassVar[str] = "vision"
    _dataset_modality: ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "image_size": (224, 224),
        "num_classes": 10
    }
```

### 2. Implement `__iter__` for Pipeline Compatibility

Datasets should yield **tensors only** (not tuples):

```python
def __iter__(self) -> Iterator[torch.Tensor]:
    """Iterate over dataset - yield tensors only."""
    for idx in range(len(self)):
        image, label = self._load_sample(idx)
        yield image  # Pipeline needs just the image
```

### 3. Use Caching for Expensive Embeddings

```python
# Cache embeddings for later experimentation
results = pipeline.execute(
    use_storage=True,
    storage_dir=f"./cache/{experiment_name}"
)
```

### 4. Leverage Configuration Management

```python
# config.yaml
device: cuda
batch_size: 64
embedders:
  - dinov2_vitb14
  - clip-vit-large-patch14

# Python
config = Config.from_file('config.yaml')
embedders = [
    DINOv2Embedder(config.get('embedders.0')),
    CLIPEmbedder(config.get('embedders.1'))
]
```

### 5. Organize Experiments

```python
experiment_name = "celeba_dinov2_analysis"
storage_dir = f"./experiments/{experiment_name}/cache"

results = pipeline.execute(
    use_storage=True,
    storage_dir=storage_dir,
    storage_description=f"Experiment: {experiment_name}"
)

# Save additional results
import json
with open(f"./experiments/{experiment_name}/results.json", 'w') as f:
    json.dump(results.metadata, f, indent=2)
```


---

## Summary

**SSLib** provides a clean, modular framework for self-supervised learning experiments with:

- **Automatic discovery** - Add a class file, it's instantly available
- **Metadata-driven** - Components self-describe their capabilities
- **Pipeline orchestration** - Compose complex workflows easily
- **Intelligent caching** - Never recompute expensive embeddings
- **Extensible design** - Simple base classes for new components

### Next Steps

- Add training functionality (TrainingPipeline)
- Expand embedder collection (audio, multimodal)
- Add more processors (spectral analysis, representation quality metrics)
- Develop comprehensive benchmarking suite

---

## License

MIT License - see LICENSE file for details

## Citation

If you use SSLib in your research, please cite:

```bibtex
@software{sslib2024,
  author = {Mikhail Kuznetov},
  title = {SSLib: A Modular Framework for Self-Supervised Learning},
  year = {2024},
  url = {https://github.com/mmkuznecov/ssllib}
}
```