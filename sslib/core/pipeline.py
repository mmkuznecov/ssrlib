from typing import List, Tuple, Any, Dict, Union, Optional, Iterator
import time
import numpy as np
import hashlib
import os
from pathlib import Path
import json

from .base import BaseDataset, BaseEmbedder, BaseProcessor
from .config import Config
from ..storage.tensor_storage import TensorStorage


class PipelineResults:
    """Container for pipeline execution results."""
    
    def __init__(self):
        # (dataset_key, embedder_name) -> embeddings
        self.embeddings: Dict[Tuple[str, str], np.ndarray] = {}
        # (dataset_key, embedder_name, processor_name) -> processed_data
        self.processed: Dict[Tuple[str, str, str], np.ndarray] = {}
        # General metadata
        self.metadata: Dict[str, Any] = {}
        # Timing information
        self.timing: Dict[str, float] = {}
        # Mapping from dataset_key to original dataset name
        self.dataset_key_mapping: Dict[str, str] = {}
        # Storage information
        self.storage_info: Optional[Dict[str, Any]] = None
        
    def get_embeddings(self, dataset_key: str, embedder_name: str) -> np.ndarray:
        """Get embeddings for specific dataset-embedder combination."""
        return self.embeddings.get((dataset_key, embedder_name))
        
    def get_processed(self, dataset_key: str, embedder_name: str, processor_name: str) -> np.ndarray:
        """Get processed data for specific dataset-embedder-processor combination."""
        return self.processed.get((dataset_key, embedder_name, processor_name))
    
    def list_dataset_keys(self) -> List[str]:
        """List all unique dataset keys."""
        return list(self.dataset_key_mapping.keys())
    
    def get_original_dataset_name(self, dataset_key: str) -> str:
        """Get original dataset name from dataset key."""
        return self.dataset_key_mapping.get(dataset_key, dataset_key)


class Pipeline:
    """Main pipeline class for orchestrating SSLib components with storage support."""
    
    def __init__(self, components: List[Tuple[str, Any]], config: Config = None):
        """Initialize pipeline.
        
        Args:
            components: List of (component_type, component) tuples
            config: Configuration object
        """
        self.components = components
        self.config = config or Config()
        
        # Organize components by type
        self.datasets = []
        self.embedders = []
        self.processors = []
        
        for comp_type, comp in components:
            if comp_type == 'dataset':
                if isinstance(comp, list):
                    self.datasets.extend(comp)
                else:
                    self.datasets.append(comp)
            elif comp_type == 'datasets':
                if isinstance(comp, list):
                    self.datasets.extend(comp)
                else:
                    self.datasets.append(comp)
            elif comp_type == 'embedder':
                if isinstance(comp, list):
                    self.embedders.extend(comp)
                else:
                    self.embedders.append(comp)
            elif comp_type == 'embedders':
                if isinstance(comp, list):
                    self.embedders.extend(comp)
                else:
                    self.embedders.append(comp)
            elif comp_type == 'processor':
                if isinstance(comp, list):
                    self.processors.extend(comp)
                else:
                    self.processors.append(comp)
            elif comp_type == 'processors':
                if isinstance(comp, list):
                    self.processors.extend(comp)
                else:
                    self.processors.append(comp)
                    
    def add_dataset(self, dataset: BaseDataset) -> 'Pipeline':
        """Add dataset to pipeline."""
        self.datasets.append(dataset)
        return self
        
    def add_embedder(self, embedder: BaseEmbedder) -> 'Pipeline':
        """Add embedder to pipeline."""
        self.embedders.append(embedder)
        return self
        
    def add_processor(self, processor: BaseProcessor) -> 'Pipeline':
        """Add processor to pipeline."""
        self.processors.append(processor)
        return self
    
    def _create_unique_dataset_keys(self) -> Dict[Any, str]:
        """Create unique keys for each dataset instance."""
        dataset_counts = {}
        dataset_keys = {}
        
        for dataset in self.datasets:
            base_name = dataset.name
            
            # Count how many instances of this dataset name we've seen
            if base_name not in dataset_counts:
                dataset_counts[base_name] = 0
            else:
                dataset_counts[base_name] += 1
            
            # Create unique key
            if dataset_counts[base_name] == 0:
                # First instance keeps original name
                unique_key = base_name
            else:
                # Subsequent instances get numbered
                unique_key = f"{base_name}[{dataset_counts[base_name]}]"
            
            dataset_keys[dataset] = unique_key
            
        return dataset_keys
    
    def _create_storage_key(self, dataset_key: str, embedder_name: str, dataset: BaseDataset) -> str:
        """Create unique storage key for dataset-embedder combination.
        
        Args:
            dataset_key: Unique dataset key
            embedder_name: Name of embedder
            dataset: Dataset instance for getting configuration hash
            
        Returns:
            Unique storage key
        """
        # Create hash from dataset configuration for cache invalidation
        dataset_config = {
            'name': dataset.name,
            'size': len(dataset),
            'metadata': dataset.get_metadata()
        }
        
        config_str = json.dumps(dataset_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{dataset_key}_{embedder_name}_{config_hash}"
    
    def _setup_storage(self, storage_dir: str, description: str = "") -> TensorStorage:
        """Setup or load existing storage.
        
        Args:
            storage_dir: Directory for storage
            description: Description for new storage
            
        Returns:
            TensorStorage instance
        """
        if os.path.exists(storage_dir) and os.path.exists(os.path.join(storage_dir, "metadata", "metadata.json")):
            print(f"Loading existing storage from {storage_dir}")
            return TensorStorage(storage_dir)
        else:
            print(f"Will create new storage in {storage_dir}")
            return None  # Will be created later when we have data
    
    def _load_embeddings_from_storage(self, storage: TensorStorage, 
                                    storage_keys_needed: List[str]) -> Dict[str, np.ndarray]:
        """Load embeddings from storage if they exist.
        
        Args:
            storage: TensorStorage instance
            storage_keys_needed: List of storage keys to look for
            
        Returns:
            Dictionary mapping storage keys to embeddings
        """
        loaded_embeddings = {}
        
        if storage is None or storage.metadata_df is None or storage.metadata_df.empty:
            return loaded_embeddings
        
        for storage_key in storage_keys_needed:
            # Query storage for this key
            matches = storage.metadata_df[storage.metadata_df['storage_key'] == storage_key]
            if len(matches) > 0:
                tensor_idx = matches.iloc[0]['tensor_idx']
                embeddings = storage[tensor_idx]
                loaded_embeddings[storage_key] = embeddings
                print(f"Loaded embeddings for {storage_key} from cache")
        
        return loaded_embeddings
    
    def _save_embeddings_to_storage(self, embeddings_data: Dict[str, Tuple[np.ndarray, Dict]], 
                                  storage_dir: str, description: str = "") -> TensorStorage:
        """Save embeddings to storage.
        
        Args:
            embeddings_data: Dictionary mapping storage keys to (embeddings, metadata) tuples
            storage_dir: Directory for storage
            description: Storage description
            
        Returns:
            TensorStorage instance
        """
        if not embeddings_data:
            return None
        
        def tensor_iterator() -> Iterator[np.ndarray]:
            for embeddings, _ in embeddings_data.values():
                yield embeddings
        
        def metadata_iterator() -> Iterator[Dict[str, Any]]:
            for storage_key, (_, metadata) in embeddings_data.items():
                metadata_dict = metadata.copy()
                metadata_dict['storage_key'] = storage_key
                yield metadata_dict
        
        print(f"Saving {len(embeddings_data)} embeddings to storage...")
        storage = TensorStorage.create_storage(
            storage_dir=storage_dir,
            data_iterator=tensor_iterator(),
            metadata_iterator=metadata_iterator(),
            description=description
        )
        
        return storage
        
    def execute(self, config_override: Dict = None, 
                use_storage: bool = False,
                storage_dir: Optional[str] = None,
                force_recompute: bool = False,
                storage_description: str = "") -> PipelineResults:
        """Execute the pipeline with optional storage caching.
        
        Args:
            config_override: Configuration overrides
            use_storage: Whether to use storage for caching embeddings
            storage_dir: Directory for storage (auto-generated if None)
            force_recompute: Whether to force recomputation even if cached
            storage_description: Description for storage
            
        Returns:
            PipelineResults containing all computed embeddings and processed data
        """
        start_time = time.time()
        results = PipelineResults()
        
        # Create unique keys for dataset instances
        dataset_keys = self._create_unique_dataset_keys()
        
        # Store the mapping in results for reference
        for dataset, unique_key in dataset_keys.items():
            results.dataset_key_mapping[unique_key] = dataset.name
        
        # Apply config overrides
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)
                
        batch_size = self.config.get('batch_size', 32)
        
        # Setup storage if requested
        storage = None
        if use_storage:
            if storage_dir is None:
                # Auto-generate storage directory name
                timestamp = int(time.time())
                storage_dir = f"./storage/pipeline_cache_{timestamp}"
            
            os.makedirs(storage_dir, exist_ok=True)
            storage = self._setup_storage(storage_dir, storage_description)
            results.storage_info = {
                'enabled': True,
                'directory': storage_dir,
                'description': storage_description
            }
        else:
            results.storage_info = {'enabled': False}
        
        # Download datasets
        print("Downloading datasets...")
        for dataset in self.datasets:
            dataset.download()
            
        # Load embedders
        print("Loading embedders...")
        for embedder in self.embedders:
            embedder.load_model()
        
        # Create storage keys for all combinations
        storage_keys_map = {}  # (dataset, embedder) -> storage_key
        storage_keys_needed = []
        
        for dataset in self.datasets:
            dataset_key = dataset_keys[dataset]
            for embedder in self.embedders:
                storage_key = self._create_storage_key(dataset_key, embedder.name, dataset)
                storage_keys_map[(dataset, embedder)] = storage_key
                storage_keys_needed.append(storage_key)
        
        # Try to load existing embeddings from storage
        cached_embeddings = {}
        if use_storage and storage and not force_recompute:
            cached_embeddings = self._load_embeddings_from_storage(storage, storage_keys_needed)
        
        # Extract embeddings
        print("Extracting embeddings...")
        embedding_time = time.time()
        
        embeddings_to_save = {}  # For new embeddings that need to be saved
        cache_hits = 0
        cache_misses = 0
        
        for dataset in self.datasets:
            dataset_key = dataset_keys[dataset]
            for embedder in self.embedders:
                storage_key = storage_keys_map[(dataset, embedder)]
                
                # Check if we have cached embeddings
                if storage_key in cached_embeddings:
                    embeddings = cached_embeddings[storage_key]
                    cache_hits += 1
                    print(f"Using cached embeddings for {dataset_key} + {embedder.name}")
                else:
                    # Compute embeddings
                    print(f"Computing embeddings for {dataset_key} + {embedder.name}")
                    embeddings = embedder.embed_dataset(dataset, batch_size)
                    cache_misses += 1
                    
                    # Prepare for storage
                    if use_storage:
                        metadata = {
                            'dataset_key': dataset_key,
                            'dataset_name': dataset.name,
                            'embedder_name': embedder.name,
                            'embeddings_shape': embeddings.shape,
                            'dataset_size': len(dataset),
                            'batch_size': batch_size,
                            'timestamp': time.time()
                        }
                        embeddings_to_save[storage_key] = (embeddings, metadata)
                
                # Store in results
                results.embeddings[(dataset_key, embedder.name)] = embeddings
                
                # Store metadata
                results.metadata[f"{dataset_key}_{embedder.name}_shape"] = embeddings.shape
                results.metadata[f"{dataset_key}_{embedder.name}_dtype"] = str(embeddings.dtype)
                
        results.timing['embedding_time'] = time.time() - embedding_time
        
        # Save new embeddings to storage
        if use_storage and embeddings_to_save:
            save_time = time.time()
            if storage is None:
                # Create new storage
                storage = self._save_embeddings_to_storage(embeddings_to_save, storage_dir, storage_description)
            else:
                # TODO: Add embeddings to existing storage (would require extending TensorStorage)
                print("Warning: Adding to existing storage not implemented, creating new storage")
                storage = self._save_embeddings_to_storage(embeddings_to_save, storage_dir + "_new", storage_description)
            
            results.timing['storage_save_time'] = time.time() - save_time
            
            if storage:
                results.storage_info.update(storage.get_storage_info())
        
        # Log cache statistics
        if use_storage:
            total_combinations = len(self.datasets) * len(self.embedders)
            print(f"Cache statistics: {cache_hits} hits, {cache_misses} misses out of {total_combinations} total")
            results.metadata['cache_hits'] = cache_hits
            results.metadata['cache_misses'] = cache_misses
            results.metadata['cache_hit_rate'] = cache_hits / total_combinations if total_combinations > 0 else 0
        
        # Process embeddings
        if self.processors:
            print("Processing embeddings...")
            processing_time = time.time()
            
            for dataset in self.datasets:
                dataset_key = dataset_keys[dataset]
                for embedder in self.embedders:
                    embeddings = results.embeddings[(dataset_key, embedder.name)]
                    
                    for processor in self.processors:
                        print(f"Processing {dataset_key}-{embedder.name} with {processor.name}")
                        
                        processed = processor.process(embeddings)
                        results.processed[(dataset_key, embedder.name, processor.name)] = processed
                        
                        # Store metadata
                        key_prefix = f"{dataset_key}_{embedder.name}_{processor.name}"
                        results.metadata[f"{key_prefix}_shape"] = processed.shape
                        results.metadata[f"{key_prefix}_dtype"] = str(processed.dtype)
                        
            results.timing['processing_time'] = time.time() - processing_time
            
        results.timing['total_time'] = time.time() - start_time
        
        # Store component metadata (using unique keys)
        results.metadata['datasets'] = []
        for dataset in self.datasets:
            dataset_meta = dataset.get_metadata().copy()
            dataset_meta['pipeline_key'] = dataset_keys[dataset]
            results.metadata['datasets'].append(dataset_meta)
            
        results.metadata['embedders'] = [e.get_metadata() for e in self.embedders]
        results.metadata['processors'] = [p.get_metadata() for p in self.processors]
        results.metadata['config'] = self.config.to_dict()
        
        # Storage metadata
        if storage:
            results.metadata['storage'] = storage.get_storage_info()
        
        print(f"Pipeline execution completed in {results.timing['total_time']:.2f}s")
        return results


# Example usage with storage integration
def example_with_storage():
    """Example showing pipeline with storage caching."""
    from sslib.datasets import SynthTestDataset
    from sslib.embedders.cv import DINOv2Embedder, CLIPEmbedder
    from sslib.processing import CovarianceProcessor, ZCAProcessor
    
    # Create pipeline
    pipeline = Pipeline([
        ('datasets', [
            SynthTestDataset(tensors_num=50, tensor_shape=(3, 224, 224), seed=1),
            SynthTestDataset(tensors_num=30, tensor_shape=(3, 224, 224), seed=2)
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
    
    print("=== First execution (cache miss) ===")
    # First execution - will compute and cache all embeddings
    results1 = pipeline.execute(
        use_storage=True,
        storage_dir="./cache/pipeline_test",
        storage_description="Test pipeline with two synthetic datasets"
    )
    
    print(f"Cache hit rate: {results1.metadata.get('cache_hit_rate', 0):.2%}")
    print(f"Storage info: {results1.storage_info}")
    
    print("\n=== Second execution (cache hit) ===")
    # Second execution - should load from cache
    results2 = pipeline.execute(
        use_storage=True,
        storage_dir="./cache/pipeline_test",
        force_recompute=False  # Use cache
    )
    
    print(f"Cache hit rate: {results2.metadata.get('cache_hit_rate', 0):.2%}")
    print(f"Timing comparison:")
    print(f"  First run: {results1.timing['total_time']:.2f}s")
    print(f"  Second run: {results2.timing['total_time']:.2f}s")
    print(f"  Speedup: {results1.timing['total_time']/results2.timing['total_time']:.1f}x")
    
    print("\n=== Force recompute ===")
    # Third execution - force recompute
    results3 = pipeline.execute(
        use_storage=True,
        storage_dir="./cache/pipeline_test",
        force_recompute=True  # Ignore cache
    )
    
    print(f"Cache hit rate: {results3.metadata.get('cache_hit_rate', 0):.2%}")


if __name__ == "__main__":
    example_with_storage()