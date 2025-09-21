from typing import List, Tuple, Any, Dict, Union
import time
import numpy as np
from .base import BaseDataset, BaseEmbedder, BaseProcessor
from .config import Config


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
    """Main pipeline class for orchestrating SSLib components."""
    
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
        """Create unique keys for each dataset instance.
        
        Returns:
            Dictionary mapping dataset objects to unique keys
        """
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
        
    def execute(self, config_override: Dict = None) -> PipelineResults:
        """Execute the pipeline.
        
        Args:
            config_override: Configuration overrides
            
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
        
        # Download datasets
        print("Downloading datasets...")
        for dataset in self.datasets:
            dataset.download()
            
        # Load embedders
        print("Loading embedders...")
        for embedder in self.embedders:
            embedder.load_model()
            
        # Extract embeddings
        print("Extracting embeddings...")
        embedding_time = time.time()
        
        for dataset in self.datasets:
            dataset_key = dataset_keys[dataset]
            for embedder in self.embedders:
                print(f"Processing {dataset_key} with {embedder.name}")
                
                embeddings = embedder.embed_dataset(dataset, batch_size)
                results.embeddings[(dataset_key, embedder.name)] = embeddings
                
                # Store metadata
                results.metadata[f"{dataset_key}_{embedder.name}_shape"] = embeddings.shape
                results.metadata[f"{dataset_key}_{embedder.name}_dtype"] = str(embeddings.dtype)
                
        results.timing['embedding_time'] = time.time() - embedding_time
        
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
        
        print(f"Pipeline execution completed in {results.timing['total_time']:.2f}s")
        return results


# Example usage showing the fix:

def test_unique_tracking():
    """Test the updated pipeline with unique dataset tracking."""
    
    from sslib.datasets import SynthTestDataset
    from sslib.embedders.cv import DINOv2Embedder, CLIPEmbedder
    from sslib.processing import CovarianceProcessor, ZCAProcessor
    
    # Create pipeline with multiple identical dataset types
    pipeline = Pipeline([
        ('datasets', [
            SynthTestDataset(tensors_num=20, tensor_shape=(3, 224, 224), seed=1),
            SynthTestDataset(tensors_num=20, tensor_shape=(3, 224, 224), seed=2),
            SynthTestDataset(tensors_num=15, tensor_shape=(3, 224, 224), seed=3)
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
    
    # Execute pipeline
    results = pipeline.execute()
    
    print(f"Total embeddings computed: {len(results.embeddings)}")  # Should be 6 (3×2)
    print(f"Total processed outputs: {len(results.processed)}")      # Should be 12 (3×2×2)
    
    # Show dataset key mapping
    print("\nDataset key mapping:")
    for key, original_name in results.dataset_key_mapping.items():
        print(f"  {key} -> {original_name}")
    
    # Show all embedding combinations - now each dataset is unique
    print("\nEmbedding combinations:")
    for (dataset_key, embedder), emb in results.embeddings.items():
        print(f"{dataset_key} + {embedder}: {emb.shape}")
    
    # Show processed combinations  
    print("\nProcessed combinations:")
    for (dataset_key, embedder, processor), proc in results.processed.items():
        print(f"{dataset_key} + {embedder} + {processor}: {proc.shape}")
    
    # Example of accessing specific results
    print("\nAccessing specific results:")
    emb1 = results.get_embeddings("SynthTest", "DINOv2_dinov2_vitb14")
    emb2 = results.get_embeddings("SynthTest[1]", "DINOv2_dinov2_vitb14") 
    emb3 = results.get_embeddings("SynthTest[2]", "DINOv2_dinov2_vitb14")
    
    print(f"First SynthTest embeddings: {emb1.shape if emb1 is not None else 'None'}")
    print(f"Second SynthTest embeddings: {emb2.shape if emb2 is not None else 'None'}")
    print(f"Third SynthTest embeddings: {emb3.shape if emb3 is not None else 'None'}")


if __name__ == "__main__":
    test_unique_tracking()