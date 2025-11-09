from typing import List, Tuple, Any, Dict, Union, Optional, Iterator
import time
import numpy as np
import hashlib
import os
from pathlib import Path
import json
import logging

from ..datasets.base import BaseDataset
from ..embedders.base import BaseEmbedder
from ..processing.base import BaseProcessor
from .config import Config
from ..storage.tensor_storage import TensorStorage

logger = logging.getLogger(__name__)


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

    def get_processed(
        self, dataset_key: str, embedder_name: str, processor_name: str
    ) -> np.ndarray:
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

        self._organize_components()

    def _organize_components(self) -> None:
        """Organize components by type from the input list."""
        for comp_type, comp in self.components:
            if comp_type in ["dataset", "datasets"]:
                if isinstance(comp, list):
                    self.datasets.extend(comp)
                else:
                    self.datasets.append(comp)
            elif comp_type in ["embedder", "embedders"]:
                if isinstance(comp, list):
                    self.embedders.extend(comp)
                else:
                    self.embedders.append(comp)
            elif comp_type in ["processor", "processors"]:
                if isinstance(comp, list):
                    self.processors.extend(comp)
                else:
                    self.processors.append(comp)

    def add_dataset(self, dataset: BaseDataset) -> "Pipeline":
        """Add dataset to pipeline."""
        self.datasets.append(dataset)
        return self

    def add_embedder(self, embedder: BaseEmbedder) -> "Pipeline":
        """Add embedder to pipeline."""
        self.embedders.append(embedder)
        return self

    def add_processor(self, processor: BaseProcessor) -> "Pipeline":
        """Add processor to pipeline."""
        self.processors.append(processor)
        return self

    def execute(
        self,
        config_override: Dict = None,
        use_storage: bool = False,
        storage_dir: Optional[str] = None,
        force_recompute: bool = False,
        storage_description: str = "",
    ) -> PipelineResults:
        """
        Execute the pipeline with optional storage caching.

        Refactored version with reduced complexity (target: ~8).

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

        try:
            # Stage 1: Validation and preparation
            self._validate_configuration()
            results = self._initialize_results()

            # Stage 2: Apply configuration
            dataset_keys = self._prepare_dataset_keys(results)
            self._apply_config_overrides(config_override)
            batch_size = self.config.get("batch_size", 32)

            # Stage 3: Setup storage
            storage = self._setup_storage_system(
                use_storage, storage_dir, storage_description, results
            )

            # Stage 4: Load data and models
            self._download_datasets()
            self._load_embedders()

            # Stage 5: Extract embeddings (with caching)
            storage_keys_map = self._prepare_storage_keys(dataset_keys)
            cached_embeddings = self._load_cached_embeddings(
                storage, storage_keys_map, use_storage, force_recompute
            )

            embedding_time = time.time()
            embeddings_to_save = self._extract_embeddings(
                dataset_keys,
                storage_keys_map,
                cached_embeddings,
                batch_size,
                use_storage,
                results,
            )
            results.timing["embedding_time"] = time.time() - embedding_time

            # Stage 6: Save new embeddings
            self._save_embeddings_to_cache(
                embeddings_to_save,
                storage,
                storage_dir,
                storage_description,
                use_storage,
                results,
            )

            # Stage 7: Log cache statistics
            self._log_cache_statistics(
                cached_embeddings, embeddings_to_save, use_storage, results
            )

            # Stage 8: Process embeddings
            if self.processors:
                self._process_embeddings(dataset_keys, results)

            # Stage 9: Collect final results
            self._collect_metadata(dataset_keys, storage, results)

            results.timing["total_time"] = time.time() - start_time
            logger.info(
                f"Pipeline execution completed in {results.timing['total_time']:.2f}s"
            )
            print(
                f"Pipeline execution completed in {results.timing['total_time']:.2f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Pipeline execution failed: {str(e)}") from e

    def _validate_configuration(self) -> None:
        """Validate pipeline configuration. Complexity: 2"""
        if not self.datasets:
            raise ValueError("Pipeline requires at least one dataset")

        if not self.embedders:
            raise ValueError("Pipeline requires at least one embedder")

    def _initialize_results(self) -> PipelineResults:
        """Initialize results container. Complexity: 1"""
        return PipelineResults()

    def _prepare_dataset_keys(self, results: PipelineResults) -> Dict[Any, str]:
        """
        Create unique keys for dataset instances.
        Complexity: 3
        """
        dataset_keys = self._create_unique_dataset_keys()

        # Store mapping in results
        for dataset, unique_key in dataset_keys.items():
            results.dataset_key_mapping[unique_key] = dataset.name

        return dataset_keys

    def _create_unique_dataset_keys(self) -> Dict[Any, str]:
        """Create unique keys for each dataset instance. Complexity: 4"""
        dataset_counts = {}
        dataset_keys = {}

        for dataset in self.datasets:
            base_name = dataset.name

            if base_name not in dataset_counts:
                dataset_counts[base_name] = 0
            else:
                dataset_counts[base_name] += 1

            # Create unique key
            if dataset_counts[base_name] == 0:
                unique_key = base_name
            else:
                unique_key = f"{base_name}[{dataset_counts[base_name]}]"

            dataset_keys[dataset] = unique_key

        return dataset_keys

    def _apply_config_overrides(self, config_override: Optional[Dict]) -> None:
        """Apply configuration overrides. Complexity: 2"""
        if not config_override:
            return

        for key, value in config_override.items():
            self.config.set(key, value)

    def _setup_storage_system(
        self,
        use_storage: bool,
        storage_dir: Optional[str],
        storage_description: str,
        results: PipelineResults,
    ) -> Optional[TensorStorage]:
        """
        Setup storage system if requested.
        Complexity: 4
        """
        if not use_storage:
            results.storage_info = {"enabled": False}
            return None

        # Auto-generate storage directory if needed
        if storage_dir is None:
            timestamp = int(time.time())
            storage_dir = f"./storage/pipeline_cache_{timestamp}"

        os.makedirs(storage_dir, exist_ok=True)
        storage = self._setup_storage(storage_dir, storage_description)

        results.storage_info = {
            "enabled": True,
            "directory": storage_dir,
            "description": storage_description,
        }

        return storage

    def _setup_storage(self, storage_dir: str, description: str = "") -> TensorStorage:
        """Setup or load existing storage. Complexity: 3"""
        metadata_path = os.path.join(storage_dir, "metadata", "metadata.json")

        if os.path.exists(storage_dir) and os.path.exists(metadata_path):
            logger.info(f"Loading existing storage from {storage_dir}")
            return TensorStorage(storage_dir)

        logger.info(f"Will create new storage in {storage_dir}")
        return None

    def _download_datasets(self) -> None:
        """Download all datasets. Complexity: 2"""
        logger.info("Downloading datasets...")
        print("Downloading datasets...")

        for dataset in self.datasets:
            try:
                dataset.download()
            except Exception as e:
                logger.error(f"Failed to download dataset {dataset.name}: {str(e)}")
                raise

    def _load_embedders(self) -> None:
        """Load all embedder models. Complexity: 2"""
        logger.info("Loading embedders...")
        print("Loading embedders...")

        for embedder in self.embedders:
            try:
                embedder.load_model()
            except Exception as e:
                logger.error(f"Failed to load embedder {embedder.name}: {str(e)}")
                raise

    def _prepare_storage_keys(
        self, dataset_keys: Dict[Any, str]
    ) -> Dict[Tuple[Any, Any], str]:
        """
        Create storage keys for all dataset-embedder combinations.
        Complexity: 3
        """
        storage_keys_map = {}

        for dataset in self.datasets:
            dataset_key = dataset_keys[dataset]
            for embedder in self.embedders:
                storage_key = self._create_storage_key(
                    dataset_key, embedder.name, dataset
                )
                storage_keys_map[(dataset, embedder)] = storage_key

        return storage_keys_map

    def _create_storage_key(
        self, dataset_key: str, embedder_name: str, dataset: BaseDataset
    ) -> str:
        """
        Create unique storage key for dataset-embedder combination.
        Complexity: 2
        """
        dataset_config = {
            "name": dataset.name,
            "size": len(dataset),
            "metadata": dataset.get_metadata(),
        }

        config_str = json.dumps(dataset_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        return f"{dataset_key}_{embedder_name}_{config_hash}"

    def _load_cached_embeddings(
        self,
        storage: Optional[TensorStorage],
        storage_keys_map: Dict[Tuple[Any, Any], str],
        use_storage: bool,
        force_recompute: bool,
    ) -> Dict[str, np.ndarray]:
        """
        Load cached embeddings from storage if available.
        Complexity: 3
        """
        if not use_storage or not storage or force_recompute:
            return {}

        storage_keys_needed = list(storage_keys_map.values())
        return self._load_embeddings_from_storage(storage, storage_keys_needed)

    def _load_embeddings_from_storage(
        self, storage: TensorStorage, storage_keys_needed: List[str]
    ) -> Dict[str, np.ndarray]:
        """Load embeddings from storage if they exist. Complexity: 4"""
        loaded_embeddings = {}

        if storage is None or storage.metadata_df is None or storage.metadata_df.empty:
            return loaded_embeddings

        for storage_key in storage_keys_needed:
            matches = storage.metadata_df[
                storage.metadata_df["storage_key"] == storage_key
            ]
            if len(matches) > 0:
                tensor_idx = matches.iloc[0]["tensor_idx"]
                embeddings = storage[tensor_idx]
                loaded_embeddings[storage_key] = embeddings
                logger.info(f"Loaded embeddings for {storage_key} from cache")
                print(f"Loaded embeddings for {storage_key} from cache")

        return loaded_embeddings

    def _extract_embeddings(
        self,
        dataset_keys: Dict[Any, str],
        storage_keys_map: Dict[Tuple[Any, Any], str],
        cached_embeddings: Dict[str, np.ndarray],
        batch_size: int,
        use_storage: bool,
        results: PipelineResults,
    ) -> Dict[str, Tuple[np.ndarray, Dict]]:
        """
        Extract embeddings for all dataset-embedder combinations.
        Complexity: 5
        """
        logger.info("Extracting embeddings...")
        print("Extracting embeddings...")

        embeddings_to_save = {}

        for dataset in self.datasets:
            dataset_key = dataset_keys[dataset]

            for embedder in self.embedders:
                storage_key = storage_keys_map[(dataset, embedder)]

                # Try to use cache or compute
                if storage_key in cached_embeddings:
                    embeddings = cached_embeddings[storage_key]
                    logger.info(
                        f"Using cached embeddings for {dataset_key} + {embedder.name}"
                    )
                    print(
                        f"Using cached embeddings for {dataset_key} + {embedder.name}"
                    )
                else:
                    embeddings = self._compute_embeddings(
                        dataset, embedder, dataset_key, batch_size
                    )

                    # Prepare for storage
                    if use_storage:
                        metadata = self._create_embedding_metadata(
                            dataset_key, dataset, embedder, embeddings, batch_size
                        )
                        embeddings_to_save[storage_key] = (embeddings, metadata)

                # Store in results
                self._store_embedding_results(
                    results, dataset_key, embedder.name, embeddings
                )

        return embeddings_to_save

    def _compute_embeddings(
        self,
        dataset: BaseDataset,
        embedder: BaseEmbedder,
        dataset_key: str,
        batch_size: int,
    ) -> np.ndarray:
        """Compute embeddings for a dataset. Complexity: 2"""
        logger.info(f"Computing embeddings for {dataset_key} + {embedder.name}")
        print(f"Computing embeddings for {dataset_key} + {embedder.name}")

        try:
            return embedder.embed_dataset(dataset, batch_size)
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {str(e)}")
            raise

    def _create_embedding_metadata(
        self,
        dataset_key: str,
        dataset: BaseDataset,
        embedder: BaseEmbedder,
        embeddings: np.ndarray,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Create metadata for embedding storage. Complexity: 1"""
        return {
            "dataset_key": dataset_key,
            "dataset_name": dataset.name,
            "embedder_name": embedder.name,
            "embeddings_shape": embeddings.shape,
            "dataset_size": len(dataset),
            "batch_size": batch_size,
            "timestamp": time.time(),
        }

    def _store_embedding_results(
        self,
        results: PipelineResults,
        dataset_key: str,
        embedder_name: str,
        embeddings: np.ndarray,
    ) -> None:
        """Store embeddings in results. Complexity: 1"""
        results.embeddings[(dataset_key, embedder_name)] = embeddings
        results.metadata[f"{dataset_key}_{embedder_name}_shape"] = embeddings.shape
        results.metadata[f"{dataset_key}_{embedder_name}_dtype"] = str(embeddings.dtype)

    def _save_embeddings_to_cache(
        self,
        embeddings_to_save: Dict[str, Tuple[np.ndarray, Dict]],
        storage: Optional[TensorStorage],
        storage_dir: str,
        storage_description: str,
        use_storage: bool,
        results: PipelineResults,
    ) -> None:
        """
        Save new embeddings to storage.
        Complexity: 5
        """
        if not use_storage or not embeddings_to_save:
            return

        save_time = time.time()

        if storage is None:
            storage = self._save_embeddings_to_storage(
                embeddings_to_save, storage_dir, storage_description
            )
        else:
            # Adding to existing storage not yet implemented
            logger.warning(
                "Adding to existing storage not implemented, creating new storage"
            )
            print(
                "Warning: Adding to existing storage not implemented, creating new storage"
            )
            storage = self._save_embeddings_to_storage(
                embeddings_to_save, storage_dir + "_new", storage_description
            )

        results.timing["storage_save_time"] = time.time() - save_time

        if storage:
            results.storage_info.update(storage.get_storage_info())

    def _save_embeddings_to_storage(
        self,
        embeddings_data: Dict[str, Tuple[np.ndarray, Dict]],
        storage_dir: str,
        description: str = "",
    ) -> TensorStorage:
        """Save embeddings to storage. Complexity: 3"""
        if not embeddings_data:
            return None

        def tensor_iterator() -> Iterator[np.ndarray]:
            for embeddings, _ in embeddings_data.values():
                yield embeddings

        def metadata_iterator() -> Iterator[Dict[str, Any]]:
            for storage_key, (_, metadata) in embeddings_data.items():
                metadata_dict = metadata.copy()
                metadata_dict["storage_key"] = storage_key
                yield metadata_dict

        logger.info(f"Saving {len(embeddings_data)} embeddings to storage...")
        print(f"Saving {len(embeddings_data)} embeddings to storage...")

        storage = TensorStorage.create_storage(
            storage_dir=storage_dir,
            data_iterator=tensor_iterator(),
            metadata_iterator=metadata_iterator(),
            description=description,
        )

        return storage

    def _log_cache_statistics(
        self,
        cached_embeddings: Dict[str, np.ndarray],
        embeddings_to_save: Dict[str, Tuple[np.ndarray, Dict]],
        use_storage: bool,
        results: PipelineResults,
    ) -> None:
        """Log cache hit/miss statistics. Complexity: 3"""
        if not use_storage:
            return

        cache_hits = len(cached_embeddings)
        cache_misses = len(embeddings_to_save)
        total_combinations = len(self.datasets) * len(self.embedders)

        logger.info(
            f"Cache statistics: {cache_hits} hits, {cache_misses} misses out of {total_combinations} total"
        )
        print(
            f"Cache statistics: {cache_hits} hits, {cache_misses} misses out of {total_combinations} total"
        )

        results.metadata["cache_hits"] = cache_hits
        results.metadata["cache_misses"] = cache_misses
        results.metadata["cache_hit_rate"] = (
            cache_hits / total_combinations if total_combinations > 0 else 0
        )

    def _process_embeddings(
        self, dataset_keys: Dict[Any, str], results: PipelineResults
    ) -> None:
        """
        Apply all processors to embeddings.
        Complexity: 4
        """
        logger.info("Processing embeddings...")
        print("Processing embeddings...")

        processing_time = time.time()

        for dataset in self.datasets:
            dataset_key = dataset_keys[dataset]

            for embedder in self.embedders:
                embeddings = results.embeddings[(dataset_key, embedder.name)]

                for processor in self.processors:
                    self._apply_single_processor(
                        processor, embeddings, dataset_key, embedder.name, results
                    )

        results.timing["processing_time"] = time.time() - processing_time

    def _apply_single_processor(
        self,
        processor: BaseProcessor,
        embeddings: np.ndarray,
        dataset_key: str,
        embedder_name: str,
        results: PipelineResults,
    ) -> None:
        """Apply a single processor. Complexity: 2"""
        logger.info(f"Processing {dataset_key}-{embedder_name} with {processor.name}")
        print(f"Processing {dataset_key}-{embedder_name} with {processor.name}")

        try:
            processed = processor.process(embeddings)
            results.processed[(dataset_key, embedder_name, processor.name)] = processed

            # Store metadata
            key_prefix = f"{dataset_key}_{embedder_name}_{processor.name}"
            results.metadata[f"{key_prefix}_shape"] = processed.shape
            results.metadata[f"{key_prefix}_dtype"] = str(processed.dtype)
        except Exception as e:
            logger.error(f"Failed to process with {processor.name}: {str(e)}")
            raise

    def _collect_metadata(
        self,
        dataset_keys: Dict[Any, str],
        storage: Optional[TensorStorage],
        results: PipelineResults,
    ) -> None:
        """Collect final metadata. Complexity: 3"""
        # Dataset metadata
        results.metadata["datasets"] = []
        for dataset in self.datasets:
            dataset_meta = dataset.get_metadata().copy()
            dataset_meta["pipeline_key"] = dataset_keys[dataset]
            results.metadata["datasets"].append(dataset_meta)

        # Embedder metadata
        results.metadata["embedders"] = [e.get_metadata() for e in self.embedders]

        # Processor metadata
        results.metadata["processors"] = [p.get_metadata() for p in self.processors]

        # Config
        results.metadata["config"] = self.config.to_dict()

        # Storage metadata
        if storage:
            results.metadata["storage"] = storage.get_storage_info()
