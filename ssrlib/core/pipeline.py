"""Pipeline orchestration for ssrlib.

The pipeline is intentionally thin: it organises components, calls
``download``/``load_model`` once each, fans out the cartesian product of
(dataset, embedder), and then applies processors to each embedding tensor.

When ``streaming=True``, processors that implement ``MapReduceMixin`` are
fed batches via ``partial_fit`` / ``finalize`` instead of receiving the
whole embedding matrix. Processors that don't support streaming fall back
to the whole-array path.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..datasets.base import BaseDataset
from ..embedders.base import BaseEmbedder
from ..processing.base import BaseProcessor
from ..processing.map_reduce import MapReduceMixin
from .config import Config

logger = logging.getLogger(__name__)


class PipelineResults:
    """Container for pipeline execution results."""

    def __init__(self) -> None:
        # (dataset_key, embedder_name) -> embeddings
        self.embeddings: Dict[Tuple[str, str], np.ndarray] = {}
        # (dataset_key, embedder_name, processor_name) -> processed_array
        self.processed: Dict[Tuple[str, str, str], np.ndarray] = {}
        self.metadata: Dict[str, Any] = {}
        self.timing: Dict[str, float] = {}
        self.dataset_key_mapping: Dict[str, str] = {}

    def get_embeddings(
        self, dataset_key: str, embedder_name: str
    ) -> Optional[np.ndarray]:
        return self.embeddings.get((dataset_key, embedder_name))

    def get_processed(
        self, dataset_key: str, embedder_name: str, processor_name: str
    ) -> Optional[np.ndarray]:
        return self.processed.get((dataset_key, embedder_name, processor_name))

    def list_dataset_keys(self) -> List[str]:
        return list(self.dataset_key_mapping.keys())

    def get_original_dataset_name(self, dataset_key: str) -> str:
        return self.dataset_key_mapping.get(dataset_key, dataset_key)


class Pipeline:
    """Pipeline orchestrating datasets, embedders, and processors."""

    def __init__(
        self, components: List[Tuple[str, Any]], config: Optional[Config] = None
    ):
        self.components = components
        self.config = config or Config()
        self.datasets: List[BaseDataset] = []
        self.embedders: List[BaseEmbedder] = []
        self.processors: List[BaseProcessor] = []
        self._organize_components()

    def _organize_components(self) -> None:
        for comp_type, comp in self.components:
            target = None
            if comp_type in ("dataset", "datasets"):
                target = self.datasets
            elif comp_type in ("embedder", "embedders"):
                target = self.embedders
            elif comp_type in ("processor", "processors"):
                target = self.processors
            else:
                raise ValueError(
                    f"Unknown component type {comp_type!r}; "
                    "expected one of {dataset(s), embedder(s), processor(s)}"
                )
            if isinstance(comp, list):
                target.extend(comp)
            else:
                target.append(comp)

    # ------------------------------------------------------------- builders
    def add_dataset(self, dataset: BaseDataset) -> "Pipeline":
        self.datasets.append(dataset)
        return self

    def add_embedder(self, embedder: BaseEmbedder) -> "Pipeline":
        self.embedders.append(embedder)
        return self

    def add_processor(self, processor: BaseProcessor) -> "Pipeline":
        self.processors.append(processor)
        return self

    # -------------------------------------------------------------- execute
    def execute(
        self,
        config_override: Optional[Dict] = None,
        streaming: bool = False,
    ) -> PipelineResults:
        """Run the pipeline.

        Args:
            config_override: dict of dotted-keys to override the existing config.
            streaming: if True, processors that implement ``MapReduceMixin`` are
                fed batches incrementally. Other processors fall back to the
                whole-array path.
        """
        start = time.time()

        if not self.datasets:
            raise ValueError("Pipeline requires at least one dataset")
        if not self.embedders:
            raise ValueError("Pipeline requires at least one embedder")

        if config_override:
            for k, v in config_override.items():
                self.config.set(k, v)
        batch_size = int(self.config.get("batch_size", 32))

        results = PipelineResults()
        dataset_keys = self._unique_dataset_keys(results)

        # Download / load
        for d in self.datasets:
            d.download()
        for e in self.embedders:
            e.load_model()

        # Extract embeddings
        t0 = time.time()
        for d in self.datasets:
            key = dataset_keys[d]
            for e in self.embedders:
                logger.info("Embedding %s with %s", key, e.name)
                embs = e.embed_dataset(d, batch_size)
                results.embeddings[(key, e.name)] = embs
                results.metadata[f"{key}_{e.name}_shape"] = embs.shape
                results.metadata[f"{key}_{e.name}_dtype"] = str(embs.dtype)
        results.timing["embedding_time"] = time.time() - t0

        # Process
        if self.processors:
            t0 = time.time()
            for d in self.datasets:
                key = dataset_keys[d]
                for e in self.embedders:
                    embs = results.embeddings[(key, e.name)]
                    for p in self.processors:
                        out = self._run_processor(p, embs, batch_size, streaming)
                        results.processed[(key, e.name, p.name)] = out
                        prefix = f"{key}_{e.name}_{p.name}"
                        results.metadata[f"{prefix}_shape"] = out.shape
                        results.metadata[f"{prefix}_dtype"] = str(out.dtype)
            results.timing["processing_time"] = time.time() - t0

        # Final metadata
        results.metadata["datasets"] = []
        for d in self.datasets:
            meta = d.get_metadata().copy()
            meta["pipeline_key"] = dataset_keys[d]
            results.metadata["datasets"].append(meta)
        results.metadata["embedders"] = [e.get_metadata() for e in self.embedders]
        results.metadata["processors"] = [p.get_metadata() for p in self.processors]
        results.metadata["config"] = self.config.to_dict()
        results.metadata["streaming"] = bool(streaming)

        results.timing["total_time"] = time.time() - start
        logger.info("Pipeline completed in %.2fs", results.timing["total_time"])
        return results

    # ------------------------------------------------------------- helpers
    def _unique_dataset_keys(self, results: PipelineResults) -> Dict[Any, str]:
        """Make unique keys per dataset instance, suffixing duplicates as ``[1]``, ``[2]``..."""
        counts: Dict[str, int] = {}
        keys: Dict[Any, str] = {}
        for d in self.datasets:
            base = d.name
            counts[base] = counts.get(base, -1) + 1
            unique = base if counts[base] == 0 else f"{base}[{counts[base]}]"
            keys[d] = unique
            results.dataset_key_mapping[unique] = d.name
        return keys

    def _run_processor(
        self,
        processor: BaseProcessor,
        embeddings: np.ndarray,
        batch_size: int,
        streaming: bool,
    ) -> np.ndarray:
        """Run a single processor, picking the streaming or batch path."""
        if streaming and isinstance(processor, MapReduceMixin):
            processor.reset()
            for start in range(0, embeddings.shape[0], batch_size):
                batch = embeddings[start : start + batch_size]
                processor.partial_fit(batch)
            return processor.finalize()
        return processor.process(embeddings)
