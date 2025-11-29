"""Mixin for loading datasets from Hugging Face Hub."""

from pathlib import Path
from typing import Optional, Dict, Any
from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)


class HFDatasetMixin:
    """
    Mixin for datasets loaded from Hugging Face Hub.

    Subclasses must implement:
    - _get_hf_dataset_id() -> str
    - _get_hf_split_name(split: str) -> str
    - _get_hf_keys() -> Dict[str, str]
    """

    @abstractmethod
    def _get_hf_dataset_id(self) -> str:
        """Return HuggingFace dataset ID (e.g., 'ethz/food101')."""
        pass

    @abstractmethod
    def _get_hf_split_name(self, split: str) -> str:
        """
        Map split name to HF dataset split.

        Args:
            split: Requested split ('train', 'test', 'val')

        Returns:
            Actual HF split name
        """
        pass

    @abstractmethod
    def _get_hf_keys(self) -> Dict[str, str]:
        """
        Get HF dataset column keys.

        Returns:
            Dict with 'image' and 'label' keys
        """
        pass

    def _load_from_huggingface(self, split: str, cache_dir: Optional[str] = None) -> None:
        """
        Load dataset from Hugging Face Hub.

        Args:
            split: Dataset split to load
            cache_dir: Optional cache directory
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for HuggingFace datasets. "
                "Install it with: pip install datasets"
            )

        dataset_id = self._get_hf_dataset_id()
        hf_split = self._get_hf_split_name(split)

        logger.info(f"Loading HuggingFace dataset: {dataset_id}")
        print(f"Loading HuggingFace dataset: {dataset_id}")
        print(f"Split: {split} -> {hf_split}")

        try:
            self.hf_dataset = load_dataset(
                dataset_id,
                split=hf_split,
                cache_dir=cache_dir,
            )

            logger.info(f"✓ Loaded {len(self.hf_dataset)} examples")
            print(f"✓ Loaded {len(self.hf_dataset)} examples")

            # Get column keys
            self.hf_keys = self._get_hf_keys()
            self.image_key = self.hf_keys["image"]
            self.label_key = self.hf_keys["label"]

            # Setup label mapping if needed
            self._setup_label_mapping()

        except ValueError as e:
            if "trust_remote_code" in str(e).lower():
                logger.error(
                    f"Dataset {dataset_id} requires a loading script (deprecated). "
                    f"The dataset may need to be updated on HuggingFace Hub."
                )
                raise RuntimeError(
                    f"Cannot load dataset '{dataset_id}'. "
                    f"It uses a deprecated loading script. "
                    f"Please check if the dataset has been updated to Parquet format."
                ) from e
            else:
                raise

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {str(e)}")
            raise RuntimeError(f"Failed to load HuggingFace dataset: {str(e)}") from e

    def _setup_label_mapping(self) -> None:
        """
        Setup label mapping for string labels.

        Some datasets use string labels instead of integers.
        Creates bidirectional mapping: label <-> index.
        """
        self.label_to_idx = None
        self.idx_to_label = None

        try:
            # Check first label type
            if len(self.hf_dataset) == 0:
                logger.warning("Empty dataset, skipping label mapping")
                return

            first_label = self.hf_dataset[0][self.label_key]

            # If already numeric, no mapping needed
            if isinstance(first_label, (int, float)):
                logger.info("Labels are numeric, no mapping needed")
                return

            # Create mapping for string labels
            if isinstance(first_label, str):
                logger.info("Detected string labels, creating mapping...")

                # Try to get from features first (faster)
                all_labels = self._get_labels_from_features()

                # Fallback: collect from dataset
                if all_labels is None:
                    all_labels = self._collect_unique_labels()

                # Create bidirectional mapping
                sorted_labels = sorted(list(all_labels))
                self.label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
                self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

                logger.info(f"✓ Created label mapping for {len(self.label_to_idx)} classes")
                print(f"✓ Label mapping: {len(self.label_to_idx)} classes")

        except Exception as e:
            logger.warning(f"Could not setup label mapping: {str(e)}")

    def _get_labels_from_features(self) -> Optional[set]:
        """Try to get labels from dataset features."""
        try:
            if hasattr(self.hf_dataset, "features"):
                label_feature = self.hf_dataset.features.get(self.label_key)
                if hasattr(label_feature, "names"):
                    labels = set(label_feature.names)
                    logger.info(f"Got {len(labels)} labels from features")
                    return labels
        except Exception as e:
            logger.debug(f"Could not get labels from features: {e}")

        return None

    def _collect_unique_labels(self) -> set:
        """Collect unique labels from dataset."""
        logger.info("Collecting unique labels from dataset...")

        try:
            # Try to get entire column (fast)
            label_column = self.hf_dataset[self.label_key]
            return set(label_column)
        except Exception:
            # Fallback: sample from dataset
            logger.info("Sampling labels from dataset...")
            all_labels = set()
            sample_size = min(1000, len(self.hf_dataset))
            step = max(1, len(self.hf_dataset) // sample_size)

            for i in range(0, len(self.hf_dataset), step):
                try:
                    label = self.hf_dataset[i][self.label_key]
                    all_labels.add(label)
                except Exception:
                    continue

            logger.info(f"Collected {len(all_labels)} unique labels from sampling")
            return all_labels

    def _convert_label(self, label: Any) -> int:
        """
        Convert label to integer index.

        Args:
            label: Label from dataset (string or int)

        Returns:
            Integer label index
        """
        # If string label and we have mapping
        if isinstance(label, str) and self.label_to_idx is not None:
            return self.label_to_idx[label]

        # If already int
        if isinstance(label, (int, float)):
            return int(label)

        # Fallback
        return label

    def _get_num_classes(self) -> int:
        """
        Get number of classes in dataset.

        Returns:
            Number of unique classes
        """
        if self.label_to_idx is not None:
            return len(self.label_to_idx)

        # Try to get from features
        try:
            if hasattr(self.hf_dataset, "features"):
                label_feature = self.hf_dataset.features.get(self.label_key)
                if hasattr(label_feature, "num_classes"):
                    return label_feature.num_classes
                elif hasattr(label_feature, "names"):
                    return len(label_feature.names)
        except Exception:
            pass

        # Fallback: count unique labels
        logger.warning("Could not determine number of classes, counting unique labels...")
        try:
            unique_labels = set(self.hf_dataset[self.label_key])
            return len(unique_labels)
        except Exception:
            return 0

    def _get_class_names(self) -> Optional[list]:
        """
        Get class names if available.

        Returns:
            List of class names or None
        """
        # From label mapping
        if self.idx_to_label is not None:
            return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]

        # From features
        try:
            if hasattr(self.hf_dataset, "features"):
                label_feature = self.hf_dataset.features.get(self.label_key)
                if hasattr(label_feature, "names"):
                    return label_feature.names
        except Exception:
            pass

        return None
