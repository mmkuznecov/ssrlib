"""Food101 dataset from HuggingFace."""

from torchvision import transforms
from typing import Dict, Any, ClassVar

from .hf_vision import HFVisionDataset


class Food101Dataset(HFVisionDataset):
    """Food-101 Dataset from HuggingFace Hub."""

    # Class-level metadata
    _dataset_category: ClassVar[str] = "vision"
    _dataset_modality: ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "num_classes": 101,
        "image_format": "jpg",
        "processed_image_size": (224, 224),
        "task_type": "multi_class_classification",
        "source": "huggingface",
        "hf_id": "ethz/food101",
    }

    DEFAULT_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self, split: str = "train", **kwargs):
        """Initialize Food101 dataset."""
        super().__init__(name="Food101", registry_key="food101", split=split, **kwargs)
