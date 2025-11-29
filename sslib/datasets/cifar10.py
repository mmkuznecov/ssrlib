"""CIFAR-10 dataset from HuggingFace."""

from torchvision import transforms
from typing import Dict, Any, ClassVar

from .hf_vision import HFVisionDataset


class CIFAR10Dataset(HFVisionDataset):
    """CIFAR-10 Dataset from HuggingFace Hub."""

    # Class-level metadata
    _dataset_category: ClassVar[str] = "vision"
    _dataset_modality: ClassVar[str] = "vision"
    _dataset_properties: ClassVar[Dict[str, Any]] = {
        "num_classes": 10,
        "image_format": "png",
        "original_image_size": (32, 32),
        "processed_image_size": (224, 224),
        "task_type": "multi_class_classification",
        "source": "huggingface",
        "hf_id": "uoft-cs/cifar10",
    }

    # Upscale from 32x32 to 224x224
    DEFAULT_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self, split: str = "train", **kwargs):
        """Initialize CIFAR-10 dataset."""
        super().__init__(name="CIFAR10", registry_key="cifar10", split=split, **kwargs)

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata with CIFAR-10 specific info."""
        metadata = super().get_metadata()
        metadata["original_size"] = "(3, 32, 32)"
        metadata["image_shape"] = "(3, 224, 224)"
        return metadata
