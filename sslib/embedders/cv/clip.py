import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any

from ...core.base import BaseEmbedder


class CLIPEmbedder(BaseEmbedder):
    """CLIP embedder for computer vision."""
    
    AVAILABLE_MODELS = {
        "clip-vit-large-patch14": {
            "embedding_dim": 768,
            "hf_name": "openai/clip-vit-large-patch14"
        },
        "clip-vit-base-patch32": {
            "embedding_dim": 512,
            "hf_name": "openai/clip-vit-base-patch32"
        },
        "clip-vit-base-patch16": {
            "embedding_dim": 512,
            "hf_name": "openai/clip-vit-base-patch16"
        },
    }
    
    def __init__(self, model_name: str = "clip-vit-large-patch14", device: str = "cpu", **kwargs):
        """Initialize CLIP embedder.
        
        Args:
            model_name: Name of the CLIP model to use
            device: Device to run on
        """
        super().__init__(f"CLIP_{model_name}", device, **kwargs)
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model {model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
            
        self.model_name = model_name
        self.hf_name = self.AVAILABLE_MODELS[model_name]["hf_name"]
        self.embedding_dim = self.AVAILABLE_MODELS[model_name]["embedding_dim"]
        self.processor = None
        
        # Update metadata
        self._metadata.update({
            "model_name": model_name,
            "hf_name": self.hf_name,
            "embedding_dim": self.embedding_dim,
            "model_type": "CLIP"
        })
        
    def load_model(self) -> None:
        """Load CLIP model from Hugging Face."""
        if self._loaded:
            return
            
        print(f"Loading CLIP model: {self.hf_name}")
        try:
            self.model = CLIPModel.from_pretrained(self.hf_name)
            self.processor = CLIPProcessor.from_pretrained(self.hf_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {str(e)}")
            
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through CLIP model.
        
        Args:
            batch: Input batch of shape (batch_size, 3, H, W)
            
        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        if not self._loaded:
            self.load_model()
            
        with torch.no_grad():
            # CLIP expects pixel values in [0, 1] range
            # Assume input is already normalized, so we need to denormalize
            # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batch.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batch.device)
            
            # Denormalize
            denormalized = batch * std + mean
            denormalized = torch.clamp(denormalized, 0, 1)
            
            # Get image features
            embeddings = self.model.get_image_features(pixel_values=denormalized)
            
        return embeddings