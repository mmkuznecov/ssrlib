import torch
import torch.nn as nn
from transformers import ResNetModel, ViTModel
from typing import Dict, Any

from ...core.base import BaseEmbedder


class DINOEmbedder(BaseEmbedder):
    """DINO (original) embedder for computer vision."""
    
    AVAILABLE_MODELS = {
        "dino_resnet50": {
            "embedding_dim": 2048,
            "hf_name": "Ramos-Ramos/dino-resnet-50",
            "model_type": "resnet"
        },
        "dino_vitb8": {
            "embedding_dim": 768,
            "hf_name": "facebook/dino-vitb8",
            "model_type": "vit"
        },
        "dino_vitb16": {
            "embedding_dim": 768,
            "hf_name": "facebook/dino-vitb16",
            "model_type": "vit"
        },
        "dino_vits8": {
            "embedding_dim": 384,
            "hf_name": "facebook/dino-vits8",
            "model_type": "vit"
        },
        "dino_vits16": {
            "embedding_dim": 384,
            "hf_name": "facebook/dino-vits16",
            "model_type": "vit"
        },
    }
    
    def __init__(self, model_name: str = "dino_vitb16", device: str = "cpu", **kwargs):
        """Initialize DINO embedder.
        
        Args:
            model_name: Name of the DINO model to use
            device: Device to run on
        """
        super().__init__(f"DINO_{model_name}", device, **kwargs)
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model {model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
            
        self.model_name = model_name
        self.hf_name = self.AVAILABLE_MODELS[model_name]["hf_name"]
        self.model_type = self.AVAILABLE_MODELS[model_name]["model_type"]
        self.embedding_dim = self.AVAILABLE_MODELS[model_name]["embedding_dim"]
        
        # Update metadata
        self._metadata.update({
            "model_name": model_name,
            "hf_name": self.hf_name,
            "model_type": f"DINO_{self.model_type}",
            "embedding_dim": self.embedding_dim
        })
        
    def load_model(self) -> None:
        """Load DINO model from Hugging Face."""
        if self._loaded:
            return
            
        print(f"Loading DINO model: {self.hf_name}")
        try:
            if self.model_type == "resnet":
                self.model = ResNetModel.from_pretrained(self.hf_name)
            elif self.model_type == "vit":
                self.model = ViTModel.from_pretrained(self.hf_name)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {str(e)}")
            
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through DINO model.
        
        Args:
            batch: Input batch of shape (batch_size, 3, H, W)
            
        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        if not self._loaded:
            self.load_model()
            
        with torch.no_grad():
            if self.model_type == "resnet":
                outputs = self.model(pixel_values=batch)
                embeddings = outputs.pooler_output
            elif self.model_type == "vit":
                outputs = self.model(pixel_values=batch)
                # Use mean pooling over sequence dimension for ViT
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        return embeddings
