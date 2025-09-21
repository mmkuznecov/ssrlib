import torch
import torch.nn as nn
from typing import Dict, Any

from ...core.base import BaseEmbedder


class DINOv2Embedder(BaseEmbedder):
    """DINOv2 embedder for computer vision."""
    
    AVAILABLE_MODELS = {
        "dinov2_vits14": {"embedding_dim": 384},
        "dinov2_vitb14": {"embedding_dim": 768},
        "dinov2_vitl14": {"embedding_dim": 1024},
        "dinov2_vitg14": {"embedding_dim": 1536},
        "dinov2_vits14_reg": {"embedding_dim": 384},
        "dinov2_vitb14_reg": {"embedding_dim": 768},
        "dinov2_vitl14_reg": {"embedding_dim": 1024},
        "dinov2_vitg14_reg": {"embedding_dim": 1536},
    }
    
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cpu", **kwargs):
        """Initialize DINOv2 embedder.
        
        Args:
            model_name: Name of the DINOv2 model to use
            device: Device to run on
        """
        super().__init__(f"DINOv2_{model_name}", device, **kwargs)
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model {model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
            
        self.model_name = model_name
        self.embedding_dim = self.AVAILABLE_MODELS[model_name]["embedding_dim"]
        
        # Update metadata
        self._metadata.update({
            "model_name": model_name,
            "embedding_dim": self.embedding_dim,
            "model_type": "DINOv2"
        })
        
    def load_model(self) -> None:
        """Load DINOv2 model from torch hub."""
        if self._loaded:
            return
            
        print(f"Loading DINOv2 model: {self.model_name}")
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {str(e)}")
            
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through DINOv2 model.
        
        Args:
            batch: Input batch of shape (batch_size, 3, H, W)
            
        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        if not self._loaded:
            self.load_model()
            
        with torch.no_grad():
            embeddings = self.model(batch)
            
        return embeddings
