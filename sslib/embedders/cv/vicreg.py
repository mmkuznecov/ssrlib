import torch
import torch.nn as nn
from typing import Dict, Any

from ...core.base import BaseEmbedder


class VICRegEmbedder(BaseEmbedder):
    """VICReg embedder for computer vision."""
    
    AVAILABLE_MODELS = {
        "resnet50": {"embedding_dim": 2048},
        "resnet50x2": {"embedding_dim": 2048},
        "resnet50x4": {"embedding_dim": 2048},
    }
    
    def __init__(self, model_name: str = "resnet50", device: str = "cpu", **kwargs):
        """Initialize VICReg embedder.
        
        Args:
            model_name: Name of the VICReg model to use
            device: Device to run on
        """
        super().__init__(f"VICReg_{model_name}", device, **kwargs)
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model {model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
            
        self.model_name = model_name
        self.embedding_dim = self.AVAILABLE_MODELS[model_name]["embedding_dim"]
        
        # Update metadata
        self._metadata.update({
            "model_name": model_name,
            "embedding_dim": self.embedding_dim,
            "model_type": "VICReg"
        })
        
    def load_model(self) -> None:
        """Load VICReg model from torch hub."""
        if self._loaded:
            return
            
        print(f"Loading VICReg model: {self.model_name}")
        try:
            self.model = torch.hub.load("facebookresearch/vicreg:main", self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {str(e)}")
            
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through VICReg model.
        
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