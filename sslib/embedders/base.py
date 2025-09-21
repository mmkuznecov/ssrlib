"""Base embedder implementation for SSLib."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import logging

from ..core.base import BaseEmbedder

logger = logging.getLogger(__name__)


class ModelWrapper(nn.Module):
    """Wrapper for different model types to extract embeddings."""

    def __init__(self, model: nn.Module, model_type: str, processor: Optional[Any] = None):
        """Initialize model wrapper.
        
        Args:
            model: The actual model
            model_type: Type of model for proper handling
            processor: Optional processor (e.g., for CLIP)
        """
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.processor = processor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Embeddings tensor
        """
        if self.model_type == "dinov2":
            # DINOv2 models
            return self.model(x)
        elif self.model_type == "clip":
            # CLIP model
            if self.processor is not None:
                # For CLIP, we need to process images differently
                return self.model.get_image_features(x)
            else:
                return self.model.vision_model(pixel_values=x).pooler_output
        elif self.model_type == "vicreg":
            # VICReg models
            return self.model(x)
        elif self.model_type == "dino_resnet":
            # DINO ResNet
            return self.model(pixel_values=x).pooler_output
        elif self.model_type == "dino_vit":
            # DINO ViT
            outputs = self.model(pixel_values=x)
            return outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


class SSLibEmbedder(BaseEmbedder):
    """Base embedder class for SSLib with common functionality."""
    
    def __init__(self, name: str, model_type: str, device: str = "cpu", 
                 batch_size: int = 32, **kwargs):
        """Initialize embedder.
        
        Args:
            name: Model name
            model_type: Type of model for proper handling
            device: Device to use
            batch_size: Default batch size
            **kwargs: Additional configuration
        """
        super().__init__(name, device, **kwargs)
        self.model_type = model_type
        self.batch_size = batch_size
        self.wrapped_model: Optional[ModelWrapper] = None
        
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from batch.
        
        Args:
            batch: Input tensor of shape (batch_size, ...)
            
        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        if not self.is_loaded:
            self.load_model()
        
        self.wrapped_model.eval()
        with torch.no_grad():
            embeddings = self.wrapped_model(batch)
            
            # Handle different output formats
            if isinstance(embeddings, torch.Tensor):
                return embeddings
            elif hasattr(embeddings, "last_hidden_state"):
                return embeddings.last_hidden_state.mean(dim=1)
            elif hasattr(embeddings, "pooler_output"):
                return embeddings.pooler_output
            else:
                raise ValueError(f"Unexpected output format: {type(embeddings)}")
    
    def embed_dataset(self, dataset, batch_size: Optional[int] = None) -> np.ndarray:
        """Extract embeddings for entire dataset with batching.
        
        Args:
            dataset: Dataset to embed
            batch_size: Batch size for processing
            
        Returns:
            Embeddings array of shape (n_samples, embedding_dim)
        """
        if not self.is_loaded:
            self.load_model()
        
        batch_size = batch_size or self.batch_size
        embeddings = []
        current_batch = []
        
        logger.info(f"Extracting embeddings with {self.name} (batch_size={batch_size})")
        
        for i, sample in enumerate(dataset):
            current_batch.append(sample)
            
            if len(current_batch) == batch_size:
                batch_tensor = torch.stack(current_batch).to(self.device)
                batch_embeddings = self.forward(batch_tensor)
                embeddings.append(batch_embeddings.cpu().numpy())
                current_batch = []
                
                if (i + 1) % (batch_size * 10) == 0:
                    logger.info(f"  Processed {i + 1} samples")
        
        # Process remaining samples
        if current_batch:
            batch_tensor = torch.stack(current_batch).to(self.device)
            batch_embeddings = self.forward(batch_tensor)
            embeddings.append(batch_embeddings.cpu().numpy())
        
        result = np.concatenate(embeddings, axis=0)
        logger.info(f"  Extracted embeddings shape: {result.shape}")
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "name": self.name,
            "model_type": self.model_type,
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "embedding_dim": self.get_embedding_dim(),
        }
        
        if self.is_loaded and self.wrapped_model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.wrapped_model.parameters())
            info["total_parameters"] = total_params
            info["trainable_parameters"] = sum(
                p.numel() for p in self.wrapped_model.parameters() if p.requires_grad
            )
        
        return info
    
    @abstractmethod
    def _load_model_implementation(self) -> Tuple[nn.Module, Optional[Any]]:
        """Load the actual model implementation.
        
        Returns:
            Tuple of (model, optional_processor)
        """
        pass
    
    def load_model(self) -> None:
        """Load pretrained model."""
        if self.is_loaded:
            return
        
        logger.info(f"Loading model: {self.name}")
        
        try:
            model, processor = self._load_model_implementation()
            model = model.to(self.device)
            model.eval()
            
            self.wrapped_model = ModelWrapper(model, self.model_type, processor)
            self.is_loaded = True
            
            logger.info(f"Successfully loaded {self.name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.name}: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.wrapped_model is not None:
            del self.wrapped_model
            self.wrapped_model = None
        
        self.is_loaded = False
        
        # Clear GPU cache if using CUDA
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info(f"Unloaded model: {self.name}")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload_model()
        except:
            pass