import unittest
import torch
from unittest.mock import patch, Mock

from sslib.embedders.cv.dinov2 import DINOv2Embedder
from sslib.embedders.cv.clip import CLIPEmbedder
from sslib.embedders.cv.vicreg import VICRegEmbedder
from sslib.embedders.cv.dino import DINOEmbedder


class TestEmbedders(unittest.TestCase):
    """Test embedder classes."""

    def test_dinov2_embedder_creation(self):
        embedder = DINOv2Embedder("dinov2_vitb14")

        self.assertEqual(embedder.model_name, "dinov2_vitb14")
        self.assertEqual(embedder.embedding_dim, 768)
        self.assertEqual(embedder.device, "cpu")

    def test_clip_embedder_creation(self):
        embedder = CLIPEmbedder("clip-vit-large-patch14")

        self.assertEqual(embedder.model_name, "clip-vit-large-patch14")
        self.assertEqual(embedder.embedding_dim, 768)

    def test_vicreg_embedder_creation(self):
        embedder = VICRegEmbedder("resnet50")

        self.assertEqual(embedder.model_name, "resnet50")
        self.assertEqual(embedder.embedding_dim, 2048)

    def test_dino_embedder_creation(self):
        embedder = DINOEmbedder("dino_vitb16")

        self.assertEqual(embedder.model_name, "dino_vitb16")
        self.assertEqual(embedder.embedding_dim, 768)

    def test_invalid_model_names(self):
        with self.assertRaises(ValueError):
            DINOv2Embedder("invalid_model")

        with self.assertRaises(ValueError):
            CLIPEmbedder("invalid_model")

    @patch("torch.hub.load")
    def test_dinov2_model_loading(self, mock_hub_load):
        mock_model = Mock()
        mock_hub_load.return_value = mock_model

        embedder = DINOv2Embedder("dinov2_vitb14")
        embedder.load_model()

        self.assertTrue(embedder._loaded)
        mock_hub_load.assert_called_once_with(
            "facebookresearch/dinov2", "dinov2_vitb14"
        )

    @patch("torch.hub.load")
    def test_dinov2_forward(self, mock_hub_load):
        mock_model = Mock()
        mock_model.return_value = torch.randn(2, 768)
        mock_hub_load.return_value = mock_model

        embedder = DINOv2Embedder("dinov2_vitb14")
        batch = torch.randn(2, 3, 224, 224)

        embeddings = embedder.forward(batch)

        self.assertEqual(embeddings.shape, (2, 768))
