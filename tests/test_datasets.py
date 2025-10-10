import unittest
import tempfile
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock
from PIL import Image
import numpy as np

from sslib.datasets.celeba import CelebADataset
from sslib.datasets.imagenet100 import ImageNet100Dataset


class TestCelebADataset(unittest.TestCase):
    """Test CelebA dataset."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        # Create mock CSV files
        split_data = pd.DataFrame(
            {
                "image_id": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "partition": [0, 0, 1],  # train, train, valid
            }
        )
        split_data.to_csv(self.root / "list_eval_partition.csv", index=False)

        attr_data = pd.DataFrame(
            {"image_id": ["img1.jpg", "img2.jpg", "img3.jpg"], "Attractive": [1, -1, 1]}
        )
        attr_data.to_csv(self.root / "list_attr_celeba.csv", index=False)

        # Create images directory
        img_dir = self.root / "img_align_celeba"
        img_dir.mkdir()

        # Create mock images
        for img_name in ["img1.jpg", "img2.jpg"]:
            img = Image.new("RGB", (64, 64), color="red")
            img.save(img_dir / img_name)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_dataset_creation(self):
        dataset = CelebADataset(
            root=str(self.root), split="train", task_name="Attractive"
        )

        self.assertEqual(dataset.name, "CelebA")
        self.assertEqual(dataset.split, "train")
        self.assertEqual(dataset.task_name, "Attractive")

    def test_dataset_loading(self):
        dataset = CelebADataset(
            root=str(self.root), split="train", task_name="Attractive"
        )

        dataset.download()  # This loads the data
        self.assertEqual(len(dataset), 2)  # 2 train samples

        # Test metadata
        metadata = dataset.get_metadata()
        self.assertEqual(metadata["num_samples"], 2)
        self.assertEqual(metadata["split"], "train")


class TestImageNet100Dataset(unittest.TestCase):
    """Test ImageNet100 dataset."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        # Create train directory structure
        train_dir = self.root / "train.X1" / "n01440764"
        train_dir.mkdir(parents=True)

        # Create val directory structure
        val_dir = self.root / "val.X" / "n01440764"
        val_dir.mkdir(parents=True)

        # Create mock images
        for i in range(3):
            img = Image.new("RGB", (64, 64), color="blue")
            img.save(train_dir / f"train_img_{i}.jpg")

        for i in range(2):
            img = Image.new("RGB", (64, 64), color="green")
            img.save(val_dir / f"val_img_{i}.jpg")

        # Create Labels.json
        labels = {"n01440764": "fish, goldfish"}
        import json

        with open(self.root / "Labels.json", "w") as f:
            json.dump(labels, f)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_dataset_creation(self):
        dataset = ImageNet100Dataset(root=str(self.root), split="train")

        self.assertEqual(dataset.name, "ImageNet100")
        self.assertEqual(dataset.split, "train")

    def test_dataset_loading(self):
        dataset = ImageNet100Dataset(root=str(self.root), split="train")

        dataset.download()
        self.assertEqual(len(dataset), 3)  # 3 train samples

        # Test metadata
        metadata = dataset.get_metadata()
        self.assertEqual(metadata["num_samples"], 3)
        self.assertEqual(metadata["split"], "train")

    def test_validation_split(self):
        dataset = ImageNet100Dataset(root=str(self.root), split="val")

        dataset.download()
        self.assertEqual(len(dataset), 2)  # 2 val samples
