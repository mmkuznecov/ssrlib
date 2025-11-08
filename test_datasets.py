"""Test script for SSLib datasets."""

import sys
import os
from pathlib import Path

# Add parent directory to path to import sslib
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sslib.datasets import CelebADataset, ImageNet100Dataset, SynthTestDataset


class DatasetTester:
    """Helper class for testing datasets."""
    
    def __init__(self, root_dir: str = "data_new"):
        self.root_dir = root_dir
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def test(self, name: str, condition: bool, error_msg: str = ""):
        """Test a condition and track results."""
        if condition:
            print(f"  ✅ {name}")
            self.passed += 1
        else:
            print(f"  ❌ {name}")
            if error_msg:
                print(f"     Error: {error_msg}")
            self.failed += 1
            self.errors.append(f"{name}: {error_msg}")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print(f"TEST SUMMARY: {self.passed} passed, {self.failed} failed")
        if self.errors:
            print("\nFailed tests:")
            for error in self.errors:
                print(f"  - {error}")
        print("="*60)
    
    def test_basic_operations(self, dataset, dataset_name: str):
        """Test basic dataset operations."""
        print(f"\n{'='*60}")
        print(f"Testing {dataset_name}")
        print(f"{'='*60}")
        
        # Test 1: Dataset has non-zero length
        try:
            length = len(dataset)
            self.test(
                "Dataset has non-zero length",
                length > 0,
                f"Length is {length}"
            )
        except Exception as e:
            self.test("Dataset has non-zero length", False, str(e))
            return  # Can't continue without length
        
        # Test 2: Can get single item
        try:
            item = dataset[0]
            if isinstance(item, tuple):
                img, label = item
                self.test(
                    "Can get single item (returns tuple)",
                    isinstance(img, torch.Tensor) and isinstance(label, torch.Tensor),
                    f"Got types: {type(img)}, {type(label)}"
                )
            else:
                self.test(
                    "Can get single item (returns tensor)",
                    isinstance(item, torch.Tensor),
                    f"Got type: {type(item)}"
                )
        except Exception as e:
            self.test("Can get single item", False, str(e))
        
        # Test 3: Can get negative index
        try:
            item = dataset[-1]
            self.test("Can get negative index", True)
        except Exception as e:
            self.test("Can get negative index", False, str(e))
        
        # Test 4: Can slice dataset
        try:
            items = dataset[0:2]
            self.test(
                "Can slice dataset",
                isinstance(items, list) and len(items) == 2,
                f"Got {type(items)} with length {len(items) if isinstance(items, list) else 'N/A'}"
            )
        except Exception as e:
            self.test("Can slice dataset", False, str(e))
        
        # Test 5: Can iterate over dataset (test first 3 items)
        try:
            count = 0
            for i, item in enumerate(dataset):
                count += 1
                if i >= 2:  # Only test first 3 items
                    break
            self.test(
                "Can iterate over dataset",
                count == 3,
                f"Iterated {count} items"
            )
        except Exception as e:
            self.test("Can iterate over dataset", False, str(e))
        
        # Test 6: __repr__ works
        try:
            repr_str = repr(dataset)
            self.test(
                "__repr__ works",
                isinstance(repr_str, str) and len(repr_str) > 0,
                f"Got: {repr_str[:50]}..."
            )
        except Exception as e:
            self.test("__repr__ works", False, str(e))
        
        # Test 7: get_metadata works
        try:
            metadata = dataset.get_metadata()
            self.test(
                "get_metadata works",
                isinstance(metadata, dict) and len(metadata) > 0,
                f"Got {len(metadata)} metadata fields"
            )
        except Exception as e:
            self.test("get_metadata works", False, str(e))
        
        print(f"\nDataset info:")
        print(f"  Length: {len(dataset)}")
        print(f"  Repr: {repr(dataset)}")
    
    def test_celeba_specific(self, dataset: CelebADataset):
        """Test CelebA-specific functionality."""
        print(f"\nCelebA-specific tests:")
        
        # Test get_classes
        try:
            classes = dataset.get_classes()
            self.test(
                "get_classes returns dict",
                isinstance(classes, dict) and "num_classes" in classes,
                f"Got: {classes}"
            )
        except Exception as e:
            self.test("get_classes returns dict", False, str(e))
        
        # Test get_all_attributes
        try:
            attrs = dataset.get_all_attributes()
            self.test(
                "get_all_attributes returns list",
                isinstance(attrs, list) and len(attrs) == 40,
                f"Got {len(attrs)} attributes"
            )
        except Exception as e:
            self.test("get_all_attributes returns list", False, str(e))
        
        # Test get_sample_info
        try:
            info = dataset.get_sample_info(0)
            self.test(
                "get_sample_info works",
                isinstance(info, dict) and "image_id" in info,
                f"Got keys: {list(info.keys())}"
            )
        except Exception as e:
            self.test("get_sample_info works", False, str(e))
    
    def test_imagenet100_specific(self, dataset: ImageNet100Dataset):
        """Test ImageNet100-specific functionality."""
        print(f"\nImageNet100-specific tests:")
        
        # Test get_classes
        try:
            classes = dataset.get_classes()
            self.test(
                "get_classes returns dict",
                isinstance(classes, dict) and "num_classes" in classes,
                f"Has {classes.get('num_classes', 0)} classes"
            )
        except Exception as e:
            self.test("get_classes returns dict", False, str(e))
        
        # Test get_class_name
        try:
            if len(dataset.class_names) > 0:
                class_name = dataset.get_class_name(0)
                self.test(
                    "get_class_name works",
                    isinstance(class_name, str),
                    f"Got: {class_name}"
                )
        except Exception as e:
            self.test("get_class_name works", False, str(e))
        
        # Test get_class_index
        try:
            if len(dataset.class_names) > 0:
                idx = dataset.get_class_index(dataset.class_names[0])
                self.test(
                    "get_class_index works",
                    isinstance(idx, int),
                    f"Got: {idx}"
                )
        except Exception as e:
            self.test("get_class_index works", False, str(e))
        
        # Test get_sample_info
        try:
            info = dataset.get_sample_info(0)
            self.test(
                "get_sample_info works",
                isinstance(info, dict) and "class_name" in info,
                f"Got keys: {list(info.keys())}"
            )
        except Exception as e:
            self.test("get_sample_info works", False, str(e))


def test_synthetic_dataset():
    """Test synthetic dataset (no download needed)."""
    print("\n" + "="*60)
    print("TESTING SYNTHETIC DATASET (No Download)")
    print("="*60)
    
    tester = DatasetTester()
    
    try:
        dataset = SynthTestDataset(tensors_num=100, seed=42)
        tester.test_basic_operations(dataset, "SynthTestDataset")
        
        # Test reproducibility with seed
        print("\nReproducibility test:")
        dataset1 = SynthTestDataset(tensors_num=10, seed=42)
        dataset2 = SynthTestDataset(tensors_num=10, seed=42)
        
        item1 = dataset1[0]
        item2 = dataset2[0]
        
        tester.test(
            "Same seed produces same data",
            torch.allclose(item1, item2),
            "Tensors don't match"
        )
        
        # Test different seeds produce different data
        dataset3 = SynthTestDataset(tensors_num=10, seed=123)
        item3 = dataset3[0]
        
        tester.test(
            "Different seed produces different data",
            not torch.allclose(item1, item3),
            "Tensors match when they shouldn't"
        )
        
    except Exception as e:
        print(f"❌ Failed to create SynthTestDataset: {e}")
        import traceback
        traceback.print_exc()
    
    return tester


def test_celeba_dataset(root_dir: str = "data_new", skip_download: bool = False):
    """Test CelebA dataset."""
    print("\n" + "="*60)
    print("TESTING CELEBA DATASET")
    print("="*60)
    
    tester = DatasetTester(root_dir)
    
    if skip_download:
        # Check if dataset exists
        celeba_path = Path(root_dir) / "CelebA"
        if not celeba_path.exists():
            print(f"⚠️  Skipping CelebA test (dataset not found at {celeba_path})")
            print("   To download, run without --skip-download flag")
            return tester
    
    try:
        print(f"\nInitializing CelebA dataset (train split)...")
        print(f"Will download to: {Path(root_dir).absolute()}")
        dataset = CelebADataset(root=root_dir, split="train", task_name="Attractive")
        
        # Basic tests
        tester.test_basic_operations(dataset, "CelebADataset")
        
        # CelebA-specific tests
        tester.test_celeba_specific(dataset)
        
        # Test different splits
        print("\nTesting validation split:")
        try:
            val_dataset = CelebADataset(root=root_dir, split="valid", task_name="Smiling")
            tester.test(
                "Can create validation split",
                len(val_dataset) > 0,
                f"Length: {len(val_dataset)}"
            )
        except Exception as e:
            tester.test("Can create validation split", False, str(e))
        
    except Exception as e:
        print(f"❌ Failed to test CelebA: {e}")
        import traceback
        traceback.print_exc()
    
    return tester


def test_imagenet100_dataset(root_dir: str = "data_new", skip_download: bool = False):
    """Test ImageNet100 dataset."""
    print("\n" + "="*60)
    print("TESTING IMAGENET100 DATASET")
    print("="*60)
    
    tester = DatasetTester(root_dir)
    
    if skip_download:
        # Check if dataset exists
        imagenet_path = Path(root_dir) / "ImageNet100"
        if not imagenet_path.exists():
            print(f"⚠️  Skipping ImageNet100 test (dataset not found at {imagenet_path})")
            print("   To download, run without --skip-download flag")
            return tester
    
    try:
        print(f"\nInitializing ImageNet100 dataset (train split)...")
        print(f"Will download to: {Path(root_dir).absolute()}")
        dataset = ImageNet100Dataset(root=root_dir, split="train", combine_train_splits=True)
        
        # Basic tests
        tester.test_basic_operations(dataset, "ImageNet100Dataset")
        
        # ImageNet100-specific tests
        tester.test_imagenet100_specific(dataset)
        
        # Test validation split
        print("\nTesting validation split:")
        try:
            val_dataset = ImageNet100Dataset(root=root_dir, split="val")
            tester.test(
                "Can create validation split",
                len(val_dataset) > 0,
                f"Length: {len(val_dataset)}"
            )
        except Exception as e:
            tester.test("Can create validation split", False, str(e))
        
    except Exception as e:
        print(f"❌ Failed to test ImageNet100: {e}")
        import traceback
        traceback.print_exc()
    
    return tester


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SSLib datasets")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="data_new",  # Changed from "data" to "data_new"
        help="Root directory for datasets (default: data_new)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip tests that require downloading (only test if data exists)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "synthetic", "celeba", "imagenet100"],
        default="all",
        help="Which dataset(s) to test (default: all)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SSLib Dataset Testing Suite")
    print("="*60)
    print(f"Root directory: {args.root_dir}")
    print(f"Absolute path: {Path(args.root_dir).absolute()}")
    print(f"Skip download: {args.skip_download}")
    print(f"Testing: {args.dataset}")
    
    all_testers = []
    
    # Test synthetic dataset (always runs, no download needed)
    if args.dataset in ["all", "synthetic"]:
        tester = test_synthetic_dataset()
        all_testers.append(tester)
    
    # Test CelebA
    if args.dataset in ["all", "celeba"]:
        if not args.skip_download:
            print(f"\n⚠️  Note: CelebA will be downloaded to {Path(args.root_dir).absolute() / 'CelebA'}")
            print("   This is approximately 1.4GB")
        tester = test_celeba_dataset(args.root_dir, args.skip_download)
        all_testers.append(tester)
    
    # Test ImageNet100
    if args.dataset in ["all", "imagenet100"]:
        if not args.skip_download:
            print(f"\n⚠️  Note: ImageNet100 will be downloaded to {Path(args.root_dir).absolute() / 'ImageNet100'}")
            print("   This is approximately 5GB")
        tester = test_imagenet100_dataset(args.root_dir, args.skip_download)
        all_testers.append(tester)
    
    # Print combined summary
    print("\n" + "="*60)
    print("COMBINED TEST SUMMARY")
    print("="*60)
    
    total_passed = sum(t.passed for t in all_testers)
    total_failed = sum(t.failed for t in all_testers)
    
    print(f"Total: {total_passed} passed, {total_failed} failed")
    
    # Print all errors
    all_errors = []
    for tester in all_testers:
        all_errors.extend(tester.errors)
    
    if all_errors:
        print("\nAll failed tests:")
        for error in all_errors:
            print(f"  - {error}")
    else:
        print("\n✅ All tests passed!")
    
    print("="*60)
    
    # Exit with error code if any tests failed
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()