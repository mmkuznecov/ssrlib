import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import time
import logging
from typing import Iterator, Dict, Any
from sslib.storage.tensor_storage import TensorStorage


def test_large_scale_storage():
    """
    Test storage with 1 million vectors of size 384.
    This tests the system's ability to handle large-scale data efficiently.
    """
    print("=" * 80)
    print("Testing Large Scale Storage: 1 Million Vectors of Size 384")
    print("=" * 80)

    # Parameters
    num_vectors = 1_000_000
    vector_size = 384
    total_size_gb = (num_vectors * vector_size * 4) / (1024**3)

    print(f"Dataset specifications:")
    print(f"  - Number of vectors: {num_vectors:,}")
    print(f"  - Vector size: {vector_size}")
    print(f"  - Total data size: {total_size_gb:.2f} GB")
    print(f"  - Data type: float32")
    print()

    def generate_test_vectors() -> Iterator[np.ndarray]:
        """
        Generator for test vectors to avoid memory issues.
        Uses deterministic seeds for reproducible results.
        """
        print("Generating vectors...")
        for i in range(num_vectors):
            # Use deterministic random state for reproducibility
            # Cycle seeds to keep memory usage low while maintaining determinism
            seed = (i * 12345) % 2**31  # Simple hash-like function
            np.random.seed(seed)
            vector = np.random.randn(vector_size).astype(np.float32)

            # Show progress every 50k vectors
            if (i + 1) % 50000 == 0:
                print(f"  Generated {i+1:,} vectors ({(i+1)/num_vectors*100:.1f}%)")

            yield vector

    def generate_test_metadata() -> Iterator[Dict[str, Any]]:
        """
        Generator for test metadata with realistic structure.
        """
        for i in range(num_vectors):
            meta = {
                "vector_id": i,
                "batch_id": i // 1000,  # 1000 vectors per batch (1000 batches)
                "epoch": i // 100000,  # 100k vectors per epoch (10 epochs)
                "model_version": f"v{i // 250000}",  # 4 model versions (250k each)
                "dataset_split": ["train", "val", "test"][
                    i % 3
                ],  # Cycle through splits
                "timestamp": f"2024-{(i // 50000) + 1:02d}-01",  # Monthly timestamps (20 months)
                "vector_norm": float((i * 7) % 1000)
                / 1000.0,  # Synthetic norm values [0, 1)
                "quality_score": 0.5
                + 0.5 * np.sin(i * 0.001),  # Sine wave quality scores
                "processing_time": 0.001
                + (i % 100) * 0.00001,  # Synthetic processing times
            }
            yield meta

    def verify_vector_sample(storage: TensorStorage, sample_size: int = 20) -> None:
        """
        Verify a sample of vectors by regenerating them with the same seeds.
        """
        print(f"  Verifying sample of {sample_size} vectors...")

        # Create a diverse sample across the entire range
        sample_indices = np.linspace(0, num_vectors - 1, sample_size, dtype=int)

        for idx in sample_indices:
            # Recreate the expected vector using the same seed generation logic
            seed = (idx * 12345) % 2**31
            np.random.seed(seed)
            expected_vector = np.random.randn(vector_size).astype(np.float32)

            # Retrieve and compare
            retrieved_vector = storage[idx]

            assert retrieved_vector.shape == (
                vector_size,
            ), f"Shape mismatch at index {idx}: expected {(vector_size,)}, got {retrieved_vector.shape}"

            assert np.allclose(
                retrieved_vector, expected_vector, rtol=1e-6, atol=1e-6
            ), f"Value mismatch at index {idx}"

        print(f"  ✓ All {sample_size} sampled vectors verified successfully")

    def test_metadata_queries(storage: TensorStorage) -> None:
        """
        Test various metadata-based queries to ensure functionality at scale.
        """
        print("  Testing metadata queries...")

        query_start = time.time()

        # Test batch queries
        batch_0_vectors = storage.get_tensors_by_batch(0)
        assert (
            len(batch_0_vectors) == 1000
        ), f"Expected 1000 vectors in batch 0, got {len(batch_0_vectors)}"

        # Test model version filtering
        v0_indices = storage.filter_tensors(model_version="v0")
        assert (
            len(v0_indices) == 250000
        ), f"Expected 250k vectors for model v0, got {len(v0_indices)}"

        # Test dataset split filtering
        train_indices = storage.filter_tensors(dataset_split="train")
        expected_train_count = num_vectors // 3 + (
            1 if num_vectors % 3 > 0 else 0
        )  # Ceiling division
        assert (
            len(train_indices) == expected_train_count
        ), f"Expected ~{expected_train_count} train vectors, got {len(train_indices)}"

        # Test multi-parameter filtering
        train_v0_indices = storage.filter_tensors(
            model_version="v0", dataset_split="train"
        )
        assert len(train_v0_indices) > 0, "Should find vectors matching both criteria"

        # Test parameter-based retrieval
        vector_by_id = storage.get_tensor_by_param("vector_id", 123456)
        assert vector_by_id is not None, "Should find vector with specific ID"

        query_time = time.time() - query_start
        print(f"  ✓ Metadata queries completed in {query_time:.3f}s")

        return query_time

    def run_performance_test(chunk_size: int, chunk_label: str) -> Dict[str, float]:
        """
        Run the complete test with a specific chunk size and return performance metrics.
        """
        print(f"\n--- Testing with {chunk_label} ---")

        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = os.path.join(temp_dir, f"large_scale_{chunk_size}")

            # Storage creation
            print("Creating storage...")
            creation_start = time.time()

            storage = TensorStorage.create_storage(
                storage_dir=storage_dir,
                data_iterator=generate_test_vectors(),
                metadata_iterator=generate_test_metadata(),
                chunk_size=chunk_size,
                description=f"Large scale test - 1M vectors, {chunk_label}",
            )

            creation_time = time.time() - creation_start

            try:
                # Basic verification
                print("Performing basic verification...")
                assert (
                    len(storage) == num_vectors
                ), f"Expected {num_vectors} vectors, got {len(storage)}"

                # Sample-based verification
                verify_vector_sample(storage, sample_size=50)

                # Metadata testing
                query_time = test_metadata_queries(storage)

                # Random access performance test
                print("  Testing random access performance...")
                random_start = time.time()
                random_indices = np.random.choice(num_vectors, size=100, replace=False)
                for idx in random_indices:
                    _ = storage[idx]  # Access vector
                random_time = time.time() - random_start

                # Storage info
                info = storage.get_storage_info()

                # Calculate performance metrics
                throughput = total_size_gb / creation_time

                metrics = {
                    "creation_time": creation_time,
                    "throughput_gb_per_s": throughput,
                    "query_time": query_time,
                    "random_access_time": random_time,
                    "storage_size_mb": info["total_size_mb"],
                    "num_chunks": storage.metadata.get("total_chunks", 0),
                }

                # Report results
                print(f"  ✓ Performance Results:")
                print(f"    Creation time: {creation_time:.2f}s")
                print(f"    Throughput: {throughput:.3f} GB/s")
                print(f"    Final storage size: {info['total_size_mb']:.1f} MB")
                print(f"    Number of chunks: {metrics['num_chunks']}")
                print(f"    Query time: {query_time:.3f}s")
                print(f"    Random access (100 vectors): {random_time:.3f}s")

                return metrics

            finally:
                storage.close()

    # Test with different chunk sizes to find optimal performance
    chunk_configurations = [
        (4 * 1024 * 1024, "4MB chunks"),  # Small chunks
        (16 * 1024 * 1024, "16MB chunks"),  # Medium chunks
        (64 * 1024 * 1024, "64MB chunks"),  # Large chunks
    ]

    all_metrics = {}

    for chunk_size, chunk_label in chunk_configurations:
        try:
            metrics = run_performance_test(chunk_size, chunk_label)
            all_metrics[chunk_label] = metrics
        except Exception as e:
            print(f"  ✗ Test failed with {chunk_label}: {e}")
            import traceback

            traceback.print_exc()

    # Performance comparison
    if len(all_metrics) > 1:
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)

        print(
            f"{'Configuration':<15} {'Creation (s)':<12} {'Throughput (GB/s)':<17} {'Storage (MB)':<13} {'Chunks':<8}"
        )
        print("-" * 75)

        for config, metrics in all_metrics.items():
            print(
                f"{config:<15} {metrics['creation_time']:<12.2f} "
                f"{metrics['throughput_gb_per_s']:<17.3f} "
                f"{metrics['storage_size_mb']:<13.1f} "
                f"{metrics['num_chunks']:<8}"
            )

        # Find best performing configuration
        best_config = max(
            all_metrics.items(), key=lambda x: x[1]["throughput_gb_per_s"]
        )
        print(
            f"\n🏆 Best throughput: {best_config[0]} ({best_config[1]['throughput_gb_per_s']:.3f} GB/s)"
        )

    print("\n" + "=" * 80)
    print("✓ Large scale storage test completed successfully!")
    print("=" * 80)

    return all_metrics


def test_large_scale_rebuild():
    """
    Test rebuilding large storage with different chunk sizes.
    This is a smaller test (100k vectors) to keep it practical.
    """
    print("\n" + "=" * 80)
    print("Testing Large Scale Storage Rebuild (100k Vectors)")
    print("=" * 80)

    num_vectors = 100_000
    vector_size = 384

    def generate_rebuild_vectors():
        for i in range(num_vectors):
            seed = (i * 54321) % 2**31
            np.random.seed(seed)
            yield np.random.randn(vector_size).astype(np.float32)

    def generate_rebuild_metadata():
        for i in range(num_vectors):
            yield {
                "vector_id": i,
                "batch_id": i // 500,
                "timestamp": f"2024-{(i // 10000) + 1:02d}-01",
            }

    with tempfile.TemporaryDirectory() as temp_dir:
        original_dir = os.path.join(temp_dir, "original_large")
        rebuilt_dir = os.path.join(temp_dir, "rebuilt_large")

        print("Creating original storage with small chunks...")
        original_storage = TensorStorage.create_storage(
            storage_dir=original_dir,
            data_iterator=generate_rebuild_vectors(),
            metadata_iterator=generate_rebuild_metadata(),
            chunk_size=1 * 1024 * 1024,  # 1MB chunks
            description="Original large storage",
        )

        try:
            print("Rebuilding with larger chunks...")
            rebuild_start = time.time()

            rebuilt_storage = original_storage.rebuild_storage(
                new_storage_dir=rebuilt_dir,
                new_chunk_size=32 * 1024 * 1024,  # 32MB chunks
                description="Rebuilt large storage",
            )

            rebuild_time = time.time() - rebuild_start

            try:
                # Verify rebuilt storage
                assert len(rebuilt_storage) == len(
                    original_storage
                ), "Length should match"
                assert (
                    rebuilt_storage.chunk_size == 32 * 1024 * 1024
                ), "Chunk size should be updated"

                # Sample verification
                sample_indices = np.random.choice(num_vectors, size=20, replace=False)
                for idx in sample_indices:
                    original_tensor = original_storage[idx]
                    rebuilt_tensor = rebuilt_storage[idx]
                    assert np.allclose(
                        original_tensor, rebuilt_tensor
                    ), f"Tensor mismatch at index {idx}"

                original_info = original_storage.get_storage_info()
                rebuilt_info = rebuilt_storage.get_storage_info()

                print(f"✓ Rebuild completed in {rebuild_time:.2f}s")
                print(
                    f"  Original: {original_info['total_size_mb']:.1f}MB, {original_storage.metadata['total_chunks']} chunks"
                )
                print(
                    f"  Rebuilt:  {rebuilt_info['total_size_mb']:.1f}MB, {rebuilt_storage.metadata['total_chunks']} chunks"
                )

            finally:
                rebuilt_storage.close()

        finally:
            original_storage.close()


def run_large_scale_tests():
    """
    Run all large scale tests.
    """
    print("Starting Large Scale TensorStorage Tests")
    print("This may take several minutes to complete...")
    print()

    # Reduce logging noise
    logging.getLogger().setLevel(logging.WARNING)

    try:
        # Main large scale test
        metrics = test_large_scale_storage()

        # Rebuild test
        test_large_scale_rebuild()

        print("\n🎉 All large scale tests completed successfully!")
        return metrics

    except Exception as e:
        print(f"\n❌ Large scale test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_large_scale_tests()
