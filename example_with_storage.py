from ssllib.datasets import SynthTestDataset
from ssllib.embedders.cv import DINOv2Embedder, CLIPEmbedder
from ssllib.processing import CovarianceProcessor, ZCAProcessor


from ssllib import Pipeline


def example_with_storage():
    """Example showing pipeline with storage caching."""

    # Create pipeline
    pipeline = Pipeline(
        [
            (
                "datasets",
                [
                    SynthTestDataset(tensors_num=50, tensor_shape=(3, 224, 224), seed=1),
                    SynthTestDataset(tensors_num=30, tensor_shape=(3, 224, 224), seed=2),
                ],
            ),
            (
                "embedders",
                [
                    DINOv2Embedder("dinov2_vitb14"),
                    CLIPEmbedder("clip-vit-large-patch14"),
                ],
            ),
            ("processors", [CovarianceProcessor(), ZCAProcessor(epsilon=1e-6)]),
        ]
    )

    print("=== First execution (cache miss) ===")
    # First execution - will compute and cache all embeddings
    results1 = pipeline.execute(
        use_storage=True,
        storage_dir="./cache/pipeline_test",
        storage_description="Test pipeline with two synthetic datasets",
    )

    print(f"Cache hit rate: {results1.metadata.get('cache_hit_rate', 0):.2%}")
    print(f"Storage info: {results1.storage_info}")

    print("\n=== Second execution (cache hit) ===")
    # Second execution - should load from cache
    results2 = pipeline.execute(
        use_storage=True,
        storage_dir="./cache/pipeline_test",
        force_recompute=False,  # Use cache
    )

    print(f"Cache hit rate: {results2.metadata.get('cache_hit_rate', 0):.2%}")
    print(f"Timing comparison:")
    print(f"  First run: {results1.timing['total_time']:.2f}s")
    print(f"  Second run: {results2.timing['total_time']:.2f}s")
    print(f"  Speedup: {results1.timing['total_time']/results2.timing['total_time']:.1f}x")

    print("\n=== Force recompute ===")
    # Third execution - force recompute
    results3 = pipeline.execute(
        use_storage=True,
        storage_dir="./cache/pipeline_test",
        force_recompute=True,  # Ignore cache
    )

    print(f"Cache hit rate: {results3.metadata.get('cache_hit_rate', 0):.2%}")


if __name__ == "__main__":
    example_with_storage()
