import numpy as np
import torch
from ssrlib import Pipeline, Config
from ssrlib.datasets import SynthTestDataset
from ssrlib.embedders.cv import DINOv2Embedder, CLIPEmbedder
from ssrlib.processing import CovarianceProcessor, ZCAProcessor


def basic_single_pipeline():
    """Basic pipeline with single dataset and embedder."""

    print("=== Basic Single Pipeline ===")

    # Create a simple pipeline
    pipeline = Pipeline(
        [
            ("dataset", SynthTestDataset(tensors_num=50, seed=42)),
            ("embedder", DINOv2Embedder("dinov2_vitb14")),
            ("processor", CovarianceProcessor()),
        ]
    )

    # Execute with custom configuration
    config = Config({"batch_size": 32, "device": "cpu"})
    results = pipeline.execute(config_override={"batch_size": 16})

    # Access results
    dataset_name = "SynthTest"
    embedder_name = "DINOv2_dinov2_vitb14"
    processor_name = "Covariance"

    embeddings = results.get_embeddings(dataset_name, embedder_name)
    covariance = results.get_processed(dataset_name, embedder_name, processor_name)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Covariance matrix shape: {covariance.shape}")
    print(f"Pipeline timing: {results.timing}")


def multi_component_pipeline():
    """Pipeline with multiple datasets, embedders, and processors."""

    print("\n=== Multi-Component Pipeline ===")

    # Create pipeline with multiple components
    pipeline = Pipeline(
        [
            (
                "datasets",
                [
                    SynthTestDataset(tensors_num=20, tensor_shape=(3, 224, 224), seed=1),
                    SynthTestDataset(tensors_num=20, tensor_shape=(3, 224, 224), seed=2),
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

    # This creates 2×2×2 = 8 different combinations
    results = pipeline.execute()

    print(f"Total embeddings computed: {len(results.embeddings)}")
    print(f"Total processed outputs: {len(results.processed)}")

    # Show all combinations
    for (dataset, embedder), emb in results.embeddings.items():
        print(f"{dataset} + {embedder}: {emb.shape}")

    for (dataset, embedder, processor), proc in results.processed.items():
        print(f"{dataset} + {embedder} + {processor}: {proc.shape}")


def configuration_driven_pipeline():
    """Example using configuration files and overrides."""

    print("\n=== Configuration-Driven Pipeline ===")

    # Create configuration
    config = Config(
        {
            "device": "cpu",
            "batch_size": 64,
            "output_dir": "./results",
            "model": {
                "dinov2_variant": "dinov2_vitb14",
                "clip_variant": "clip-vit-large-patch14",
            },
            "processing": {"zca_epsilon": 1e-9, "compute_covariance": True},
        }
    )

    # Create pipeline using config values
    embedders = []
    if config.get("model.dinov2_variant"):
        embedders.append(DINOv2Embedder(config.get("model.dinov2_variant")))
    if config.get("model.clip_variant"):
        embedders.append(CLIPEmbedder(config.get("model.clip_variant")))

    processors = []
    if config.get("processing.compute_covariance"):
        processors.append(CovarianceProcessor())
    if config.get("processing.zca_epsilon"):
        processors.append(ZCAProcessor(epsilon=config.get("processing.zca_epsilon")))

    pipeline = Pipeline(
        [
            (
                "dataset",
                SynthTestDataset(tensors_num=30, tensor_shape=(3, 224, 224), seed=123),
            ),
            ("embedders", embedders),
            ("processors", processors),
        ],
        config=config,
    )

    # Execute with runtime overrides
    results = pipeline.execute(config_override={"batch_size": 32})

    print(f"Used batch size: {config.get('batch_size')}")
    print(f"Results metadata keys: {list(results.metadata.keys())}")


def analyze_results():
    """Example of analyzing pipeline results."""

    print("\n=== Analyzing Results ===")

    # Simple pipeline for analysis
    pipeline = Pipeline(
        [
            (
                "dataset",
                SynthTestDataset(tensors_num=30, tensor_shape=(3, 224, 224), seed=123),
            ),
            ("embedder", DINOv2Embedder("dinov2_vits14")),  # Smaller model
            ("processors", [CovarianceProcessor(), ZCAProcessor()]),
        ]
    )

    results = pipeline.execute()

    # Analyze embeddings
    dataset_name = "SynthTest"
    embedder_name = "DINOv2_dinov2_vits14"

    embeddings = results.get_embeddings(dataset_name, embedder_name)
    covariance = results.get_processed(dataset_name, embedder_name, "Covariance")
    whitened = results.get_processed(dataset_name, embedder_name, "ZCA")

    print(f"\nEmbedding Analysis:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {np.mean(embeddings):.4f}")
    print(f"  Std: {np.std(embeddings):.4f}")
    print(f"  Min: {np.min(embeddings):.4f}")
    print(f"  Max: {np.max(embeddings):.4f}")

    print(f"\nCovariance Matrix Analysis:")
    print(f"  Shape: {covariance.shape}")
    print(f"  Trace: {np.trace(covariance):.4f}")
    print(f"  Determinant: {np.linalg.det(covariance):.4e}")
    print(f"  Max eigenvalue: {np.max(np.linalg.eigvals(covariance)):.4f}")
    print(f"  Min eigenvalue: {np.min(np.linalg.eigvals(covariance)):.4f}")

    print(f"\nWhitened Embeddings Analysis:")
    print(f"  Shape: {whitened.shape}")
    print(f"  Mean: {np.mean(whitened):.4f}")
    print(f"  Std: {np.std(whitened):.4f}")

    # Check if whitening worked (covariance should be close to identity)
    whitened_cov = np.cov(whitened.T)
    identity_error = np.mean((whitened_cov - np.eye(whitened_cov.shape[0])) ** 2)
    print(f"  Whitening quality (MSE from identity): {identity_error:.6f}")


def main():
    """Run all examples."""

    print("SSLib Framework - Usage Examples")
    print("================================")

    try:
        basic_single_pipeline()
        multi_component_pipeline()
        configuration_driven_pipeline()
        analyze_results()

        print("\n=== All Examples Completed Successfully! ===")

    except Exception as e:
        print(f"\nExample failed with error: {str(e)}")
        print("Note: Make sure you have the required datasets downloaded.")
        print("For testing without real data, see test_examples.py")


if __name__ == "__main__":
    main()
