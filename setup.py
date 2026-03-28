from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ssrlib",
    version="0.1.0",
    author="Mikhail Kuznetov",
    author_email="mmkuznecov2002@gmail.com",
    description="A modular Python framework for Self-Supervised Learning with automatic component discovery and intelligent caching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmkuznecov/ssrlib",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        # Image processing
        "Pillow>=8.3.0",
        # Model loading
        "transformers>=4.20.0",
        "huggingface-hub>=0.16.0",
        # Utilities
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "requests>=2.28.0",
        # Data handling
        "safetensors>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "isort>=5.10",
            "pylint>=2.15",
        ],
        "examples": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "all": [
            # Optional advanced features
            "scikit-learn>=1.0.0",
            "sentencepiece>=0.1.96",  # For some NLP embedders
        ],
    },
)
