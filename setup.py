from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sslib",
    version="0.1.0",
    author="Mikhail Kuznetov",
    author_email="mmkuznecov2002@gmail.com",
    description="A minimal Python framework for Self-Supervised Learning (SSL) focused on embedding extraction and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmkuznecov/sslib",
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
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "Pillow>=8.3.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "transformers>=4.20.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
        "examples": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
        ],
    },
    # project_urls={
    #     "Bug Reports": "https://github.com/sslib/sslib/issues",
    #     "Source": "https://github.com/sslib/sslib",
    #     "Documentation": "https://sslib.readthedocs.io/",
    # },
)
