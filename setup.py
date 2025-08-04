from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="revnet-zero",
    version="1.0.0",
    author="RevNet-Zero Team",
    author_email="team@revnet-zero.org",
    description="Memory-efficient reversible transformers for long-context training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/revnet-zero/revnet-zero",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "einops>=0.6.0",
        "packaging",
        "matplotlib>=3.5.0",
        "tqdm>=4.60.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "triton": [
            "triton>=2.0.0",
        ],
        "jax": [
            "jax[cuda]>=0.4.0",
            "flax>=0.6.0",
        ],
        "full": [
            "triton>=2.0.0",
            "jax[cuda]>=0.4.0",
            "flax>=0.6.0",
            "transformers>=4.20.0",
            "datasets>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "revnet-benchmark=revnet_zero.cli.benchmark:main",
            "revnet-convert=revnet_zero.cli.convert:main",
        ],
    },
)