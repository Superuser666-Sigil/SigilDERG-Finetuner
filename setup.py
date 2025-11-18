"""
Setup script for SigilDERG-Finetuner.

This is a fallback for systems that don't support pyproject.toml.
For modern Python, use: pip install -e .
"""

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="sigilderg-finetuner",
    version="2.5.0",
    description="Model finetuner for the SigilDERG Ecosystem using QLoRA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Dave Tofflemire",
    license="MIT",
    python_requires=">=3.12.10",
    packages=["rust_qlora"],
    package_dir={"rust_qlora": "rust-qlora"},
    package_data={"rust_qlora": ["configs/*.yml"]},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sigilderg-train=rust_qlora.train:main",
            "sigilderg-eval=rust_qlora.eval_rust:main",
            "sigilderg-sweep=rust_qlora.hyperparameter_sweep:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

