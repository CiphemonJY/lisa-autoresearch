"""
LISA-AutoResearch - Setup Configuration
"""

from setuptools import setup, find_packages

setup(
    name="lisa-autoresearch",
    version="1.0.0",
    author="Ciphemon",
    description="Large Model Training on Consumer Hardware using LISA + Disk Offload",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CiphemonJY/lisa-autoresearch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Machine Learning",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lisa-train=lisa_autoresearch.main:main",
        ],
    },
)