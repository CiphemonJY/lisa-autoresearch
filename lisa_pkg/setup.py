from setuptools import setup, find_packages

setup(
    name="lisa-pkg",
    version="1.0.0",
    description="LISA: Train 32B-120B models on limited RAM",
    author="LISA Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "psutil>=5.9",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "lisa-train=lisa_pkg.scripts.train:main",
            "lisa-infer=lisa_pkg.scripts.infer:main",
        ],
    },
)
