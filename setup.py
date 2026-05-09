from setuptools import setup, find_packages

setup(
    name="jampr_vrptw",
    version="1.0.0",
    description="JAMPR - Joint Attention Model for Parallel Route-Construction (VRPTW)",
    author="JAMPR Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tensorboard>=2.13.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
