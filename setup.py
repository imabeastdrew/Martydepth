from setuptools import setup, find_packages

setup(
    name="martydepth",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "wandb>=0.15.12",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
)
