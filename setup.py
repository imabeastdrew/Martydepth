from setuptools import setup, find_packages

setup(
    name="martydepth",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "numpy",
        "wandb",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "pathlib>=1.0.1",
    ],
    python_requires=">=3.8",
)
