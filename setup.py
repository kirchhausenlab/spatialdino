from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cell_interactome",
    version="0.0.1",
    description="A foundation model for molecular biology.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "loguru>=0.7.2",
        "numpy>=1.26.3",
        "pandas>=2.2.2",
        "matplotlib>=3.9.2",
        "h5py>=3.11.0",
        "pillow>=10.2.0",
        "tqdm>=4.64.1",
        "ipython>=8.26.0",
        "loky>=3.4.1",
        "tifffile>=2024.8.28",
        "imagecodecs>=2024.6.1",
        "lightning>=2.4.0",
        "wandb>=0.18.1",
        "hydra-core>=1.3.2",
        "fvcore>=0.1.5.post20221221",
        "submitit>=1.5.2",
    ],
    extras_require={
        "xformers": ["xformers>=0.0.27.post2", "triton>=3.0.0"],
        "dev": ["pytest>=8.3.2", "ruff>=0.6.2"],
    },
    dependency_links=[
        "https://download.pytorch.org/whl/cu121",
        "https://pypi.nvidia.com",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
