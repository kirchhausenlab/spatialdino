from setuptools import setup, find_packages
from pathlib import Path

WORKING_DIR = Path(__file__).parent

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
        "torch<=2.5.0",
        "torchvision<=0.20.0",
        "loguru>=0.7.2",
        "numpy>=1.26.3",
        "pandas>=2.2.2",
        "matplotlib>=3.9.2",
        "pillow>=10.2.0",
        "tqdm>=4.64.1",
        "ipython>=8.26.0",
        "loky>=3.4.1",
        "tifffile",
        "scikit-image",
        "imagecodecs",
        "wandb>=0.18.1",
        "hydra-core>=1.3.2",
        "scipy",
        "scikit-learn",
        "ninja",
        "jupyter",
        "ipywidgets",
        "",
    ],
    extras_require={
        "xformers": ["xformers", "triton"],
        "viz3d-cpu": [
            "pyvista>=0.44.1",
            "vtk_osmesa",
            "trame",
            "trame-vuetify",
            "trame-vtk",
        ],
        "dev": ["pytest>=8.3.2", "ruff>=0.6.2"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
