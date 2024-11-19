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
        "torch==2.5.0",
        "torchvision==0.20.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "matplotlib==3.9.2",
        "pillow==11.0.0",
        "tqdm==4.66.5",
        "loky==3.4.1",
        "tifffile==2024.9.20",
        "scikit-image==0.24.0",
        "imagecodecs==2024.9.22",
        "wandb==0.18.5",
        "hydra-core==1.3.2",
        "scipy==1.14.1",
        "scikit-learn==1.5.2",
        "ninja==1.11.1.1",
        "jupyter==1.1.1",
        "ipython==8.28.0",
        "ipywidgets==8.1.5",
    ],
    extras_require={
        "xformers": ["xformers", "triton"],
        "viz3d-cpu": [
            "pyvista==0.44.1",
            "vtk_osmesa==9.3.1",
            "trame==3.7.0",
            "trame-vuetify==2.7.1",
            "trame-vtk==2.8.11",
            "imageio[ffmpeg]==2.36.0",
            "pygifsicle==1.1.0",
        ],
        "dev": ["pytest", "ruff"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
