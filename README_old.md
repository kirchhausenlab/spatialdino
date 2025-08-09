# Cell Interactome

Automated detection of events of interest in cell interactions. Please refer to the wiki for more details.

- In the wiki you have an experiments folder which includes tiff files and results of segmentations of different biological objects. The experiments.md file explains the experiments used.
- Spatial_dino.pdf has a more technical view of our model, preprocessing pipeline, inference, segmentation and tracking.
- The run_spatial_dino folder has 4 videos showing how to run inference, segmentation and tracking.
- Supplementary folder has interesting code and ideas that were never used but can be incorporated in the future. They have ideas like feaure pyramids, DesD heads, 3D transformations in pytorch etc.
- The technical summary fodler has a more technical view of the model, preprocessing pipeline, inference, segmentation and tracking.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment (pip), e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── environment.yml   <- The requirements file for reproducing the analysis environment (conda).
    │
    ├── pyproject.toml           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

---
