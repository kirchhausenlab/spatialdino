"""Filter saved experiments by fluorophore name and write a new path list.

Reads the ``data_paths_3D.txt`` index produced by the save pipeline,
resolves each entry back to its experiment root, loads its metadata, and
checks whether the original acquisition path matches any of a predefined
set of fluorophore names (dextran, lamp1, npc1, eea1, transferrin).
Matching experiment paths are written to a new ``data_paths_3D.txt`` in
the target save directory.

Usage::

    python select_experiments.py

"""

import re
from pathlib import Path

import torch
from tqdm.auto import tqdm

pattern = re.compile(r"(?i)(dextran|lamp1|npc1|eea1|transferrin)")

data_dir = Path("/nfs/scratch2/shared_image_recog_ml/llsm_3d_ds")
save_dir = Path("/raid1/shared_image_recog_ml/llsm_3d_ds")

experiment_names = data_dir.joinpath("data_paths_3D.txt").read_text().splitlines()

unique_experiments = set()
for exp in tqdm(experiment_names):
    experiment_dir = data_dir.joinpath(exp)
    unique_experiments.add(experiment_dir.parents[2])

with open(save_dir.joinpath("data_paths_3D.txt"), "w") as f:
    for channel_dir in tqdm(unique_experiments):
        metadata_file = channel_dir.joinpath("3D", "frame_0", "part_0", "metadata.pth")
        metadata = torch.load(metadata_file, weights_only=False)
        if re.search(pattern, metadata["path"]):
            f.write(f"{channel_dir}\n")
