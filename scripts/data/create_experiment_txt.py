from pathlib import Path
import pandas as pd
from glob import iglob
import re
from tqdm.auto import tqdm

grid_save_path = Path("/nfs/scratch2/shared_image_recog_ml/save_grid_dir2")
save_cs_paths = "valid_paths.txt"
cs_paths = []
data_path = Path("/nfs/datasync4/tklab-llsm")
PATTERN = re.compile("(?i)(dextran|lamp1|npc1|eea1|transferrin)")

# for cs_path in tqdm(iglob(str(data_path.joinpath("**", "CS*")), recursive=True)):
#     cs_path = Path(cs_path)
#     if cs_path.is_dir() and PATTERN.search(str(cs_path)):
#         cs_paths.append(str(cs_path))


# with open(save_cs_paths, "a") as f:
#     for cs_path in tqdm(cs_paths):
#         f.write(cs_path + "\n")

cs_paths = []

with open(save_cs_paths, "r") as f:
    for cs_path in f:
        cs_paths.append(Path(cs_path.strip()))

channel_paths = []

for cs_path in tqdm(cs_paths):
    for experiment_path in cs_path.iterdir():
        if experiment_path.is_dir() and experiment_path.name.startswith("Ex"):
            for channel_dir in experiment_path.iterdir():
                if channel_dir.is_dir() and "ch" in channel_dir.name:
                    channel_paths.append(str(channel_dir))

with open("channel_paths_valid.txt", "w") as f:
    for channel_path in channel_paths:
        f.write(str(channel_path) + "\n")

# cs_paths = {}

# with open(save_cs_paths, "r") as f:
#     for cs_path in f:
#         cs_path = Path(cs_path.strip())
#         cs_paths[cs_path.name] = cs_path

# for grid in grid_save_path.iterdir():
#     if grid.is_dir():
#         metadata_path = grid.joinpath("metadata.csv")
#         metadata = pd.read_csv(metadata_path)
#         correct_indices_file = next(grid.glob("correct_grid*.txt"))
#         correct_indices = [
#             int(i) for i in correct_indices_file.read_text().splitlines()
#         ]
#         metadata = metadata.loc[metadata["cell_id"].isin(correct_indices)]
#         metadata["cs_path"] = metadata["cs_path"].map(cs_paths)
#         metadata.to_csv(grid.joinpath("metadata_filtered.csv"), index=False)

# experiment_paths = []
# for grid in grid_save_path.iterdir():
#     if grid.is_dir():
#         metadata_path = grid.joinpath("metadata_filtered.csv")
#         metadata = pd.read_csv(metadata_path)
#         for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
#             for experiment_path in Path(row["cs_path"]).iterdir():
#                 if experiment_path.is_dir():
#                     experiment_paths.append(
#                         experiment_path.joinpath(row["channel_path"])
#                     )

# with open("experiment_paths.txt", "w") as f:
#     for experiment_path in experiment_paths:
#         f.write(str(experiment_path) + "\n")

# experiment_paths = []
# with open("experiment_paths.txt", "r") as f:
#     for line in f:
#         experiment_path = Path(line.strip())
#         if experiment_path.is_dir():
#             experiment_paths.append(experiment_path)

# with open("experiment_paths.txt", "w") as f:
#     for experiment_path in experiment_paths:
#         f.write(str(experiment_path) + "\n")

# # sanity check
# with open("experiment_paths.txt", "r") as f:
#     for line in f:
#         experiment_path = Path(line.strip())
#         assert experiment_path.is_dir()
