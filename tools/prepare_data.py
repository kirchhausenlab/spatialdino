import tifffile as tif
from pathlib import Path
from argparse import ArgumentParser
from typing import List

# - **make new data folder**
#     - **flatten data (avg or max on z)**
# - **take several tif files and stack them on the time axis**
# - shape of data (3, t, z, y, x)
#     - 3 corresponds to number of color channels
#     - t is each file or frame which has a shape of z,y,x
#     - mask as necessary and normalize across t axis
#     - filter by outliers (say 3 outliers away)
# - normalize movies over frames for each channels
#     - take all movies part of 488 nm color channel and stack them (normalize through time)
#         - leads to consistent intensity


# data for lattice microscopes is located in: /nfs/datasync4/tklab-llsm/
# data for adaptive optics is located in: /nfs/datasync4/AO-LLSM/


def directory_walk(
    parent_directory: Path, target_folder: str, max_depth: int
) -> List[Path]:
    if max_depth == 0 or parent_directory.is_file():
        return []
    elif parent_directory.name == target_folder:
        return [parent_directory]
    return [
        path
        for child in parent_directory.iterdir()
        for path in directory_walk(child, target_folder, max_depth - 1)
    ]


def extract_target_directories(
    parent_directory: Path, target_folder: str, max_directories: int, max_depth: int
) -> List[Path]:
    target_directories = []
    while max_directories > 0:
        target_directories.extend(
            directory_walk(next(parent_directory.iterdir()), target_folder, max_depth)
        )
        max_directories -= 1
    return target_directories


def extract_tif_files(directory: Path) -> List[Path]:
    return list(
        filter(
            lambda file: file.suffix == ".tif", (file for file in directory.iterdir())
        )
    )


def filter_tif_files(
    target_directories: List[Path], min_files: int
) -> List[List[Path]]:
    all_files = []
    for directory in target_directories:
        files = extract_tif_files(directory)
        if len(files) > min_files:
            all_files.append(files)
    return all_files


def filter_target_directories(
    target_directories: List[Path], min_directories: int
) -> List[Path]:
    return [
        directory
        for directory in target_directories
        if len(extract_tif_files(directory)) > min_directories
    ]


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Prepare data lattice microscopy data into .h5 format."
    )
    parser.add_argument(
        "parent_directory", type=Path, help="Parent directory of the data."
    )
    parser.add_argument("--target_folder", type=str, help="Name of the target folder.")
    parser.add_argument(
        "--max_directories",
        type=int,
        default=1,
        help="Max number of directories to search in for target folder.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=6,
        help="Max depth to search for target directory.",
    )
    args = parser.parse_args()
    target_directories = extract_target_directories(
        args.parent_directory, args.target_folder, args.max_directories, args.max_depth
    )
    for t in target_directories:
        print(t)

    filtered_tif_files = filter_tif_files(target_directories, 10)
    print(filtered_tif_files)
