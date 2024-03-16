from collections import defaultdict
from concurrent.futures import as_completed
from functools import partial
import tifffile as tif
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from loky import get_reusable_executor
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm
import torch
from uuid import uuid4
import h5py

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


@dataclass
class Experiment:
    tif_files: List[str]
    parent_directory: Path


@dataclass
class DataExtractor:
    max_search_depth: int
    found_directories_limit: int
    min_tif_files_in_folder: int
    timeout: int
    max_workers: int = 4

    def extract_experiments(
        self, parent_directory: Path, target_directory: str
    ) -> List[Experiment]:
        target_directories = DataExtractor.directory_walk(
            parent_directory,
            target_directory,
            self.max_search_depth,
            self.found_directories_limit,
        )

        experiments = DataExtractor._extract_experiments(
            target_directories, self.min_tif_files_in_folder
        )

        experiments = [
            Experiment(tif_files, parent_directory)
            for parent_directory, tif_files in experiments.items()
        ]

        return experiments

    @staticmethod
    def directory_walk(
        parent_directory: Path,
        target_folder: str,
        max_search_depth: int,
        found_directories_limit: int,
        found_so_far: Optional[List[Path]] = None,
    ) -> List[Path]:
        if found_so_far is None:
            found_so_far = []

        elif (
            max_search_depth == 0
            or parent_directory.is_file()
            or len(found_so_far) >= found_directories_limit
        ):
            return found_so_far

        elif parent_directory.name == target_folder:
            found_so_far.append(parent_directory)

        for child in parent_directory.iterdir():
            if len(found_so_far) < found_directories_limit:
                DataExtractor.directory_walk(
                    child,
                    target_folder,
                    max_search_depth - 1,
                    found_directories_limit,
                    found_so_far,
                )
            else:
                return found_so_far
        return found_so_far

    def __call__(
        self, parent_directory: Path, target_directory: str, save_path: Path
    ) -> None:
        experiments = self.extract_experiments(parent_directory, target_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        for experiment in experiments:
            try:
                DataExtractor.save_data(
                    experiment,
                    save_path.joinpath(f"{uuid4().hex}.h5"),
                    self.max_workers,
                    self.timeout,
                )
                logger.log("info", f"Saved data for {experiment.parent_directory}")
            except Exception as e:
                logger.log(
                    "error", f"Failed to save data for {experiment.parent_directory}"
                )
                logger.log("error", e)

    @staticmethod
    def _peek_image_shape(tif_file: Path) -> Tuple[int, ...]:
        image = DataExtractor.extract_tif_file_image(tif_file)
        return image.shape

    @staticmethod
    def save_dict_to_hdf5(data: Dict[str, Any], hdf5_group: h5py.Group) -> None:
        """
        Recursively saves a nested dictionary to an HDF5 group.

        Parameters
        ----------
        data : Dict[str, Any])
            The dictionary to save.
        hdf5_group : h5py.Group
            The HDF5 group to save the data into. This can be the HDF5 file itself or a subgroup within the file.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                # Create a subgroup for nested dictionaries
                subgroup = hdf5_group.create_group(key)
                DataExtractor.save_dict_to_hdf5(value, subgroup)
            elif type(value) in {str, bytes}:
                # Directly save non-dictionary values
                hdf5_group[key] = value
            else:
                # Convert non-dictionary values to numpy arrays and save them
                hdf5_group.create_dataset(key, data=value, compression="gzip")

    @staticmethod
    def save_data(
        experiment: Experiment,
        save_path: Path,
        max_workers: int,
        timeout: int,
    ) -> None:
        data = DataExtractor.extract_data(
            experiment.tif_files,
            experiment.parent_directory,
            max_workers=max_workers,
            timeout=timeout,
        )
        with h5py.File(save_path, "w") as f:
            DataExtractor.save_dict_to_hdf5(data, f)
        # torch.save(data, save_path, pickle_protocol=4)

    @staticmethod
    def process_file(tif_file: Path) -> Tuple[np.ndarray, dict]:
        file_name = tif_file.name
        metadata = DataExtractor.extract_tif_file_metadata(file_name)
        image_data = DataExtractor.extract_tif_file_image(tif_file)
        return image_data, metadata

    @logger.catch
    @staticmethod
    def extract_data(
        tif_files: List[str],
        parent_directory: Path,
        max_workers: int,
        timeout: int,
    ) -> Dict[str, dict]:
        data = DataExtractor.create_data_dict()

        tif_file_paths = [parent_directory.joinpath(tif_file) for tif_file in tif_files]

        dims = DataExtractor._peek_image_shape(tif_file_paths[0])
        shape = (len(tif_file_paths), *dims)

        with get_reusable_executor(
            max_workers=max_workers, timeout=timeout
        ) as executor:
            futures = [
                executor.submit(DataExtractor.process_file, tif_file)
                for tif_file in tif_file_paths
            ]

            for idx, future in tqdm(
                enumerate(as_completed(futures)), total=len(futures)
            ):
                image_data, metadata = future.result()
                wavelength = metadata["wavelength"]
                if wavelength not in data["wavelength"]:
                    data["wavelength"][wavelength] = {
                        "data": {
                            "values": np.zeros(shape, dtype=np.float32),
                            "indices": [],
                        },
                        "metadata": {
                            "file_names": [],
                            "parent_directory": str(parent_directory).encode("utf-8"),
                        },
                    }
                data["wavelength"][wavelength]["data"]["values"][idx] = image_data
                data["wavelength"][wavelength]["data"]["indices"].append(
                    metadata["index"]
                )
                data["wavelength"][wavelength]["metadata"]["file_names"].append(
                    metadata["file_name"].encode("utf-8")
                )

        return data

    @staticmethod
    def create_data_dict() -> Dict[str, dict]:
        data = {
            "wavelength": dict(),
        }
        return data

    @logger.catch
    @staticmethod
    def extract_tif_file_image(file_name: Path) -> np.ndarray:
        data = tif.imread(file_name)  # (Z, Y, X)
        return data

    @staticmethod
    def extract_tif_file_metadata(file_name: str) -> Dict[str, str]:
        metadata = dict()
        fname = file_name.split("_")
        index = int(
            fname[3][5:]
        )  # e.g. stack0145 -> 145 (will later be used as index for sparse tensor)
        wavelength = fname[4]
        metadata["index"] = index
        metadata["wavelength"] = wavelength
        metadata["file_name"] = file_name
        return metadata

    @staticmethod
    def get_tif_files(directory: Path) -> List[Path]:
        return list(file for file in directory.rglob("*.tif"))

    @staticmethod
    def _extract_experiments(
        directories: List[Path],
        min_tif_files: int,
    ) -> Dict[Path, List[str]]:
        experiments = defaultdict(list)
        for directory in directories:
            files = DataExtractor.get_tif_files(directory)
            if len(files) >= min_tif_files:
                for file in files:
                    parent_directory = file.parent
                    for file in files:
                        experiments[parent_directory].append(file.name)
        return experiments


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Prepare data for lattice microscopy data into .pt format."
    )
    parser.add_argument(
        "parent_directory", type=Path, help="Parent directory of the data."
    )
    parser.add_argument(
        "--target_directory", type=str, help="Name of the target directory."
    )
    parser.add_argument("--save_path", type=Path, help="Path to save the .h5 files.")
    parser.add_argument(
        "--max_search_depth",
        type=int,
        default=6,
        help="Max depth to search for the target directory.",
    )
    parser.add_argument(
        "--found_directories_limit", type=int, help="Max number of directories to find."
    )
    parser.add_argument(
        "--min_tif_files_in_folder",
        type=int,
        default=20,
        help="Min number of .tif files to consider a directory as a target directory.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max number of workers."
    )
    parser.add_argument(
        "--timeout", type=int, default=10, help="Timeout for each worker."
    )

    args = parser.parse_args()
    data_extractor = DataExtractor(
        args.max_search_depth,
        args.found_directories_limit,
        args.min_tif_files_in_folder,
        args.max_workers,
        args.timeout,
    )
    data_extractor(
        args.parent_directory,
        args.target_directory,
        args.save_path,
    )
