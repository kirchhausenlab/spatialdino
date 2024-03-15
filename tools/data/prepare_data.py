from collections import defaultdict
from concurrent.futures import as_completed
from functools import partial
import tifffile as tif
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loky import get_reusable_executor
import h5py
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm
import numpy.typing as npt
import torch

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


# possible schema


"""
{"data": {"488nm": DATA, "560nm": DATA, "642nm": DATA, ...}, "metadata": {"stack": []}}

How to extract metadata from filename?

1. split filename by "_"
2. fname = ex01_CamB_ch1_stack0145_560nm_0469852msec_0178557836msecAbs_000x_000y_000z_0000t
3. camera = fname[1]
4. stack = int(fname[3])

where: DATA is concatenated along time axis to form an array of shape (t, z, y, x)
"""


@dataclass
class Experiment:
    tif_files: List[Path]
    experiment_name: str


class DataExtractor:
    def __init__(
        self,
        max_parent_directories: int,
        min_tif_files: int,
        max_search_depth: int,
        timeout: int,
        max_workers: int = 4,
    ) -> None:
        self.max_parent_directories = max_parent_directories
        self.min_tif_files = min_tif_files
        self.max_search_depth = max_search_depth
        self.max_workers = max_workers
        self.timeout = timeout

    def extract_experiments(
        self, parent_directory: Path, target_directory: str
    ) -> List[Experiment]:
        target_directories = DataExtractor.extract_target_directories(
            parent_directory,
            target_directory,
            self.max_parent_directories,
            self.max_search_depth,
        )

        parent_directory_components = parent_directory.parts
        experiments = DataExtractor._extract_experiments(
            target_directories, self.min_tif_files, parent_directory_components
        )

        experiments = [
            Experiment(tif_files, experiment_name)
            for experiment_name, tif_files in experiments.items()
        ]

        return experiments

    @staticmethod
    def _get_experiment_name(
        experiment_name: Path, parent_directory_components: Tuple[str, ...]
    ) -> str:
        return "".join(
            [
                path_component
                for path_component in experiment_name.parts
                if path_component not in set(parent_directory_components)
            ]
        ).replace(".tif", "")

    def __call__(
        self, parent_directory: Path, target_directory: str, save_path: Path
    ) -> None:
        experiments = self.extract_experiments(parent_directory, target_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        DataExtractor.save_data(
            experiments[0], save_path, self.max_workers, self.timeout
        )

    @staticmethod
    def save_data(
        experiment: Experiment, save_path: Path, max_workers: int, timeout: int
    ) -> None:
        data = DataExtractor.extract_data(
            experiment.tif_files, max_workers=max_workers, timeout=timeout
        )
        save_file_path = save_path.joinpath(f"{experiment.experiment_name}.pt")
        torch.save(data, save_file_path, pickle_protocol=4)

    @staticmethod
    def process_file(tif_file: Path) -> Tuple[np.ndarray, dict]:
        file_name = tif_file.name
        metadata = DataExtractor.extract_tif_file_metadata(file_name)
        image_data = DataExtractor.extract_tif_file_image(tif_file)
        return image_data, metadata

    @staticmethod
    def _peek_image_shape(tif_file: Path) -> Tuple[int, ...]:
        image = DataExtractor.extract_tif_file_image(tif_file)
        return image.shape

    @logger.catch
    @staticmethod
    def extract_data(
        tif_files: List[Path], max_workers: int, timeout: int
    ) -> Dict[str, dict]:
        data = DataExtractor.create_data_dict()

        # peek to find np.array shape
        dims = DataExtractor._peek_image_shape(tif_files[0])

        with get_reusable_executor(
            max_workers=max_workers, timeout=timeout
        ) as executor:
            futures = [
                executor.submit(DataExtractor.process_file, tif_file)
                for tif_file in tif_files
            ]

            for future in as_completed(futures):
                image_data, metadata = future.result()
                wavelength = metadata["wavelength"]
                if "wavelength" not in data["data"]:
                    data["data"][wavelength] = np.zeros(
                        (len(tif_files), *dims), dtype=np.float32
                    )
                data["data"][wavelength][metadata["stack"]] = image_data
                data["metadata"][wavelength]["stack"].append(metadata["stack"])

        for wavelength in data["metadata"].keys():
            data["metadata"][wavelength]["stack"].sort()

        return data

    @staticmethod
    def create_data_dict() -> Dict[str, dict]:
        data = {
            "data": dict(),
            "metadata": defaultdict(lambda: {"stack": []}),
        }
        return data

    @staticmethod
    def extract_tif_file_image(file_name: Path) -> np.ndarray:
        data = tif.imread(file_name)  # (Z, Y, X)
        return data

    @staticmethod
    def extract_tif_file_metadata(file_name: str) -> Dict[str, str]:
        metadata = dict()
        fname = file_name.split("_")
        stack = int(fname[3][5:])  # e.g. stack0145 -> 145
        wavelength = fname[4]
        metadata["stack"] = stack
        metadata["wavelength"] = wavelength
        return metadata

    @staticmethod
    def directory_walk(
        parent_directory: Path, target_folder: str, max_search_depth: int
    ) -> List[Path]:
        if max_search_depth == 0 or parent_directory.is_file():
            return []
        elif parent_directory.name == target_folder:
            return [parent_directory]
        return [
            path
            for child in parent_directory.iterdir()
            for path in DataExtractor.directory_walk(
                child, target_folder, max_search_depth - 1
            )
        ]

    @staticmethod
    def extract_target_directories(
        parent_directory: Path,
        target_folder: str,
        max_parent_directories: int,
        max_search_depth: int,
    ) -> List[Path]:
        target_directories = []
        for _ in range(max_parent_directories):
            target_directories.extend(
                DataExtractor.directory_walk(
                    next(parent_directory.iterdir()), target_folder, max_search_depth
                )
            )
        return target_directories

    @staticmethod
    def get_tif_files(directory: Path) -> List[Path]:
        return list(
            filter(
                lambda file: file.suffix == ".tif",
                (file for file in directory.iterdir()),
            )
        )

    @staticmethod
    def _extract_experiments(
        directories: List[Path],
        min_tif_files: int,
        parent_directory_components: Tuple[str, ...],
    ) -> Dict[str, List[Path]]:
        experiments = defaultdict(list)
        for directory in directories:
            files = DataExtractor.get_tif_files(directory)
            experiment_name = DataExtractor._get_experiment_name(
                files[0], parent_directory_components
            )
            if len(files) > min_tif_files:
                experiments[experiment_name].extend(files)
        return experiments


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Prepare data for lattice microscopy data into .h5 format."
    )
    parser.add_argument(
        "parent_directory", type=Path, help="Parent directory of the data."
    )
    parser.add_argument(
        "--target_directory", type=str, help="Name of the target directory."
    )
    parser.add_argument("--save_path", type=Path, help="Path to save the .h5 files.")
    parser.add_argument(
        "--max_parent_directories",
        type=int,
        default=1,
        help="Max number of directories to search in parent directory.",
    )
    parser.add_argument(
        "--min_tif_files",
        type=int,
        default=50,
        help="Min number of .tif files to consider a directory as a target directory.",
    )
    parser.add_argument(
        "--max_search_depth",
        type=int,
        default=6,
        help="Max depth to search for target directory.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max number of workers."
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout for each worker."
    )
    args = parser.parse_args()
    data_extractor = DataExtractor(
        args.max_parent_directories,
        args.min_tif_files,
        args.max_search_depth,
        args.max_workers,
        args.timeout,
    )
    data_extractor(
        args.parent_directory,
        args.target_directory,
        args.save_path,
    )
