from concurrent.futures import as_completed
import tifffile as tif
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple
import numpy as np
from loky import get_reusable_executor
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm
import h5py
import glob
import time
import numpy.typing as npt


@dataclass
class Experiment:
    tif_files: List[Path]
    name: str
    path: Path


@dataclass
class DataExtractor:
    found_directories_limit: int
    min_tif_files_in_folder: int
    timeout: int
    max_workers: int = 4

    def extract_experiments(
        self,
        parent_directory: Path,
        search_pattern: str,
        directories_up: int,
    ) -> List[Experiment]:
        target_directories = DataExtractor.directory_walk(
            parent_directory,
            search_pattern,
            self.found_directories_limit,
            directories_up,
        )

        experiments = DataExtractor._extract_experiments(
            target_directories,
            self.min_tif_files_in_folder,
            directories_up,
            search_pattern,
        )

        return experiments

    @staticmethod
    def directory_walk(
        parent_directory: Path,
        search_pattern: str,
        found_directories_limit: int,
        directories_up: int,
    ) -> List[Path]:
        found_so_far = set()
        #lst = [parent_directory.iterdir()]
        for directory in parent_directory.iterdir():
            # if len(found_so_far) >= found_directories_limit:
            #     return list(found_so_far)
            for matched_dir in glob.glob(
                str(directory.joinpath(search_pattern)), recursive=True
            ):
                matched_dir = Path(matched_dir)
                print(f"********* matched_dir is{matched_dir} and the parent is {matched_dir.parent}\n ")                

                if matched_dir.is_dir():  # Ensure matched_dir is a directory
                    if len(found_so_far) >= found_directories_limit:
                        return list(found_so_far)
                    else:
                        # if save_path.joinpath(f"{experiment.name}.h5").is_file():
                        #     logger.info(f"Data for {experiment.name} already exists. Skipping.")
                        #     continue
                        if directories_up == 0:
                            found_so_far.add(matched_dir)
                        else:
                            print(f"\n\n ||||| PRINTING MATCH DIR PARENTS {matched_dir.parents[directories_up - 1]}")
                            found_so_far.add(matched_dir.parents[directories_up - 1])

        return list(found_so_far)

    def __call__(
        self,
        parent_directory: Path,
        search_pattern: str,
        directories_up: int,
        save_path: Path,
    ) -> None:
        experiments = self.extract_experiments(
            parent_directory, search_pattern, directories_up
        )
        save_path.mkdir(parents=True, exist_ok=True)
        for experiment in experiments:
            if save_path.joinpath(f"{experiment.name}.h5").is_file():
                logger.info(f"Data for {experiment.name} already exists. Skipping.")
                continue
            try:
                DataExtractor.save_data(
                    experiment,
                    save_path.joinpath(f"{experiment.name}.h5"),
                    self.max_workers,
                    self.timeout,
                )
            except Exception as e:
                logger.error(
                    f"Failed to save data for {experiment.name} with error: {e}"
                )

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
            elif isinstance(value, bytes):
                hdf5_group[key] = value
            elif isinstance(value, str):
                hdf5_group[key] = value.encode("utf-8")
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
            experiment,
            max_workers=max_workers,
            timeout=timeout,
        )
        logger.info(f"Saving data for {experiment.name}")
        start = time.perf_counter()
        with h5py.File(save_path, "w") as f:
            DataExtractor.save_dict_to_hdf5(data, f)
        end = time.perf_counter()
        time_elapsed = end - start
        logger.info(f"Saved data for {experiment.name} in {time_elapsed} seconds.")

    @staticmethod
    def process_file(tif_file: Path, base_path: Path) -> Tuple[np.ndarray, dict]:
        file_name = tif_file.name
        metadata = DataExtractor.extract_tif_file_metadata(file_name)
        image_data = DataExtractor.extract_tif_file_image(base_path.joinpath(tif_file))
        return image_data, metadata

    @logger.catch
    @staticmethod
    def extract_data(
        experiment: Experiment,
        max_workers: int,
        timeout: int,
        dtype: npt.DTypeLike = np.float32,
    ) -> Dict[str, dict]:
        data = DataExtractor.create_data_dict(experiment.path)
        data["metadata"]["base_path"] = data["metadata"]["base_path"]

        base_path = experiment.path
        tif_files = experiment.tif_files

        dims = DataExtractor._peek_image_shape(base_path.joinpath(tif_files[0]))
        shape = (len(tif_files), *dims)

        with get_reusable_executor(
            max_workers=max_workers, timeout=timeout
        ) as executor:
            futures = [
                executor.submit(DataExtractor.process_file, tif_file, base_path)
                for tif_file in tif_files
            ]

            for idx, future in tqdm(
                enumerate(as_completed(futures)), total=len(futures)
            ):
                image_data, metadata = future.result()
                wavelength = metadata["wavelength"]
                if wavelength not in data["wavelength"]:
                    data["wavelength"][wavelength] = {
                        "data": {
                            "values": np.zeros(shape, dtype=dtype),
                            "indices": [],
                        },
                        "metadata": {"relative_paths": []},
                    }
                data["wavelength"][wavelength]["data"]["values"][idx] = image_data
                data["wavelength"][wavelength]["data"]["indices"].append(
                    metadata["index"]
                )
                data["wavelength"][wavelength]["metadata"]["relative_paths"].append(
                    metadata["relative_path"]
                )

        return data

    @staticmethod
    def create_data_dict(experiment_path: Path) -> Dict[str, dict]:
        data = {
            "wavelength": dict(),
            "metadata": {"base_path": str(experiment_path)},
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
        metadata["relative_path"] = file_name
        return metadata

    @staticmethod
    def get_tif_files(directory: Path, search_pattern: str) -> List[Path]:
        return list(file for file in directory.rglob(f"{search_pattern}/*.tif"))

    @staticmethod
    def _extract_experiments(
        directories: List[Path],
        min_tif_files: int,
        directories_up: int,
        search_pattern: str,
    ) -> List[Experiment]:
        experiments = []
        for directory in directories:
            files = DataExtractor.get_tif_files(directory, search_pattern)
            if len(files) >= min_tif_files:
                experiment_path = files[0].parents[directories_up]
                experiment_name = experiment_path.name
                files = [file.relative_to(experiment_path) for file in files]
                experiments.append(Experiment(files, experiment_name, experiment_path))
        return experiments


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Prepare data for lattice microscopy data into .h5 format."
    )
    parser.add_argument(
        "--parent_directory",
        type=Path,
        required=True,
        help="Parent directory to start searching.",
    )
    parser.add_argument(
        "--search_pattern",
        type=str,
        required=True,
        help="Pattern of directory to match including parent directory. Should not have a leading /.",
    )
    parser.add_argument(
        "--directories_up",
        type=int,
        required=False,
        default=0,
        help="Number of directories to go up from the matched directory.",
    )
    parser.add_argument("--save_path", type=Path, help="Path to save the .h5 files.")
    parser.add_argument(
        "--found_directories_limit",
        type=int,
        default=1,
        help="Max number of directories to find.",
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
        args.found_directories_limit,
        args.min_tif_files_in_folder,
        args.timeout,
        args.max_workers,
    )
    data_extractor(
        args.parent_directory,
        args.search_pattern,
        args.directories_up,
        args.save_path,
    )
