from cell_interactome.data.utils import save_dict_to_hdf5
from concurrent.futures import as_completed
import tifffile as tif
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import numpy as np
from loky import get_reusable_executor
from dataclasses import dataclass
from cell_interactome.logging.config import logger
from functools import lru_cache
import h5py
import glob
import time
import numpy.typing as npt


class Experiment(TypedDict):
    tif_files: List[Path]
    name: str
    path: Path
    metadata: Dict[str, Any]


@dataclass
class DataExtractor:
    found_directories_limit: int
    min_tif_files_in_folder: int
    timeout: Optional[int]
    max_workers: int = 4

    def __post_init__(self) -> None:
        if self.timeout == 0:
            self.timeout = None

    def extract_experiments(
        self,
        parent_directory: Path,
        search_pattern: str,
        min_tif_files: int,
        save_path: Path,
    ) -> List[Experiment]:
        experiments = DataExtractor._extract_experiments(
            parent_directory=parent_directory,
            search_pattern=search_pattern,
            found_directories_limit=self.found_directories_limit,
            min_tif_files=min_tif_files,
            save_path=save_path,
        )
        return experiments

    @lru_cache
    @staticmethod
    def _extract_experiments(
        parent_directory: Path,
        search_pattern: str,
        found_directories_limit: int,
        min_tif_files: int,
        save_path: Path,
    ) -> List[Experiment]:
        found_so_far = set()
        experiments_found_so_far = set()
        experiments = []

        for directory in parent_directory.iterdir():
            for matched_dir in glob.glob(
                str(directory.joinpath(search_pattern)), recursive=True
            ):
                experiment_name = DataExtractor.get_experiment_name(
                    matched_dir=matched_dir,
                    prefix_length=len(str(parent_directory)),
                )
                if save_path.joinpath(experiment_name).is_dir():
                    logger.info(
                        "Data for %s already exists. Skipping." % (experiment_name)
                    )
                    continue
                experiments_found_so_far.add(experiment_name)

                if len(experiments_found_so_far) > found_directories_limit:
                    return experiments

                matched_path = Path(matched_dir)
                if matched_path.is_dir() and matched_path not in found_so_far:
                    tif_files = DataExtractor.get_tif_files(matched_path)
                    if len(tif_files) >= min_tif_files:
                        found_so_far.add(matched_path)
                        metadata = DataExtractor.get_experiment_metadata(matched_dir)
                        experiments.append(
                            Experiment(
                                tif_files=tif_files,
                                name=experiment_name,
                                path=matched_path,
                                metadata=metadata,
                            )
                        )

        return experiments

    @staticmethod
    def get_experiment_metadata(matched_dir: str) -> Dict[str, str]:
        metadata_string = matched_dir.split("/")[-2]
        wavelength = metadata_string[2:7]
        camera = metadata_string[7:]
        metadata = {"wavelength": wavelength, "camera": camera}
        return metadata

    @staticmethod
    def get_experiment_name(matched_dir: str, prefix_length: int) -> str:
        """
        Get the specific experiment from parent folder
        full path looks like - /nfs/datasync4/tklab-llsm/20220121_p5_p55_sCMOS_Anand_phrodo_NPC/CS1_Phrodo_NPC/Ex03_488_300mW_560_500mW_642_500mW_z0p5/ch560nmCamB/DS
        """
        matched_dir = matched_dir[prefix_length + 1 :]
        split_dir = matched_dir.split("/")[:-2]
        experiment_name = "__".join(split_dir)
        return experiment_name

    def __call__(
        self,
        parent_directory: Path,
        min_tif_files: int,
        search_pattern: str,
        save_path: Path,
    ) -> None:
        experiments = self.extract_experiments(
            parent_directory=parent_directory,
            search_pattern=search_pattern,
            min_tif_files=min_tif_files,
            save_path=save_path,
        )

        with get_reusable_executor(max_workers=self.max_workers) as executor:
            futures = []
            for experiment in experiments:
                data_save_path = save_path.joinpath(experiment["name"])
                data_save_path.mkdir(parents=True, exist_ok=True)
                file_name = f"{experiment['metadata']['wavelength']}_{experiment['metadata']['camera']}.h5"
                future = executor.submit(
                    DataExtractor.save_data,
                    base_path=experiment["path"],
                    tif_files=experiment["tif_files"],
                    experiment_metadata=experiment["metadata"],
                    experiment_name=experiment["name"],
                    save_path=data_save_path.joinpath(file_name),
                )
                futures.append(future)

            for future in as_completed(futures, timeout=self.timeout):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to save data with error: {e}")

    @staticmethod
    def _peek_image_shape(tif_file: Path) -> Tuple[int, ...]:
        image = DataExtractor.extract_tif_file_image(tif_file)

        return image.shape

    @staticmethod
    def save_data(
        base_path: Path,
        tif_files: List[Path],
        experiment_metadata: Dict[str, Any],
        experiment_name: str,
        save_path: Path,
    ) -> None:
        logger.info(f"Saving data for {experiment_name}")
        start = time.perf_counter()
        data = DataExtractor.extract_data(base_path, tif_files, experiment_metadata)
        with h5py.File(save_path, "w") as f:
            save_dict_to_hdf5(data, f)
        end = time.perf_counter()
        time_elapsed = end - start
        logger.info(f"Saved data for {experiment_name} in {time_elapsed} seconds.")

    @staticmethod
    def process_file(tif_file: Path, base_path: Path) -> Tuple[np.ndarray, dict]:
        file_name = tif_file.name
        metadata = DataExtractor.extract_tif_file_metadata(file_name)
        image_data = DataExtractor.extract_tif_file_image(base_path.joinpath(tif_file))
        return image_data, metadata

    @staticmethod
    def extract_data(
        base_path: Path,
        tif_files: List[Path],
        experiment_metadata: Dict[str, Any],
        dtype: npt.DTypeLike = np.float32,
    ) -> Dict[str, dict]:
        dims = DataExtractor._peek_image_shape(base_path.joinpath(tif_files[0]))
        shape = (len(tif_files), *dims)
        data = DataExtractor.create_data_dict(
            metadata=experiment_metadata, shape=shape, dtype=dtype
        )
        for idx, tif_file in enumerate(tif_files):
            image_data, metadata = DataExtractor.process_file(tif_file, base_path)
            data["data"]["values"][idx] = image_data
            data["data"]["indices"].append(metadata["index"])
            data["metadata"]["paths"].append(metadata["path"])

        return data

    @staticmethod
    def create_data_dict(
        metadata: Dict[str, Any], shape: Tuple[int, ...], dtype: npt.DTypeLike
    ) -> Dict[str, Any]:
        data = {
            "data": {"values": np.zeros(shape, dtype=dtype), "indices": []},
            "metadata": {"paths": []},
        }
        data["metadata"].update(metadata)
        return data

    @logger.catch()
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
        metadata["index"] = index
        metadata["path"] = file_name
        return metadata

    @staticmethod
    def get_tif_files(directory: Path) -> List[Path]:
        return list(file for file in directory.rglob("*.tif"))


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
        "--timeout", type=int, default=0, help="Timeout for each worker."
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
        args.min_tif_files_in_folder,
        args.search_pattern,
        args.save_path,
    )
