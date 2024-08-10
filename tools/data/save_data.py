from cell_interactome.data.utils import save_dict_to_hdf5
from concurrent.futures import as_completed
import tifffile as tif
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import numpy as np
from loky import get_reusable_executor
from dataclasses import dataclass
from loguru import logger
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
    found_experiments_limit: int
    timeout: Optional[int]
    max_workers: int = 4
    search_pattern: str = "**/DS"

    def __post_init__(self) -> None:
        if self.timeout == 0:
            self.timeout = None

    @staticmethod
    def extract_experiments(
        parent_directory: Path,
        search_pattern: str,
        found_experiments_limit: int,
        save_path: Path,
    ) -> List[Experiment]:
        experiments = DataExtractor._extract_experiments(
            parent_directory=parent_directory,
            search_pattern=search_pattern,
            found_experiments_limit=found_experiments_limit,
            save_path=save_path,
        )
        return experiments

    @staticmethod
    def _extract_experiments(
        parent_directory: Path,
        search_pattern: str,
        found_experiments_limit: int,
        save_path: Path,
    ) -> List[Experiment]:
        experiment_names = []
        experiment_dirs = []
        for directory in glob.iglob(
            str(parent_directory.joinpath(search_pattern)), recursive=True
        ):
            if len(experiment_names) == found_experiments_limit:
                break

            directory = Path(directory)
            parent_dir = directory.parents[1]
            if directory.is_dir():
                experiment_name = DataExtractor.get_experiment_name(
                    directory=str(parent_dir),
                    prefix_length=len(str(parent_directory)),
                )

                if save_path.joinpath(experiment_name).is_dir():
                    logger.info(
                        "Data for experiment: %s, already exists. Skipping."
                        % (experiment_name)
                    )
                    continue

                experiment_names.append(experiment_name)
                experiment_dirs.append(parent_dir)

        experiments = []
        for experiment_name, experiment_dir in zip(experiment_names, experiment_dirs):
            for folder in glob.iglob(
                str(experiment_dir.joinpath(search_pattern)),
                recursive=True,
            ):
                experiment_dir = Path(folder)
                tif_files = DataExtractor.get_tif_files(experiment_dir)
                metadata = DataExtractor.get_experiment_metadata(str(experiment_dir))
                experiments.append(
                    Experiment(
                        tif_files=tif_files,
                        name=experiment_name,
                        path=experiment_dir,
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
    def get_experiment_name(directory: str, prefix_length: int) -> str:
        """
        Get the specific experiment from parent folder
        full path looks like - /nfs/datasync4/tklab-llsm/20220121_p5_p55_sCMOS_Anand_phrodo_NPC/CS1_Phrodo_NPC/Ex03_488_300mW_560_500mW_642_500mW_z0p5
        """
        matched_dir = directory[prefix_length + 1 :]
        split_dir = matched_dir.split("/")
        experiment_name = "__".join(split_dir)
        return experiment_name

    def __call__(
        self,
        parent_directory: Path,
        save_path: Path,
    ) -> None:
        experiments = self.extract_experiments(
            parent_directory=parent_directory,
            search_pattern=self.search_pattern,
            found_experiments_limit=self.found_experiments_limit,
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
                    save_file_name=data_save_path.joinpath(file_name),
                )
                futures.append(future)

            for future in as_completed(futures, timeout=self.timeout):
                try:
                    future.result()
                except Exception as e:
                    logger.error("Failed to save data with error: %s" % (e))

    @staticmethod
    def _peek_image_shape(tif_file: Path) -> Tuple[int, ...]:
        image = DataExtractor.extract_tif_file_image(tif_file)

        return image.shape

    @staticmethod
    def save_data(
        base_path: Path,
        tif_files: List[Path],
        experiment_metadata: Dict[str, Any],
        save_file_name: Path,
    ) -> None:
        logger.info("Saving data in %s" % (save_file_name))
        start = time.perf_counter()
        data = DataExtractor.extract_data(base_path, tif_files, experiment_metadata)
        with h5py.File(save_file_name, "w") as f:
            save_dict_to_hdf5(data, f)
        end = time.perf_counter()
        time_elapsed = end - start
        logger.info("Saved data in %s in %s seconds." % (save_file_name, time_elapsed))

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
        stack = [part for part in file_name.split("_") if "stack" in part][0]
        index = int(
            stack[5:]
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
    parser.add_argument("--save_path", type=Path, help="Path to save the .h5 files.")
    parser.add_argument(
        "--found_experiments_limit",
        type=int,
        default=1,
        help="Max number of experiments to extract.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max number of workers."
    )
    parser.add_argument(
        "--timeout", type=int, default=0, help="Timeout for each worker."
    )

    args = parser.parse_args()
    data_extractor = DataExtractor(
        found_experiments_limit=args.found_experiments_limit,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )
    data_extractor(
        parent_directory=args.parent_directory,
        save_path=args.save_path,
    )
