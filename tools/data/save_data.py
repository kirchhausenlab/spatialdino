from cell_interactome.data.utils import save_dict_to_hdf5
from hashlib import sha512
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
        min_tif_files: int,
        save_path: Path
    ) -> List[Experiment]:
        experiments = DataExtractor._extract_experiments(
            parent_directory,
            search_pattern,
            min_tif_files,
            self.found_directories_limit,
            save_path
        )
        return experiments

    @staticmethod
    def _extract_experiments(
        parent_directory: Path,
        search_pattern: str,
        found_directories_limit: int,
        min_tif_files: int,
        save_path: Path
    ) -> List[Experiment]:
        found_so_far = set()
        experiments = []

        for directory in parent_directory.iterdir():
            for matched_dir in glob.glob(
                str(directory.joinpath(search_pattern)), recursive=True
            ):
                experiment_name= DataExtractor.get_experiment_name(matched_dir=matched_dir,
                                                                   len_parent_directory=len(str(parent_directory))) 
                #logger.info(f"{list(save_path.iterdir())} and matched dir is {matched_dir} found_so far {found_so_far}")
                if save_path.joinpath(f"{experiment_name}.h5").is_file():
                    logger.info("Data for %s already exists. Skipping." % (experiment_name))
                    continue
                
                matched_path = Path(matched_dir)
                print(f"*****************{len(experiments)}\n\n\n\n")
                if matched_path.is_dir() and len(found_so_far) >= found_directories_limit:
                    return experiments

                if matched_path not in found_so_far:
                    tif_files = DataExtractor.get_tif_files(matched_path, search_pattern)
                    if len(tif_files) >= min_tif_files:
                        found_so_far.add(matched_path)
                        experiments.append(Experiment(tif_files, experiment_name, matched_path))
                    
        return experiments
    
    @staticmethod
    def get_experiment_name(
        matched_dir: str,
        len_parent_directory: int
    ) -> str:
        """
        Get the specific experiment from parent folder 
        full path looks like - /nfs/datasync4/tklab-llsm/20220121_p5_p55_sCMOS_Anand_phrodo_NPC/CS1_Phrodo_NPC/Ex03_488_300mW_560_500mW_642_500mW_z0p5/ch560nmCamB/DS
        """
        matched_dir = matched_dir[len_parent_directory+1:]
        split_dir = matched_dir.split("/")[:-2]
        experiment_name= "__".join(split_dir)
        return experiment_name 

    def __call__(
        self,
        parent_directory: Path,
        min_tif_files: int,
        search_pattern: str,
        save_path: Path,
    ) -> None:
        experiments = self.extract_experiments(
            parent_directory, search_pattern, min_tif_files, save_path
        )
        save_path.mkdir(parents=True, exist_ok=True)
        for experiment in experiments:
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
            save_dict_to_hdf5(data, f)
        end = time.perf_counter()
        time_elapsed = end - start
        logger.info(f"Saved data for {experiment.name} in {time_elapsed} seconds.")

    @staticmethod
    def process_file(tif_file: Path, base_path: Path) -> Tuple[np.ndarray, dict]:
        file_name = tif_file.name
        metadata = DataExtractor.extract_tif_file_metadata(file_name)
        image_data = DataExtractor.extract_tif_file_image(base_path.joinpath(tif_file))
        return image_data, metadata

    @staticmethod
    def extract_data(
        experiment: Experiment,
        max_workers: int,
        timeout: int,
        dtype: npt.DTypeLike = np.float32,
    ) -> Dict[str, dict]:
        data = DataExtractor.create_data_dict()
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
            # shows the number of tiff files
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
                        "metadata": {"paths": []},
                    }
                data["wavelength"][wavelength]["data"]["values"][idx] = image_data
                data["wavelength"][wavelength]["data"]["indices"].append(
                    metadata["index"]
                )
                data["wavelength"][wavelength]["metadata"]["paths"].append(
                    metadata["path"]
                )

        return data

    @staticmethod
    def create_data_dict() -> Dict[str, dict]:
        data = {
            "wavelength": dict()
        }
        return data

    @staticmethod
    def extract_tif_file_image(file_name: Path) -> np.ndarray:
        try:
            data = tif.imread(file_name) # (Z, Y, X)
            return data
        except Exception as e:
            print(f"Failed with error {e}")
            raise ValueError
        # print(type(data), data))
        #raise ValueError

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
        metadata["path"] = file_name
        return metadata

    @staticmethod
    def get_tif_files(directory: Path, search_pattern: str) -> List[Path]:
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
        args.min_tif_files_in_folder,
        args.search_pattern,
        args.save_path,
    )
