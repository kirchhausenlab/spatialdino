from cell_interactome.data.utils import save_dict_to_hdf5
from concurrent.futures import as_completed
import tifffile as tif
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple, TypedDict, ClassVar, Generator
import numpy as np
from loky import get_reusable_executor
from dataclasses import dataclass
from loguru import logger
import h5py
import glob
import numpy.typing as npt


class Experiment(TypedDict):
    tif_files: List[Path]
    name: str
    path: Path
    metadata: Dict[str, Any]


class VoxelData(TypedDict):
    values: np.ndarray
    position: Tuple[int, int, int]


class VoxelMetadata(TypedDict):
    path: str
    stack: int


class Voxel(TypedDict):
    data: VoxelData
    metadata: VoxelMetadata


@dataclass
class DataExtractor:
    # instance vars
    z_voxel_size: int = 32
    y_voxel_size: int = 224
    x_voxel_size: int = 224
    found_experiments_limit: int = 1
    timeout: Optional[float] = None
    max_workers: int = 4
    # class vars
    SEARCH_PATTERN: ClassVar[str] = "**/DS"
    DTYPE: ClassVar[npt.DTypeLike] = np.float32

    def __post_init__(self) -> None:
        if self.timeout == 0:
            self.timeout = None
        self.experiment_workers = max(
            int(0.1 * self.max_workers), 1
        )  # 10% of max workers
        available_workers = self.max_workers - self.experiment_workers
        self.save_data_workers = max(
            available_workers // self.experiment_workers, 1
        )  # evenly distribute workers

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
            search_pattern=DataExtractor.SEARCH_PATTERN,
            found_experiments_limit=self.found_experiments_limit,
            save_path=save_path,
        )

        with get_reusable_executor(max_workers=self.experiment_workers) as executor:
            futures = []
            experiment_paths = []

            for experiment in experiments:
                experiment_save_path = save_path.joinpath(
                    experiment["name"],
                    f"{experiment['metadata']['wavelength']}_{experiment['metadata']['camera']}",
                )
                experiment_save_path.mkdir(parents=True, exist_ok=True)
                experiment_paths.append(experiment_save_path)

                futures.append(
                    executor.submit(
                        self.save_data,
                        z_voxel_size=self.z_voxel_size,
                        y_voxel_size=self.y_voxel_size,
                        x_voxel_size=self.x_voxel_size,
                        base_path=experiment["path"],
                        tif_files=experiment["tif_files"],
                        save_path=experiment_save_path,
                        dtype=DataExtractor.DTYPE,
                    )
                )

            for future in as_completed(futures, timeout=self.timeout):
                experiment_save_path = experiment_paths[futures.index(future)]
                try:
                    future.result()
                    logger.info(f"Successfully saved data in {experiment_save_path}")
                except Exception as e:
                    logger.error(f"Failed to save data in {experiment_save_path}")
                    logger.error(str(e))

    def save_data(
        self,
        z_voxel_size: int,
        y_voxel_size: int,
        x_voxel_size: int,
        base_path: Path,
        tif_files: List[Path],
        save_path: Path,
        dtype: npt.DTypeLike,
    ) -> None:
        with get_reusable_executor(max_workers=self.save_data_workers) as executor:
            futures = [
                executor.submit(
                    DataExtractor._save_data,
                    z_voxel_size=z_voxel_size,
                    y_voxel_size=y_voxel_size,
                    x_voxel_size=x_voxel_size,
                    base_path=base_path,
                    tif_file=tif_file,
                    save_path=save_path,
                    dtype=dtype,
                )
                for tif_file in tif_files
            ]

            for future in as_completed(futures, timeout=self.timeout):
                future.result()

    @staticmethod
    def _save_data(
        z_voxel_size: int,
        y_voxel_size: int,
        x_voxel_size: int,
        base_path: Path,
        tif_file: Path,
        save_path: Path,
        dtype: npt.DTypeLike,
    ) -> None:
        image_data, metadata = DataExtractor.process_file(tif_file, base_path, dtype)
        stack_save_path = save_path.joinpath(f"stack_{metadata['stack']}")
        stack_save_path.mkdir(parents=False, exist_ok=True)
        voxels = DataExtractor.voxelize(
            image_data=image_data,
            metadata=metadata,
            z_voxel_size=z_voxel_size,
            y_voxel_size=y_voxel_size,
            x_voxel_size=x_voxel_size,
        )
        for idx, voxel in enumerate(voxels):
            voxel_save_path = stack_save_path.joinpath(f"part_{idx}.h5")
            with h5py.File(voxel_save_path, "w") as f:
                save_dict_to_hdf5(voxel, f)

    @staticmethod
    def process_file(
        tif_file: Path, base_path: Path, dtype: npt.DTypeLike
    ) -> Tuple[np.ndarray, dict]:
        file_name = tif_file.name
        stack = DataExtractor.extract_stack(file_name)
        path = base_path.joinpath(tif_file)
        metadata = {"stack": stack, "path": str(path)}
        image_data = DataExtractor.extract_tif_file_image(path, dtype=dtype)
        return image_data, metadata

    @staticmethod
    def extract_tif_file_image(file_name: Path, dtype: npt.DTypeLike) -> np.ndarray:
        data = tif.imread(file_name, chunkdtype=dtype)  # (Z, Y, X)
        return data

    @staticmethod
    def extract_stack(file_name: str) -> int:
        stack_number = [part for part in file_name.split("_") if "stack" in part][0]
        stack = int(
            stack_number[5:]
        )  # e.g. stack0145 -> 145 (will later be used as index for sparse tensor)
        return stack

    @staticmethod
    def get_tif_files(directory: Path) -> List[Path]:
        return list(file for file in directory.rglob("*.tif"))

    @staticmethod
    def voxelize(
        image_data: np.ndarray,
        metadata: Dict[str, Any],
        z_voxel_size: int,
        y_voxel_size: int,
        x_voxel_size: int,
    ) -> Generator[Voxel, None, None]:
        z, y, x = image_data.shape
        for z_idx in range(0, z, z_voxel_size):
            for y_idx in range(0, y, y_voxel_size):
                for x_idx in range(0, x, x_voxel_size):
                    z_end = min(z_idx + z_voxel_size, z)
                    y_end = min(y_idx + y_voxel_size, y)
                    x_end = min(x_idx + x_voxel_size, x)
                    voxel_data = image_data[z_idx:z_end, y_idx:y_end, x_idx:x_end]
                    voxel_metadata = VoxelMetadata(
                        path=metadata["path"], stack=metadata["stack"]
                    )
                    voxel = Voxel(
                        data=VoxelData(
                            values=voxel_data,
                            position=(z_idx, y_idx, x_idx),
                        ),
                        metadata=voxel_metadata,
                    )
                    yield voxel


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Prepare data for lattice microscopy data into .h5 format."
    )

    parser.add_argument(
        "--z_voxel_size", type=int, required=True, help="Z voxel size for chunking."
    )
    parser.add_argument(
        "--y_voxel_size", type=int, required=True, help="Y voxel size for chunking."
    )
    parser.add_argument(
        "--x_voxel_size", type=int, required=True, help="X voxel size for chunking."
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
        "--timeout", type=float, default=0, help="Timeout for each worker."
    )

    args = parser.parse_args()
    data_extractor = DataExtractor(
        z_voxel_size=args.z_voxel_size,
        y_voxel_size=args.y_voxel_size,
        x_voxel_size=args.x_voxel_size,
        found_experiments_limit=args.found_experiments_limit,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )
    data_extractor(
        parent_directory=args.parent_directory,
        save_path=args.save_path,
    )
