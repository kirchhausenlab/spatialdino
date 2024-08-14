from cell_interactome.data import Voxel, VoxelData, VoxelMetadata
from concurrent.futures import ThreadPoolExecutor, as_completed
import tifffile as tif
from pathlib import Path
from argparse import ArgumentParser
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    TypedDict,
    ClassVar,
    Generator,
)
import numpy as np
from dataclasses import dataclass
from loguru import logger
import glob
from loky import ProcessPoolExecutor
import torch


class Experiment(TypedDict):
    tif_files: List[Path]
    name: str
    path: Path
    metadata: Dict[str, Any]


@dataclass
class DataExtractor:
    # instance vars
    z_voxel_size: int = 32
    y_voxel_size: int = 224
    x_voxel_size: int = 224
    found_experiments_limit: int = 1
    min_tif_files: int = 20
    max_workers: int = 4
    # class vars
    SEARCH_PATTERN: ClassVar[str] = "**/DS"
    DTYPE: ClassVar[str] = "float32"

    @staticmethod
    def extract_experiments(
        parent_directory: Path,
        search_pattern: str,
        found_experiments_limit: int,
        min_tif_files: int,
        save_path: Path,
    ) -> List[Experiment]:
        experiments = DataExtractor._extract_experiments(
            parent_directory=parent_directory,
            search_pattern=search_pattern,
            found_experiments_limit=found_experiments_limit,
            min_tif_files=min_tif_files,
            save_path=save_path,
        )
        return experiments

    @staticmethod
    def _extract_experiments(
        parent_directory: Path,
        search_pattern: str,
        found_experiments_limit: int,
        min_tif_files: int,
        save_path: Path,
    ) -> List[Experiment]:
        experiment_names = set()
        experiment_dirs = []
        for directory in glob.iglob(
            str(parent_directory.joinpath(search_pattern)), recursive=True
        ):
            if len(experiment_names) == found_experiments_limit:
                break

            directory = Path(directory)
            experiment_dir = directory.parents[1]
            if directory.is_dir():
                experiment_name = DataExtractor.get_experiment_name(
                    directory=str(experiment_dir),
                    prefix_length=len(str(parent_directory)),
                )
                if save_path.joinpath(experiment_name).is_dir():
                    continue

                num_tif_files = len(DataExtractor.get_tif_files(experiment_dir))
                if num_tif_files >= min_tif_files:
                    if experiment_name not in experiment_names:
                        experiment_names.add(experiment_name)
                        experiment_dirs.append(experiment_dir)

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
        experiments = DataExtractor.extract_experiments(
            parent_directory=parent_directory,
            search_pattern=DataExtractor.SEARCH_PATTERN,
            found_experiments_limit=self.found_experiments_limit,
            min_tif_files=self.min_tif_files,
            save_path=save_path,
        )

        all_futures = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for experiment in experiments:
                experiment_save_path = save_path.joinpath(
                    experiment["name"],
                    f"{experiment['metadata']['wavelength']}_{experiment['metadata']['camera']}",
                )
                experiment_save_path.mkdir(parents=True, exist_ok=True)
                futures = [
                    executor.submit(
                        DataExtractor._save_data,
                        z_voxel_size=self.z_voxel_size,
                        y_voxel_size=self.y_voxel_size,
                        x_voxel_size=self.x_voxel_size,
                        base_path=str(experiment["path"]),
                        tif_file=str(tif_file),
                        wavelength=int(
                            experiment["metadata"]["wavelength"].split("nm")[0]
                        ),
                        save_path=str(experiment_save_path),
                        dtype=DataExtractor.DTYPE,
                        max_workers=self.max_workers,
                    )
                    for tif_file in experiment["tif_files"]
                ]

                all_futures.extend(futures)

        for future in as_completed(all_futures, timeout=None):
            try:
                stack_save_path = future.result()
                logger.info("Successfully saved data in %s" % (stack_save_path))
            except Exception as e:
                logger.error("Failed to save data due to error: %s" % (e))

    @staticmethod
    def _save_data(
        z_voxel_size: int,
        y_voxel_size: int,
        x_voxel_size: int,
        base_path: str,
        tif_file: str,
        wavelength: int,
        save_path: str,
        dtype: str,
        max_workers: int,
    ) -> Path:
        base_path = Path(base_path)  # type: ignore
        tif_file = Path(tif_file)  # type: ignore
        save_path = Path(save_path)  # type: ignore
        image_data, metadata = DataExtractor.process_file(tif_file, base_path, dtype)  # type: ignore
        metadata["wavelength"] = wavelength
        stack_save_path = save_path.joinpath(f"stack_{metadata['stack']}")  # type: ignore
        stack_save_path.mkdir(parents=False, exist_ok=True)
        voxels = DataExtractor.voxelize(
            image_data=image_data,
            metadata=metadata,
            z_voxel_size=z_voxel_size,
            y_voxel_size=y_voxel_size,
            x_voxel_size=x_voxel_size,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, voxel in enumerate(voxels):
                voxel_save_path = stack_save_path.joinpath(f"part_{idx}.pth")
                executor.submit(DataExtractor.save_voxel, voxel, voxel_save_path)
        return stack_save_path

    @staticmethod
    def save_voxel(voxel: Voxel, save_path: Path) -> None:
        torch.save(voxel, save_path)

    @staticmethod
    def process_file(
        tif_file: Path, base_path: Path, dtype: str
    ) -> Tuple[np.ndarray, dict]:
        file_name = tif_file.name
        stack = DataExtractor.extract_stack(file_name)
        path = base_path.joinpath(tif_file)
        metadata = {"stack": stack, "path": str(path)}
        image_data = DataExtractor.extract_tif_file_image(path, dtype=dtype)
        return image_data, metadata

    @staticmethod
    def extract_tif_file_image(file_name: Path, dtype: str) -> np.ndarray:
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
                        path=metadata["path"],
                        stack=metadata["stack"],
                        wavelength=metadata["wavelength"],
                    )
                    voxel = Voxel(
                        data=VoxelData(
                            values=voxel_data,
                            position=np.array([z_idx, y_idx, x_idx]),
                        ),
                        metadata=voxel_metadata,
                    )
                    yield voxel


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Prepare data for lattice microscopy data into .pth format."
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
    parser.add_argument("--save_path", type=Path, help="Path to save the .pth files.")
    parser.add_argument(
        "--found_experiments_limit",
        type=int,
        default=1,
        help="Max number of experiments to extract.",
    )

    parser.add_argument(
        "--min_tif_files",
        type=int,
        default=20,
        help="Minimum number of tif files to consider an experiment valid.",
    )

    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max number of workers."
    )

    args = parser.parse_args()

    data_extractor = DataExtractor(
        z_voxel_size=args.z_voxel_size,
        y_voxel_size=args.y_voxel_size,
        x_voxel_size=args.x_voxel_size,
        found_experiments_limit=args.found_experiments_limit,
        min_tif_files=args.min_tif_files,
        max_workers=args.max_workers,
    )

    data_extractor(
        parent_directory=args.parent_directory,
        save_path=args.save_path,
    )
