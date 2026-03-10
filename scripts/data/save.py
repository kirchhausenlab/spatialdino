import glob
import hashlib
import logging
import multiprocessing as mp
import re
import time
from argparse import ArgumentParser
from concurrent.futures import as_completed
from dataclasses import dataclass
from multiprocessing.managers import DictProxy
from pathlib import Path
from typing import Any, Dict, Final, Generator, List, Set, Tuple, TypedDict, Union
from spatialdino.data import EMPTY_VOXEL_THRESHOLD
import numpy as np
import webdataset as wds
from loky import get_reusable_executor
from skimage import io
from tqdm.auto import tqdm

from spatialdino.data import (
    Image2D,
    Image2DData,
    Image3D,
    Image3DData,
    Metadata2D,
    Metadata3D,
    ZFlattenMode,
    utils,
)

IMG_DTYPE: Final[str] = "uint8"
POS_DTYPE: Final[str] = "uint32"
EXPERIMENTS_TXT_NAME: Final[str] = "experiments.txt"


logger = logging.getLogger(__name__)
# Add these lines
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Experiment(TypedDict):
    tif_files: List[Path]
    name: str
    path: Path
    metadata: Dict[str, Any]


@dataclass
class DataExtractor:
    save_2d: bool
    save_3d: bool
    auto_crop: bool
    z_flatten_mode: ZFlattenMode
    z_chunk_size: int = 32
    y_chunk_size: int = 224
    x_chunk_size: int = 224
    dz: float = 1.0
    dy: float = 1.0
    dx: float = 1.0
    found_experiments_limit: int = 1
    min_tif_files: int = 20
    max_workers: int = 16
    experiment_filter_pattern: re.Pattern = re.compile(".*")
    k: int = 4
    lower_percentile: float = 99.75
    upper_percentile: float = 99.9
    min_bbox_ratio: float = 0.3
    max_bbox_ratio: float = 0.6
    median_filter_size: int = 3

    def __post_init__(self):
        if self.save_2d and self.save_3d:
            raise ValueError("Only one of save_2d or save_3d should be True.")
        if not self.save_2d and not self.save_3d:
            raise ValueError("Either save_2d or save_3d must be True.")

        # Validate chunk sizes
        if self.z_chunk_size <= 0 and self.z_chunk_size != -1:
            raise ValueError(f"Invalid z_chunk_size: {self.z_chunk_size}")
        if self.y_chunk_size <= 0 and self.y_chunk_size != -1:
            raise ValueError(f"Invalid y_chunk_size: {self.y_chunk_size}")
        if self.x_chunk_size <= 0 and self.x_chunk_size != -1:
            raise ValueError(f"Invalid x_chunk_size: {self.x_chunk_size}")

        # Validate other parameters
        if self.max_workers <= 0:
            raise ValueError(f"Invalid max_workers: {self.max_workers}")
        if self.min_tif_files <= 0:
            raise ValueError(f"Invalid min_tif_files: {self.min_tif_files}")
        if self.found_experiments_limit <= 0:
            raise ValueError(
                f"Invalid found_experiments_limit: {self.found_experiments_limit}"
            )

        if self.save_2d:
            self.use_deskewed_data = False
        elif self.save_3d:
            self.use_deskewed_data = True

    @staticmethod
    def extract_experiments(
        data_paths: List[Path],
        found_experiments_limit: int,
        min_tif_files: int,
        use_deskewed_data: bool,
        experiment_filter_pattern: re.Pattern,
        existing_experiments: Set[str],
    ) -> List[Experiment]:
        experiments = DataExtractor._extract_experiments(
            data_paths=data_paths,
            found_experiments_limit=found_experiments_limit,
            use_deskewed_data=use_deskewed_data,
            min_tif_files=min_tif_files,
            experiment_filter_pattern=experiment_filter_pattern,
            existing_experiments=existing_experiments,
        )
        return experiments

    @staticmethod
    def _experiment_directory_matches_pattern(
        directory: str, pattern: re.Pattern
    ) -> bool:
        return re.search(pattern, directory) is not None

    @staticmethod
    def _extract_experiments(
        data_paths: List[Path],
        found_experiments_limit: int,
        use_deskewed_data: bool,
        min_tif_files: int,
        experiment_filter_pattern: re.Pattern,
        existing_experiments: Set[str],
    ) -> List[Experiment]:
        experiments = []
        for channel_dir in data_paths:
            if len(experiments) == found_experiments_limit:
                break

            if (
                not DataExtractor._experiment_directory_matches_pattern(
                    directory=str(channel_dir), pattern=experiment_filter_pattern
                )
                or not channel_dir.is_dir()
            ):
                continue

            if not experiment_filter_pattern.search(str(channel_dir)):
                continue

            experiment_name = DataExtractor.get_experiment_name(
                directory=str(channel_dir)
            )

            if experiment_name in existing_experiments:
                continue

            existing_experiments.add(experiment_name)

            if use_deskewed_data:
                search_pattern = "**/DS/*.tif"
            else:
                search_pattern = "*.tif"

            tif_files = [
                Path(tif_file)
                for tif_file in glob.iglob(
                    str(channel_dir.joinpath(search_pattern)),
                    recursive=True,
                )
            ]

            if len(tif_files) >= min_tif_files:
                metadata = DataExtractor.get_experiment_metadata(tif_files[0].stem)
                experiments.append(
                    Experiment(
                        tif_files=tif_files,
                        name=experiment_name,
                        path=channel_dir,
                        metadata=metadata,
                    )
                )

        return experiments

    @staticmethod
    def get_experiment_metadata(tif_file_name: str) -> Dict[str, str]:
        metadata_parts = tif_file_name.split("_")
        metadata = {}
        for part in metadata_parts:
            part_lowercase = part.lower()
            if "Cam" in part:
                metadata["camera"] = part
            elif "ch" in part_lowercase:
                metadata["channel"] = part
            elif "nm" in part_lowercase:
                metadata["wavelength"] = part
        return metadata

    @staticmethod
    def get_experiment_name(directory: str) -> str:
        """
        Create unique experiment name by hashing the directory path.
        """
        experiment_name = hashlib.md5(directory.encode()).hexdigest()
        return experiment_name

    def __call__(
        self, data_paths_txt: Path, save_path: Path, maxsize: int = int(1e9)
    ) -> None:
        start_time = time.perf_counter()
        save_path.mkdir(parents=True, exist_ok=True)
        if not data_paths_txt.exists():
            raise FileNotFoundError(f"Data paths file not found: {data_paths_txt}")
        data_paths = [Path(path) for path in data_paths_txt.read_text().splitlines()]
        assert len(data_paths) > 0, "No data paths found."
        experiments_txt_path = save_path.joinpath(EXPERIMENTS_TXT_NAME)
        existing_experiments = set()
        experiments_txt_path.parent.mkdir(parents=True, exist_ok=True)
        if experiments_txt_path.exists():
            with open(experiments_txt_path, "r") as file:
                for line in file:
                    existing_experiments.add(line.strip())

        experiments = DataExtractor.extract_experiments(
            data_paths=data_paths,
            found_experiments_limit=self.found_experiments_limit,
            min_tif_files=self.min_tif_files,
            use_deskewed_data=self.use_deskewed_data,
            experiment_filter_pattern=self.experiment_filter_pattern,
            existing_experiments=existing_experiments.copy(),
        )
        unique_experiments = list(set(experiment["name"] for experiment in experiments))

        logger.info("Processing %d experiments." % len(unique_experiments))
        dims = "2D" if self.save_2d else "3D"
        logger.info(f"Experiments will be saved as {dims} data.")

        total_files = sum(len(exp["tif_files"]) for exp in experiments)
        executor = get_reusable_executor(max_workers=self.max_workers)
        success_count = 0

        start_shard = len(glob.glob(str(save_path.joinpath("dataset-*.tar"))))
        logger.info(f"Starting from shard {start_shard}")

        sink = wds.ShardWriter(
            str(save_path.joinpath("dataset-%06d.tar")),
            maxsize=maxsize,
            start_shard=start_shard,
            verbose=0,
        )

        manager = mp.Manager()
        keys = manager.dict()

        try:
            with tqdm(total=total_files, desc="Processing tif files") as pbar:
                # Process experiments one at a time
                for experiment in experiments:
                    experiment_futures = []
                    experiment_names = set()
                    future_to_experiment_name = {}
                    # Submit all files for current experiment
                    for tif_file in experiment["tif_files"]:
                        future = executor.submit(
                            _data_iterator,
                            save_2d=self.save_2d,
                            save_3d=self.save_3d,
                            auto_crop=self.auto_crop,
                            tif_file=tif_file,
                            z_flatten_mode=self.z_flatten_mode,
                            z_chunk_size=self.z_chunk_size,
                            y_chunk_size=self.y_chunk_size,
                            x_chunk_size=self.x_chunk_size,
                            dz=self.dz,
                            dy=self.dy,
                            dx=self.dx,
                            wavelength=int(
                                experiment["metadata"]["wavelength"].split("nm")[0]
                            ),
                            experiment_path=experiment["path"],
                            experiment_name=experiment["name"],
                            experiment_metadata=experiment["metadata"],
                            keys=keys,
                            k=self.k,
                            lower_percentile=self.lower_percentile,
                            upper_percentile=self.upper_percentile,
                            min_bbox_ratio=self.min_bbox_ratio,
                            max_bbox_ratio=self.max_bbox_ratio,
                            median_filter_size=self.median_filter_size,
                        )
                        experiment_futures.append(future)
                        future_to_experiment_name[future] = experiment["name"]

                    # Wait for all files in current experiment to complete
                    for future in as_completed(experiment_futures):
                        experiment_name = future_to_experiment_name[future]
                        batch = []
                        try:
                            batch = future.result()
                            for item in batch:
                                sink.write(item)
                            if experiment_name not in experiment_names:
                                experiment_names.add(experiment_name)
                        except Exception as e:
                            logger.error(
                                "Failed to process %s with error: %s"
                                % (experiment_name, e)
                            )
                            if experiment_name in experiment_names:
                                experiment_names.remove(experiment_name)
                        finally:
                            # Clear the future and its result
                            del future_to_experiment_name[future]
                            del batch
                            pbar.update(1)
                    with open(experiments_txt_path, "a") as experiment_names_file:
                        for experiment_name in experiment_names:
                            if experiment_name not in existing_experiments:
                                experiment_names_file.write(f"{experiment_name}\n")
                                existing_experiments.add(experiment_name)
                    success_count += len(experiment_names)
        finally:
            sink.close()
            executor.shutdown()
            end_time = time.perf_counter()
            failed_count = len(unique_experiments) - success_count
            logger.info("Total time taken: %f seconds" % (end_time - start_time))
            logger.info(
                "Successfully processed [%d / %d] experiments."
                % (success_count, len(unique_experiments))
            )
            logger.info(
                "Failed to process [%d / %d] experiments."
                % (failed_count, len(unique_experiments))
            )


def _data_iterator(
    save_2d: bool,
    save_3d: bool,
    auto_crop: bool,
    tif_file: Path,
    z_flatten_mode: ZFlattenMode,
    z_chunk_size: int,
    y_chunk_size: int,
    x_chunk_size: int,
    dz: float,
    dy: float,
    dx: float,
    wavelength: int,
    experiment_path: Path,
    experiment_name: str,
    experiment_metadata: Dict[str, str],
    keys: DictProxy,
    k: int = 4,
    lower_percentile: float = 99.75,
    upper_percentile: float = 99.9,
    min_bbox_ratio: float = 0.3,
    max_bbox_ratio: float = 0.6,
    median_filter_size: int = 3,
) -> List[Dict[str, Any]]:
    image_data, metadata = process_file(
        tif_file=tif_file,
        base_path=experiment_path,
    )
    batch = []
    try:
        metadata["wavelength"] = wavelength
        metadata.update(experiment_metadata)
        metadata.update({
            "dz": dz,
            "dy": dy,
            "dx": dx,
        })
        if save_2d:
            image_data = utils.convert_16bit_to_8bit(
                image_data, flatten_mode=z_flatten_mode
            )
            for stack, img_data in enumerate(image_data):
                chunks = chunk_2d_images(
                    image_data=utils.image_to_color_mode(image=img_data),
                    stack=stack,
                    base_metadata=metadata,
                    y_chunk_size=y_chunk_size,
                    x_chunk_size=x_chunk_size,
                )
                batch.extend(
                    _batch_chunks(
                        chunks=chunks,
                        experiment_name=experiment_name,
                        keys=keys,
                    )
                )
        elif save_3d:
            if auto_crop:
                chunks = chunk_3d_images_auto_crop(
                    image_data=image_data,
                    stack=-1,
                    base_metadata=metadata,
                    z_chunk_size=z_chunk_size,
                    y_chunk_size=y_chunk_size,
                    x_chunk_size=x_chunk_size,
                    k=k,
                    lower_percentile=lower_percentile,
                    upper_percentile=upper_percentile,
                    min_bbox_ratio=min_bbox_ratio,
                    max_bbox_ratio=max_bbox_ratio,
                    median_filter_size=median_filter_size,
                )
            else:
                chunks = chunk_3d_images(
                    image_data=image_data,
                    stack=-1,
                    base_metadata=metadata,
                    z_chunk_size=z_chunk_size,
                    y_chunk_size=y_chunk_size,
                    x_chunk_size=x_chunk_size,
                )
            if chunks is not None:
                batch.extend(
                    _batch_chunks(
                        chunks=chunks,
                        experiment_name=experiment_name,
                        keys=keys,
                    )
                )
        else:
            raise ValueError("Invalid save mode. Must be either save_2d or save_3d.")
    finally:
        del image_data
        return batch


def _batch_chunks(
    chunks: Generator[Union[Image2D, Image3D], None, None],
    experiment_name: str,
    keys: DictProxy,
) -> List[Dict[str, Any]]:
    batch = []
    for idx, chunk in enumerate(chunks):
        wavelength = chunk["metadata"]["wavelength"]
        frame = chunk["metadata"]["frame"]
        stack = chunk["metadata"]["stack"]
        suffix = (
            f"{frame}_{wavelength}_{idx}"
            if stack == -1
            else f"{frame}_{stack}_{wavelength}_{idx}"
        )
        key = f"{experiment_name}_{suffix}"

        if key in keys:
            continue

        keys[key] = True

        batch.append({
            "__key__": key,
            "values.npy": chunk["data"]["values"],
            "metadata.pth": chunk["metadata"],
        })
    return batch


def process_file(tif_file: Path, base_path: Path) -> Tuple[np.ndarray, dict]:
    file_name = tif_file.name
    frame = extract_frame(file_name)
    path = base_path.joinpath(tif_file)
    metadata = {"frame": frame, "path": str(path)}
    image_data = extract_tif_file_image(path)
    return image_data, metadata


def extract_tif_file_image(file_name: Path) -> np.ndarray:
    data = io.imread(
        file_name
    )  # (Z, Y, X), dtype = np.float32 if deskewed, uint16 if skewed
    return data


def extract_frame(file_name: str) -> int:
    return utils.extract_frame(file_name)


def chunk_2d_images(
    image_data: np.ndarray,
    stack: int,
    base_metadata: Dict[str, Any],
    y_chunk_size: int,
    x_chunk_size: int,
) -> Generator[Image2D, None, None]:
    if image_data.ndim != 3:
        raise ValueError(f"Expected 3D array (y,x,c), got shape {image_data.shape}")
    y, x, _ = image_data.shape

    # Adjust chunk sizes if -1 is provided
    y_chunk_size = y if y_chunk_size == -1 else y_chunk_size
    x_chunk_size = x if x_chunk_size == -1 else x_chunk_size

    metadata = base_metadata.copy()
    metadata["stack"] = stack
    for y_idx in range(0, y, y_chunk_size):
        for x_idx in range(0, x, x_chunk_size):
            y_end = min(y_idx + y_chunk_size, y)
            x_end = min(x_idx + x_chunk_size, x)
            img2d_data = image_data[y_idx:y_end, x_idx:x_end]
            if not np.any(img2d_data > EMPTY_VOXEL_THRESHOLD):
                continue
            metadata["shape"] = tuple(img2d_data.shape)
            metadata["dtype"] = str(img2d_data.dtype)
            img2d_metadata = Metadata2D(**metadata)
            image2d = Image2D(
                data=Image2DData(
                    values=img2d_data,
                    position=np.array([y_idx, x_idx], dtype=POS_DTYPE),
                ),
                metadata=img2d_metadata,
            )
            yield image2d


def chunk_3d_images_auto_crop(
    image_data: np.ndarray,
    base_metadata: Dict[str, Any],
    stack: int,
    z_chunk_size: int,
    y_chunk_size: int,
    x_chunk_size: int,
    k: int = 4,
    lower_percentile: float = 99.75,
    upper_percentile: float = 99.9,
    min_bbox_ratio: float = 0.3,
    max_bbox_ratio: float = 0.6,
    median_filter_size: int = 3,
) -> Generator[Image3D, None, None]:
    if image_data.ndim != 3:
        raise ValueError(f"Expected 3D array (z,y,x), got shape {image_data.shape}")
    crop_params = utils.auto_extract_crop_params(
        image_data,
        k=k,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        min_bbox_ratio=min_bbox_ratio,
        max_bbox_ratio=max_bbox_ratio,
        median_filter_size=median_filter_size,
    )
    if crop_params is None:
        return

    metadata = base_metadata.copy()
    metadata["stack"] = stack
    N = crop_params.shape[0]
    for i in range(N):
        voxel_data = image_data[
            crop_params[i, 0] : crop_params[i, 1],
            crop_params[i, 2] : crop_params[i, 3],
            crop_params[i, 4] : crop_params[i, 5],
        ]
        metadata["shape"] = tuple(voxel_data.shape)
        metadata["dtype"] = str(voxel_data.dtype)
        # for compatibility purposes
        non_zero_stacks = np.any(voxel_data > EMPTY_VOXEL_THRESHOLD, axis=(-2, -1))
        metadata["non_zero_stacks"] = non_zero_stacks
        voxel_metadata = Metadata3D(**metadata)
        voxel = Image3D(
            data=Image3DData(
                values=voxel_data,
                position=np.array(
                    [crop_params[i, 0], crop_params[i, 2], crop_params[i, 4]],
                    dtype=POS_DTYPE,
                ),
            ),
            metadata=voxel_metadata,
        )
        yield voxel


def chunk_3d_images(
    image_data: np.ndarray,
    stack: int,
    base_metadata: Dict[str, Any],
    z_chunk_size: int,
    y_chunk_size: int,
    x_chunk_size: int,
) -> Generator[Image3D, None, None]:
    z, y, x = image_data.shape

    # Adjust chunk sizes if -1 is provided
    z_chunk_size = z if z_chunk_size == -1 else z_chunk_size
    y_chunk_size = y if y_chunk_size == -1 else y_chunk_size
    x_chunk_size = x if x_chunk_size == -1 else x_chunk_size

    metadata = base_metadata.copy()
    metadata["stack"] = stack
    for z_idx in range(0, z, z_chunk_size):
        for y_idx in range(0, y, y_chunk_size):
            for x_idx in range(0, x, x_chunk_size):
                z_end = min(z_idx + z_chunk_size, z)
                y_end = min(y_idx + y_chunk_size, y)
                x_end = min(x_idx + x_chunk_size, x)
                voxel_data = image_data[z_idx:z_end, y_idx:y_end, x_idx:x_end]
                metadata["shape"] = tuple(voxel_data.shape)
                metadata["dtype"] = str(voxel_data.dtype)
                non_zero_stacks = np.any(
                    voxel_data > EMPTY_VOXEL_THRESHOLD, axis=(-2, -1)
                )
                metadata["non_zero_stacks"] = non_zero_stacks
                voxel_metadata = Metadata3D(**metadata)
                voxel = Image3D(
                    data=Image3DData(
                        values=voxel_data,
                        position=np.array([z_idx, y_idx, x_idx], dtype=POS_DTYPE),
                    ),
                    metadata=voxel_metadata,
                )
                yield voxel


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Prepare data for lattice microscopy data into .pth format."
    )

    parser.add_argument("--save_2d", action="store_true", help="Save 2D data.")
    parser.add_argument("--save_3d", action="store_true", help="Save 3D data.")
    parser.add_argument(
        "--auto_crop",
        action="store_true",
        default=False,
        help="Whether to automatically crop the 3D data using k-means clustering.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of clusters to use for k-means clustering.",
    )
    parser.add_argument(
        "--lower_percentile",
        type=float,
        default=99.75,
        help="Lower percentile to use for k-means clustering.",
    )
    parser.add_argument(
        "--upper_percentile",
        type=float,
        default=99.9,
        help="Upper percentile to use for k-means clustering.",
    )
    parser.add_argument(
        "--min_bbox_ratio",
        type=float,
        default=0.3,
        help="Minimum bbox ratio to use for k-means clustering.",
    )
    parser.add_argument(
        "--max_bbox_ratio",
        type=float,
        default=0.6,
        help="Maximum bbox ratio to use for k-means clustering.",
    )
    parser.add_argument(
        "--median_filter_size",
        type=int,
        default=3,
        help="Size of the median filter to use for k-means clustering.",
    )
    parser.add_argument(
        "--z_flatten_mode",
        default=ZFlattenMode.NONE,
        type=ZFlattenMode,
        help="Z flatten mode. One of: MAX, MEAN. Only applies for 2D data, i.e., when save_2d is True.",
    )

    parser.add_argument(
        "--dz",
        type=float,
        default=1.0,
        help="Spacing between z voxels, defaults to 1.0. Only applies for 3D data, i.e., when save_3d is True.",
    )
    parser.add_argument(
        "--dy",
        type=float,
        default=1.0,
        help="Spacing between y voxels, defaults to 1.0. Only applies for 3D data, i.e., when save_3d is True.",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=1.0,
        help="Spacing between x voxels, defaults to 1.0. Only applies for 3D data, i.e., when save_3d is True.",
    )

    parser.add_argument(
        "--z_chunk_size",
        type=int,
        default=-1,
        help="Z voxel size for chunking. Only applies for 3D data, i.e., when save_3d is True. Defaults to -1, i.e., no chunking.",
    )
    parser.add_argument(
        "--y_chunk_size",
        type=int,
        default=-1,
        help="Y voxel size for chunking. Only applies for 3D data, i.e., when save_3d is True. Defaults to -1, i.e., no chunking.",
    )
    parser.add_argument(
        "--x_chunk_size",
        type=int,
        default=-1,
        help="X voxel size for chunking. Only applies for 3D data, i.e., when save_3d is True. Defaults to -1, i.e., no chunking.",
    )
    parser.add_argument(
        "--data_paths_txt",
        type=Path,
        required=True,
        help="Path to a text file containing the paths to the data directories. The text file should contain one path per line.",
    )
    parser.add_argument("--save_path", type=Path, help="Path to save the .pth files.")
    parser.add_argument(
        "--found_experiments_limit",
        type=int,
        default=5,
        help="Max number of experiments to extract.",
    )

    parser.add_argument(
        "--min_tif_files",
        type=int,
        default=20,
        help="Minimum number of tif files to consider an experiment valid.",
    )
    parser.add_argument(
        "--use_deskewed_data",
        default=True,
        action="store_true",
        help="Whether to use deskewed data.",
    )
    parser.add_argument(
        "--experiment_filter_pattern",
        type=str,
        default=".*",
        help="Additional filter to match experiments of interest. Should be a regex pattern. Matches against the full absolute path of the experiment directory. By default, all found experiments are included.",
    )

    parser.add_argument(
        "--max_workers", type=int, default=10, help="Max number of workers."
    )

    args = parser.parse_args()

    data_extractor = DataExtractor(
        save_2d=args.save_2d,
        save_3d=args.save_3d,
        auto_crop=args.auto_crop,
        z_flatten_mode=args.z_flatten_mode,
        z_chunk_size=args.z_chunk_size,
        y_chunk_size=args.y_chunk_size,
        x_chunk_size=args.x_chunk_size,
        dz=args.dz,
        dy=args.dy,
        dx=args.dx,
        found_experiments_limit=args.found_experiments_limit,
        min_tif_files=args.min_tif_files,
        max_workers=args.max_workers,
        experiment_filter_pattern=re.compile(args.experiment_filter_pattern),
        k=args.k,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        min_bbox_ratio=args.min_bbox_ratio,
        max_bbox_ratio=args.max_bbox_ratio,
        median_filter_size=args.median_filter_size,
    )

    data_extractor(
        data_paths_txt=args.data_paths_txt,
        save_path=args.save_path,
    )
