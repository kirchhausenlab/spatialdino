from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LR_FEATS_DIRNAME = "lr_feats"
RAW_DIRNAME = "raw"
TMP_DIRNAME = "tmp"
HR_FEATS_DIRNAME = "hr_feats"
SEG_VORONOI_DIRNAME = "seg_voronoi"
SEG_PROBMAP_DIRNAME = "seg_probmap"
PROBMAP_DIRNAME = "probmap"
PROBMAP_DENSITIES_FILENAME = "probmap_densities.npz"
TRACKS_FILENAME = "tracks.csv"
NORM_PER_VOL_FILENAME = "norm_per_vol.txt"
PCA_DIR_RE = re.compile(r"^pca_(\d+)$")


@dataclass(frozen=True)
class InferenceTimepointPaths:
    name: str
    lr_path: Path
    raw_path: Path


def natural_sort_key(value: str) -> list[Any]:
    parts = re.split(r"(\d+)", value)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def timepoint_name_for_input(path: str | Path) -> str:
    return Path(path).stem


def inference_lr_feats_dir(root: str | Path) -> Path:
    return Path(root) / LR_FEATS_DIRNAME


def inference_raw_dir(root: str | Path) -> Path:
    return Path(root) / RAW_DIRNAME


def inference_tmp_dir(root: str | Path) -> Path:
    return Path(root) / TMP_DIRNAME


def inference_lr_feats_path(root: str | Path, timepoint_name: str) -> Path:
    return inference_lr_feats_dir(root) / f"{timepoint_name}.npy"


def inference_raw_path(root: str | Path, timepoint_name: str) -> Path:
    return inference_raw_dir(root) / f"{timepoint_name}.tif"


def process_features_hr_timepoint_dir(root: str | Path, timepoint_name: str) -> Path:
    return Path(root) / HR_FEATS_DIRNAME / timepoint_name


def process_features_pca_dir(root: str | Path, n_components: int) -> Path:
    return Path(root) / f"pca_{int(n_components)}"


def segmentation_voronoi_dir(root: str | Path) -> Path:
    return Path(root) / SEG_VORONOI_DIRNAME


def segmentation_probmap_dir(root: str | Path) -> Path:
    return Path(root) / SEG_PROBMAP_DIRNAME


def probability_map_dir(root: str | Path) -> Path:
    return Path(root) / PROBMAP_DIRNAME


def probability_map_densities_path(root: str | Path) -> Path:
    return Path(root) / PROBMAP_DENSITIES_FILENAME


def tracks_csv_path(root: str | Path) -> Path:
    return Path(root) / TRACKS_FILENAME


def norm_per_vol_stats_path(root: str | Path) -> Path:
    return Path(root) / NORM_PER_VOL_FILENAME


def iter_visible_child_dirs(root: str | Path) -> list[Path]:
    directories: list[Path] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                directories.append(Path(entry.path))
    directories.sort(key=lambda path: natural_sort_key(path.name))
    return directories


def _build_named_file_map(
    directory: Path,
    *,
    suffixes: tuple[str, ...],
) -> dict[str, Path]:
    names_to_paths: dict[str, Path] = {}
    if not directory.is_dir():
        return names_to_paths

    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if not entry.is_file():
                continue
            lowered = entry.name.lower()
            if not lowered.endswith(suffixes):
                continue
            path = Path(entry.path)
            name = path.stem
            existing = names_to_paths.get(name)
            if existing is not None:
                raise ValueError(
                    f"Duplicate files map to the same timepoint name {name!r}: {existing.name} and {path.name}."
                )
            names_to_paths[name] = path
    return names_to_paths


def discover_inference_timepoints(root: str | Path) -> list[InferenceTimepointPaths]:
    output_root = Path(root)
    lr_dir = inference_lr_feats_dir(output_root)
    raw_dir = inference_raw_dir(output_root)
    lr_paths = _build_named_file_map(lr_dir, suffixes=(".npy",))
    raw_paths = _build_named_file_map(raw_dir, suffixes=(".tif", ".tiff"))

    if not lr_paths and not raw_paths:
        raise ValueError(
            f"Input folder does not contain any timepoints under {LR_FEATS_DIRNAME}/ and {RAW_DIRNAME}/: {output_root}"
        )

    missing_lr = sorted(set(raw_paths) - set(lr_paths), key=natural_sort_key)
    if missing_lr:
        missing_name = missing_lr[0]
        raise FileNotFoundError(f"Missing feature file for {missing_name}: {missing_name}.npy.")

    missing_raw = sorted(set(lr_paths) - set(raw_paths), key=natural_sort_key)
    if missing_raw:
        missing_name = missing_raw[0]
        raise FileNotFoundError(f"Missing raw file for {missing_name}: {missing_name}.tif.")

    names = sorted(lr_paths, key=natural_sort_key)
    return [
        InferenceTimepointPaths(
            name=name,
            lr_path=lr_paths[name],
            raw_path=raw_paths[name],
        )
        for name in names
    ]


def list_process_features_output_paths(root: str | Path) -> list[Path]:
    output_root = Path(root)
    paths: list[Path] = []
    with os.scandir(output_root) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.name == HR_FEATS_DIRNAME or PCA_DIR_RE.fullmatch(entry.name):
                paths.append(Path(entry.path))
    paths.sort(key=lambda path: natural_sort_key(path.name))
    return paths


def inference_managed_output_paths(root: str | Path) -> list[Path]:
    output_root = Path(root)
    return [
        inference_lr_feats_dir(output_root),
        inference_raw_dir(output_root),
        inference_tmp_dir(output_root),
        norm_per_vol_stats_path(output_root),
    ]


def has_duplicate_timepoint_names(paths: Iterable[Path]) -> bool:
    seen: set[str] = set()
    for path in paths:
        name = path.stem
        if name in seen:
            return True
        seen.add(name)
    return False
