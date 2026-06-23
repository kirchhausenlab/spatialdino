from __future__ import annotations

import base64
import html
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Literal, TextIO
import urllib.request
import uuid
import zipfile

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import tifffile

from spatialdino_server.fs_api import router as fs_router
from spatialdino_server.fs_roots import _configured_fs_roots_from_env
from spatialdino_server import jobs_api
from spatialdino_server.jobs_api import router as jobs_router
from spatialdino_server.status import get_cpu_activity, get_nvidia_gpu_memory
from spatialdino.inference.input_files import list_tiff_paths
from spatialdino.inference.output_layout import (
    TRACKS_FILENAME,
    discover_inference_timepoints,
    has_duplicate_timepoint_names,
    inference_raw_dir,
    inference_lr_feats_path,
    inference_managed_output_paths,
    natural_sort_key,
    norm_per_vol_stats_path as inference_norm_per_vol_stats_path,
    probability_map_dir,
    probability_map_densities_path,
)


JOB_LOGS_DIRNAME = ".spatialdino_job_logs"
RAW_DATA_RELATIVE_PATH = "data/raw_data"
DEFAULT_INFERENCE_BACKBONE_FILENAME = "backbone.pth"
DEFAULT_INFERENCE_BACKBONE_RELATIVE_PATH = f"models/{DEFAULT_INFERENCE_BACKBONE_FILENAME}"
DEFAULT_INFERENCE_BACKBONE_URL = (
    "https://spatialdino.s3.us-east-1.amazonaws.com/models/spatial_dino/step%3D244999/backbone.pth"
)
INFERENCE_BACKBONE_MODEL_LEARNED = "learned"
INFERENCE_BACKBONE_MODEL_NOPE = "nope"
INFERENCE_BACKBONE_MODEL_ROPE = "rope"
INFERENCE_PADDING_MODES = {"reflect", "replicate", "edge", "constant"}
INFERENCE_BACKBONE_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    INFERENCE_BACKBONE_MODEL_LEARNED: {
        "pos_embed_type": "learned",
        "num_register_tokens": 0,
        "num_tt_register_tokens": 1,
        "ffn_layer": "swiglufused",
    },
    INFERENCE_BACKBONE_MODEL_NOPE: {
        "pos_embed_type": "none",
        "num_register_tokens": 0,
        "num_tt_register_tokens": 1,
        "ffn_layer": "mlp",
    },
    INFERENCE_BACKBONE_MODEL_ROPE: {
        "pos_embed_type": "rope",
        "num_register_tokens": 4,
        "num_tt_register_tokens": 0,
        "rope_theta": 200,
        "rope_normalize_coords": True,
        "rope_coord_shift": 0.15,
        "rope_coord_jitter": 1.3,
        "rope_coord_rescale": 1.5,
        "rope_drop_prob": 0.1,
        "ffn_layer": "swiglufused",
    },
}
DEFAULT_PUBLIC_DATA_MANIFEST_URL = (
    "https://spatialdino.s3.us-east-1.amazonaws.com/inference_data/raw_data/manifest.json"
)
PUBLIC_DATA_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_default_inference_backbone_download_lock = threading.Lock()
PROBABILITY_MAP_PREVIEW_CACHE_MAX_ITEMS = 16
_probability_map_preview_cache_lock = threading.Lock()
_probability_map_preview_cache: dict[tuple[Any, ...], dict[str, Any]] = {}


def get_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "apps" / "web").is_dir():
            return parent
    parents = list(here.parents)
    if len(parents) >= 4:
        return parents[3]
    return Path.cwd()


def get_dist_dir() -> Path:
    override = os.environ.get("SPATIALDINO_DIST_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (get_repo_root() / "apps" / "web" / "dist").resolve()


def get_job_logs_dir() -> Path:
    override = os.environ.get("SPATIALDINO_JOB_LOG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (get_repo_root() / JOB_LOGS_DIRNAME).resolve()


def get_public_data_dir() -> Path:
    return (get_repo_root() / RAW_DATA_RELATIVE_PATH).resolve()


def get_public_data_manifest_url() -> str:
    value = (os.environ.get("SPATIALDINO_DATA_MANIFEST_URL") or "").strip()
    return value or DEFAULT_PUBLIC_DATA_MANIFEST_URL


def get_server_hostname() -> str:
    value = os.environ.get("SPATIALDINO_SERVER_HOSTNAME") or os.environ.get("SERVER_HOSTNAME")
    if value:
        return value
    return socket.gethostname()


def _normalize_ip_text(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    if "%" in text:
        text = text.split("%", 1)[0]

    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return text.lower()

    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return str(ip.ipv4_mapped)
    return str(ip)


@lru_cache(maxsize=1)
def get_server_ip_addresses() -> frozenset[str]:
    addresses = {"127.0.0.1", "::1"}
    hostnames = {socket.gethostname(), socket.getfqdn(), "localhost"}
    configured = os.environ.get("SPATIALDINO_SERVER_HOSTNAME") or os.environ.get("SERVER_HOSTNAME")
    if configured:
        hostnames.add(configured)

    for hostname in hostnames:
        try:
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            continue

        for _family, _socktype, _proto, _canonname, sockaddr in infos:
            normalized = _normalize_ip_text(sockaddr[0])
            if normalized:
                addresses.add(normalized)

    for family, target in ((socket.AF_INET, ("8.8.8.8", 80)), (socket.AF_INET6, ("2001:4860:4860::8888", 80, 0, 0))):
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as sock:
                sock.connect(target)
                normalized = _normalize_ip_text(sock.getsockname()[0])
                if normalized:
                    addresses.add(normalized)
        except OSError:
            continue

    return frozenset(addresses)


def classify_client_session(client_host: str | None) -> str:
    normalized = _normalize_ip_text(client_host)
    if not normalized:
        return "Unknown"

    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return "Unknown"

    if normalized in get_server_ip_addresses() or ip.is_loopback:
        return "Local"
    if ip.is_private or ip.is_link_local:
        return "Local network"
    return "Remote"


def list_backbone_weights() -> list[dict[str, str]]:
    repo_root = get_repo_root()
    models_dir = get_backbone_weights_dir()
    if not models_dir.is_dir():
        return []

    files: set[Path] = set()
    for pattern in ("*.pt", "*.pth"):
        files.update(path.resolve() for path in models_dir.glob(pattern) if path.is_file())

    return [
        {
            "label": path.name,
            "value": str(path.relative_to(repo_root)),
        }
        for path in sorted(files, key=lambda item: (item.name.casefold(), item.name))
    ]


def get_backbone_weights_dir() -> Path:
    return (get_repo_root() / "models").resolve()


def get_default_inference_backbone_path() -> Path:
    return (get_backbone_weights_dir() / DEFAULT_INFERENCE_BACKBONE_FILENAME).resolve()


def _download_url_to_file(url: str, target_path: Path) -> None:
    temp_path = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _validate_public_data_dataset_name(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        raise ValueError("Dataset name cannot be empty.")
    if value in {".", ".."}:
        raise ValueError(f"Invalid dataset name: {value!r}")
    if not PUBLIC_DATA_DATASET_NAME_RE.fullmatch(value):
        raise ValueError(
            f"Invalid dataset name: {value!r}. Only letters, digits, '.', '_' and '-' are supported."
        )
    return value


def _load_public_data_manifest() -> list[dict[str, str]]:
    manifest_url = get_public_data_manifest_url()
    try:
        with urllib.request.urlopen(manifest_url, timeout=60) as response:
            payload = json.load(response)
    except Exception as exc:
        raise RuntimeError(f"Could not load public data manifest from {manifest_url}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Public data manifest must be a JSON object.")

    datasets_payload = payload.get("datasets")
    if not isinstance(datasets_payload, list):
        raise RuntimeError("Public data manifest is missing a 'datasets' list.")

    datasets: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for item in datasets_payload:
        if not isinstance(item, dict):
            raise RuntimeError("Each manifest dataset entry must be an object.")

        name_raw = item.get("name")
        archive_url_raw = item.get("archiveUrl")
        if not isinstance(name_raw, str) or not isinstance(archive_url_raw, str):
            raise RuntimeError("Each manifest dataset entry must include string 'name' and 'archiveUrl' fields.")

        try:
            name = _validate_public_data_dataset_name(name_raw)
        except ValueError as exc:
            raise RuntimeError(f"Invalid dataset name in public data manifest: {exc}") from exc

        archive_url = archive_url_raw.strip()
        if not archive_url.startswith(("http://", "https://")):
            raise RuntimeError(f"Manifest dataset {name!r} has an invalid archiveUrl.")
        if name in seen_names:
            raise RuntimeError(f"Duplicate dataset name in public data manifest: {name!r}")

        seen_names.add(name)
        datasets.append({"name": name, "archiveUrl": archive_url})

    if not datasets:
        raise RuntimeError("Public data manifest contains no datasets.")
    return datasets


class ValidateInferenceInputRequest(BaseModel):
    path: str = Field(..., min_length=1)


class ValidateProcessFeaturesInputRequest(BaseModel):
    path: str = Field(..., min_length=1)


class ValidateTrackingInputRequest(BaseModel):
    path: str = Field(..., min_length=1)


class ValidateTrackingSegmentationFolderRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    segmentation_path: str = Field(..., min_length=1)


class DownloadBackboneWeightsRequest(BaseModel):
    overwrite: bool = False


class RunDataDownloadRequest(BaseModel):
    datasets: list[str] = Field(default_factory=list)
    existing_mode: Literal["skip", "overwrite"] | None = None


class InferenceAxisRequest(BaseModel):
    x: float | None = None
    y: float | None = None
    z: float | None = None


class InferenceCropBoundsRequest(BaseModel):
    x_start: int | None = None
    x_end: int | None = None
    y_start: int | None = None
    y_end: int | None = None
    z_start: int | None = None
    z_end: int | None = None


class InferenceFileRangeRequest(BaseModel):
    start: int | None = None
    end: int | None = None


class RunInferenceRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    output_path: str = Field(..., min_length=1)
    backbone_weight: str = Field("", min_length=0)
    backbone_model: str = Field(INFERENCE_BACKBONE_MODEL_NOPE, min_length=1)
    gpu_indices: list[int] = Field(default_factory=list)
    upsample_factor: float | InferenceAxisRequest | None = None
    route: str = Field("full", min_length=1)
    precision: str = Field("bfloat16", min_length=1)
    padding_mode: str = Field("reflect", min_length=1)
    crop_bounds: InferenceCropBoundsRequest = Field(default_factory=InferenceCropBoundsRequest)
    anisotropy: InferenceAxisRequest = Field(default_factory=InferenceAxisRequest)
    file_range: InferenceFileRangeRequest = Field(default_factory=InferenceFileRangeRequest)
    normalization_mode: str = Field("per_volume", min_length=1)
    global_hist_min: float | None = None
    global_hist_max: float | None = None
    overwrite: bool = False


class RunProcessFeaturesRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    output_path: str = Field(..., min_length=1)
    gpu_index: int | None = None
    file_range: InferenceFileRangeRequest = Field(default_factory=InferenceFileRangeRequest)
    save_high_resolution_features: bool = False
    high_resolution_save_format: str = Field(".tif", min_length=1)
    save_pca: bool = False
    pca_components: int = Field(3, ge=1)
    pca_save_format: str = Field(".tif", min_length=1)
    global_pca: bool = True


class RunSegmentationRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    output_path: str | None = None
    densities_path: str | None = None
    gpu_index: int | None = None
    mode: str = Field("voronoi_otsu", min_length=1)
    enable_voronoi_otsu: bool = True
    gaussian_blur_sigma: int = Field(3, ge=0)
    rolling_ball_radius: float = Field(10.0, ge=0.0)
    run_density_estimation: bool = True
    run_stage_2: bool = True
    training_timepoint: str | None = None
    seg_tif: str | None = None
    valid_mask_tif: str | None = None
    density_method: str = Field("gpu-hist", min_length=1)
    feature_batch: int = Field(32, ge=1)
    kde_points: int = Field(512, ge=2)
    kde_max_samples: int = Field(200000, ge=1)
    kde_bandwidth: float | None = Field(None, gt=0.0)
    hist_sigma_bins: float = Field(1.5, gt=0.0)
    bg_prob_threshold: float = Field(0.4, ge=0.0, le=1.0)
    fg_prob_threshold: float = Field(0.95, ge=0.0, le=1.0)
    probmap_threshold: float = Field(0.5, ge=0.0, le=1.0)
    run_connected_components: bool = True
    seed: int = 1337


class ProbabilityMapPreviewMetadataRequest(BaseModel):
    input_path: str = Field(..., min_length=1)


class ProbabilityMapPreviewImageRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    timepoint: str = Field(..., min_length=1)
    view: Literal["slice", "max_projection"] = "slice"
    z_index: int | None = Field(None, ge=0)


class RunForegroundProbabilityMapRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    output_path: str | None = None
    densities_path: str | None = None
    gpu_index: int | None = None
    run_density_estimation: bool = True
    training_timepoint: str | None = None
    seg_tif: str | None = None
    valid_mask_tif: str | None = None
    density_method: str = Field("gpu-hist", min_length=1)
    feature_batch: int = Field(32, ge=1)
    kde_points: int = Field(512, ge=2)
    kde_max_samples: int = Field(200000, ge=1)
    kde_bandwidth: float | None = Field(None, gt=0.0)
    hist_sigma_bins: float = Field(1.5, gt=0.0)
    seed: int = 1337


class RunTrackingRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    segmentation_path: str = Field(..., min_length=1)
    output_path: str = Field(..., min_length=1)
    output_filename: str = Field(TRACKS_FILENAME, min_length=1)
    max_distance_xy: float = Field(35.0, gt=0.0)
    max_distance_z: float = Field(15.0, gt=0.0)
    z_distance_weight: float = Field(2.5, gt=0.0)
    min_distance_to_remove_cand: float = Field(0.0, ge=0.0)
    vote_thresholds: str = Field("360,340,320,300", min_length=0)
    dice_threshold: float = Field(0.5, ge=0.0, le=1.0)
    corr_threshold: float = Field(0.5, ge=-1.0, le=1.0)
    invert_z: bool = False
    save_extended_results: bool = False
    ignore_features: bool = False
    disable_centroid_fallback: bool = False
    aggressive_feature_matching: bool = False
    min_feature_votes: int = Field(1, ge=1)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_allowed_inference_path(raw_path: str) -> Path:
    if "\x00" in raw_path:
        raise HTTPException(status_code=400, detail="Invalid path.")

    configured = Path(raw_path).expanduser()
    if not configured.is_absolute():
        raise HTTPException(status_code=400, detail="Path must be absolute.")

    resolved = Path(os.path.realpath(configured))
    roots = _configured_fs_roots_from_env().roots
    if roots and not any(_is_relative_to(resolved, root) for root in roots):
        raise HTTPException(status_code=403, detail="Path is outside configured filesystem roots.")
    return resolved


def _invalid_inference_input(reason_code: str, message: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reasonCode": reason_code,
        "message": message,
    }


def _shape_xyz(shape: tuple[int, ...]) -> dict[str, int]:
    return {
        "x": int(shape[2]),
        "y": int(shape[1]),
        "z": int(shape[0]),
    }


def _validate_inference_crop_params(
    crop_params: tuple[int, int, int, int, int, int],
    raw_volume_shape: tuple[int, int, int],
) -> None:
    start_z, end_z, start_y, end_y, start_x, end_x = crop_params
    assert start_z < end_z, "start_z must be less than end_z"
    assert start_y < end_y, "start_y must be less than end_y"
    assert start_x < end_x, "start_x must be less than end_x"
    assert start_z < raw_volume_shape[0], "start_z must be less than the number of z-stacks"
    assert start_y < raw_volume_shape[1], "start_y must be less than the number of y-stacks"
    assert start_x < raw_volume_shape[2], "start_x must be less than the number of x-stacks"
    assert end_z <= raw_volume_shape[0], "end_z must be less than or equal to the number of z-stacks"
    assert end_y <= raw_volume_shape[1], "end_y must be less than or equal to the number of y-stacks"
    assert end_x <= raw_volume_shape[2], "end_x must be less than or equal to the number of x-stacks"


def _list_inference_tiff_paths(input_path: Path) -> list[Path]:
    return [path.resolve() for path in list_tiff_paths(input_path)]


def _selected_inference_tiff_paths(input_path: Path, file_start: int, file_end: int | None) -> list[Path]:
    return _list_inference_tiff_paths(input_path)[file_start:file_end]


def _list_child_directories(input_path: Path) -> list[Path]:
    child_dirs: list[Path] = []
    with os.scandir(input_path) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue
            child_dirs.append(Path(entry.path))

    child_dirs.sort(key=lambda path: (path.name.casefold(), path.name))
    return child_dirs


def _post_processing_timepoints_or_error(raw_path: str) -> tuple[dict[str, Any] | None, list[str] | None]:
    input_path = _resolve_allowed_inference_path(raw_path)

    if not input_path.exists():
        return _invalid_process_features_input("missing", "Input folder does not exist."), None
    if not input_path.is_dir():
        return _invalid_process_features_input("not_directory", "Input path is not a folder."), None

    try:
        timepoints = discover_inference_timepoints(input_path)
    except FileNotFoundError as exc:
        return _invalid_process_features_input("missing_required_files", str(exc)), None
    except ValueError as exc:
        message = str(exc)
        if "does not contain any timepoints" in message:
            return _invalid_process_features_input(
                "missing_required_files",
                "Input folder must contain matching lr_feats/*.npy and raw/*.tif files.",
            ), None
        return _invalid_process_features_input("missing_required_files", message), None

    return None, [timepoint.name for timepoint in timepoints]


def _invalid_process_features_input(reason_code: str, message: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reasonCode": reason_code,
        "message": message,
    }


def _parse_tracking_vote_thresholds(raw_value: str) -> tuple[int, ...] | None:
    text = (raw_value or "").strip()
    if not text:
        return None

    thresholds: list[int] = []
    for token in text.split(","):
        value = token.strip()
        if not value:
            continue
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("Vote thresholds must be positive integers.")
        thresholds.append(parsed)
    return tuple(thresholds) if thresholds else None


def _normalize_tracking_output_filename(raw_value: str | None) -> str:
    text = TRACKS_FILENAME if raw_value is None else raw_value.strip()
    if not text:
        raise ValueError("Output file name must not be empty.")
    path = Path(text)
    if path.name != text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError("Output file name must be a file name, not a path.")
    if path.suffix == "":
        text = f"{text}.csv"
        path = Path(text)
    if path.suffix.lower() != ".csv":
        raise ValueError("Output file name must end in .csv.")
    return text


def validate_process_features_input_folder(raw_path: str) -> dict[str, Any]:
    error, timepoint_names = _post_processing_timepoints_or_error(raw_path)
    if error is not None or timepoint_names is None:
        return error or _invalid_process_features_input("missing_required_files", "Invalid input folder.")

    subfolder_count = len(timepoint_names)
    return {
        "valid": True,
        "message": f"Valid feature folder. Found {subfolder_count} timepoint{'s' if subfolder_count != 1 else ''}.",
        "subfolderCount": subfolder_count,
        "subfolderNames": timepoint_names,
    }


def validate_segmentation_input_folder(raw_path: str) -> dict[str, Any]:
    validation = validate_process_features_input_folder(raw_path)
    if not validation["valid"]:
        return validation

    input_path = _resolve_allowed_inference_path(raw_path)
    densities_path = probability_map_densities_path(input_path)
    return {
        **validation,
        "probmapDensitiesPath": str(densities_path),
        "probmapDensitiesExists": densities_path.is_file(),
    }


def _named_tiff_file_map(directory: Path) -> dict[str, Path]:
    names_to_paths: dict[str, Path] = {}
    if not directory.is_dir():
        return names_to_paths

    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.startswith(".") or not entry.is_file():
                continue
            lowered = entry.name.lower()
            if not lowered.endswith((".tif", ".tiff")):
                continue
            path = Path(entry.path)
            name = path.stem
            existing = names_to_paths.get(name)
            if existing is not None:
                raise ValueError(
                    f"Duplicate TIFF files map to the same timepoint name {name!r}: "
                    f"{existing.name} and {path.name}."
                )
            names_to_paths[name] = path
    return names_to_paths


def validate_probability_map_input_folder(raw_path: str) -> dict[str, Any]:
    input_path = _resolve_allowed_inference_path(raw_path)
    if not input_path.exists():
        return _invalid_process_features_input("missing", "Input folder does not exist.")
    if not input_path.is_dir():
        return _invalid_process_features_input("not_directory", "Input path is not a folder.")

    raw_path_root = inference_raw_dir(input_path)
    if not raw_path_root.is_dir():
        return _invalid_process_features_input(
            "missing_raw_folder",
            f"Input folder is missing {raw_path_root.name}/.",
        )

    probmap_path = probability_map_dir(input_path)
    if not probmap_path.is_dir():
        return _invalid_process_features_input(
            "missing_probmap_folder",
            f"Input folder is missing {probmap_path.name}/.",
        )

    try:
        raw_paths = _named_tiff_file_map(raw_path_root)
        probmap_paths = _named_tiff_file_map(probmap_path)
    except ValueError as exc:
        return _invalid_process_features_input("duplicate_timepoint_names", str(exc))

    if not raw_paths:
        return _invalid_process_features_input("missing_required_files", f"{raw_path_root.name}/ contains no TIFF files.")
    if not probmap_paths:
        return _invalid_process_features_input("missing_required_files", f"{probmap_path.name}/ contains no TIFF files.")

    missing_probmap = sorted(set(raw_paths) - set(probmap_paths), key=natural_sort_key)
    if missing_probmap:
        missing_name = missing_probmap[0]
        return _invalid_process_features_input(
            "missing_required_files",
            f"Probability-map folder is missing {missing_name}.tif.",
        )

    missing_raw = sorted(set(probmap_paths) - set(raw_paths), key=natural_sort_key)
    if missing_raw:
        missing_name = missing_raw[0]
        return _invalid_process_features_input(
            "missing_required_files",
            f"Raw folder is missing {missing_name}.tif.",
        )

    subfolder_names = sorted(raw_paths, key=natural_sort_key)
    subfolder_count = len(subfolder_names)

    return {
        "valid": True,
        "message": f"Valid probability-map folder. Found {subfolder_count} timepoint{'s' if subfolder_count != 1 else ''}.",
        "subfolderCount": subfolder_count,
        "subfolderNames": subfolder_names,
        "rawPath": str(raw_path_root),
        "probmapPath": str(probmap_path),
        "probmapExists": True,
    }


def _shape_zyx_dict(shape: tuple[int, int, int]) -> dict[str, int]:
    return {"z": int(shape[0]), "y": int(shape[1]), "x": int(shape[2])}


def _read_tiff_zyx_shape(path: Path) -> tuple[int, int, int]:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path.name} does not contain a readable TIFF volume.")
        shape = tuple(int(dim) for dim in tif.series[0].shape)
    if len(shape) != 3:
        raise ValueError(f"{path.name} must be a 3D TIFF volume.")
    return shape  # type: ignore[return-value]


def _read_tiff_z_slice(path: Path, *, z_index: int) -> np.ndarray:
    shape = _read_tiff_zyx_shape(path)
    if z_index < 0 or z_index >= shape[0]:
        raise ValueError(f"Z plane {z_index} is outside the valid range 0..{shape[0] - 1}.")

    try:
        mapped = tifffile.memmap(path)
        if mapped.ndim == 3 and tuple(int(dim) for dim in mapped.shape) == shape:
            return np.asarray(mapped[z_index])
    except Exception:
        pass

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if len(series.pages) == shape[0]:
            return np.asarray(series.pages[z_index].asarray())
        return np.asarray(series.asarray()[z_index])


def _read_tiff_max_projection(path: Path, *, ignore_nan: bool = False) -> np.ndarray:
    shape = _read_tiff_zyx_shape(path)

    try:
        mapped = tifffile.memmap(path)
        if mapped.ndim == 3 and tuple(int(dim) for dim in mapped.shape) == shape:
            if ignore_nan:
                return np.asarray(np.fmax.reduce(mapped, axis=0))
            return np.asarray(np.max(mapped, axis=0))
    except Exception:
        pass

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if len(series.pages) == shape[0]:
            projection: np.ndarray | None = None
            for page in series.pages:
                plane = np.asarray(page.asarray())
                if projection is None:
                    projection = plane.copy()
                else:
                    if ignore_nan:
                        np.fmax(projection, plane, out=projection)
                    else:
                        np.maximum(projection, plane, out=projection)
            if projection is not None:
                return projection
        volume = series.asarray()
        if ignore_nan:
            return np.asarray(np.fmax.reduce(volume, axis=0))
        return np.asarray(np.max(volume, axis=0))


def _normalize_raw_preview(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return np.zeros(values.shape, dtype=np.uint8), 0.0, 0.0

    finite_values = values[finite]
    low = float(np.percentile(finite_values, 1.0))
    high = float(np.percentile(finite_values, 99.8))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))

    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8), low, high

    safe_values = np.nan_to_num(values, nan=low, neginf=low, posinf=high)
    scaled = np.clip((safe_values - low) / (high - low), 0.0, 1.0)
    return np.asarray(np.rint(scaled * 255.0), dtype=np.uint8), low, high


def _preview_array_base64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _file_cache_stamp(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), int(stat.st_size), int(stat.st_mtime_ns)


def _preview_cache_get(key: tuple[Any, ...]) -> dict[str, Any] | None:
    with _probability_map_preview_cache_lock:
        cached = _probability_map_preview_cache.get(key)
        if cached is None:
            return None
        _probability_map_preview_cache.pop(key)
        _probability_map_preview_cache[key] = cached
        return dict(cached)


def _preview_cache_put(key: tuple[Any, ...], value: dict[str, Any]) -> None:
    with _probability_map_preview_cache_lock:
        _probability_map_preview_cache[key] = dict(value)
        while len(_probability_map_preview_cache) > PROBABILITY_MAP_PREVIEW_CACHE_MAX_ITEMS:
            oldest_key = next(iter(_probability_map_preview_cache))
            _probability_map_preview_cache.pop(oldest_key, None)


def _probability_map_timepoint_paths(input_path: Path, timepoint_name: str) -> tuple[Path, Path]:
    try:
        raw_paths = _named_tiff_file_map(inference_raw_dir(input_path))
        probmap_paths = _named_tiff_file_map(probability_map_dir(input_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_path = raw_paths.get(timepoint_name)
    probmap_path = probmap_paths.get(timepoint_name)
    if raw_path is not None and probmap_path is not None:
        return raw_path, probmap_path

    raise HTTPException(status_code=404, detail=f"Unknown timepoint: {timepoint_name}.")


def probability_map_preview_metadata(raw_path: str) -> dict[str, Any]:
    validation = validate_probability_map_input_folder(raw_path)
    if not validation["valid"]:
        return validation

    input_path = _resolve_allowed_inference_path(raw_path)
    try:
        raw_paths = _named_tiff_file_map(inference_raw_dir(input_path))
        probmap_paths = _named_tiff_file_map(probability_map_dir(input_path))
    except ValueError as exc:
        return _invalid_process_features_input("missing_required_files", str(exc))

    preview_timepoints: list[dict[str, Any]] = []
    default_timepoint: str | None = None
    subfolder_names = [str(name) for name in validation.get("subfolderNames", [])]
    for timepoint_name in subfolder_names:
        raw_timepoint_path = raw_paths[timepoint_name]
        probmap_path = probmap_paths[timepoint_name]
        entry: dict[str, Any] = {"name": timepoint_name, "compatible": False}
        try:
            raw_shape = _read_tiff_zyx_shape(raw_timepoint_path)
            probmap_shape = _read_tiff_zyx_shape(probmap_path)
            compatible = raw_shape == probmap_shape
            entry.update(
                {
                    "rawShape": _shape_zyx_dict(raw_shape),
                    "probmapShape": _shape_zyx_dict(probmap_shape),
                    "shape": _shape_zyx_dict(raw_shape),
                    "zCount": int(raw_shape[0]),
                    "height": int(raw_shape[1]),
                    "width": int(raw_shape[2]),
                    "compatible": compatible,
                    "message": "Ready" if compatible else "Raw and probability-map TIFF shapes do not match.",
                }
            )
            if compatible and default_timepoint is None:
                default_timepoint = timepoint_name
        except Exception as exc:
            entry["message"] = str(exc)
        preview_timepoints.append(entry)

    return {
        **validation,
        "timepoints": preview_timepoints,
        "defaultTimepoint": default_timepoint or (preview_timepoints[0]["name"] if preview_timepoints else None),
    }


def probability_map_preview_image(payload: ProbabilityMapPreviewImageRequest) -> dict[str, Any]:
    validation = validate_probability_map_input_folder(payload.input_path)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["message"])

    input_path = _resolve_allowed_inference_path(payload.input_path)
    raw_path, probmap_path = _probability_map_timepoint_paths(input_path, payload.timepoint)
    raw_shape = _read_tiff_zyx_shape(raw_path)
    probmap_shape = _read_tiff_zyx_shape(probmap_path)
    if raw_shape != probmap_shape:
        raise HTTPException(status_code=400, detail="Raw and probability-map TIFF shapes do not match.")

    if payload.view == "slice":
        z_index = 0 if payload.z_index is None else int(payload.z_index)
        if z_index < 0 or z_index >= raw_shape[0]:
            raise HTTPException(status_code=400, detail=f"Z plane must be between 0 and {raw_shape[0] - 1}.")
    else:
        z_index = None

    cache_key = (
        "probability-map-preview-v1",
        _file_cache_stamp(raw_path),
        _file_cache_stamp(probmap_path),
        payload.view,
        z_index,
    )
    cached = _preview_cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        if payload.view == "slice":
            assert z_index is not None
            raw_image = _read_tiff_z_slice(raw_path, z_index=z_index)
            probability_image = _read_tiff_z_slice(probmap_path, z_index=z_index)
        else:
            raw_image = _read_tiff_max_projection(raw_path)
            probability_image = _read_tiff_max_projection(probmap_path, ignore_nan=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read preview TIFF data: {exc}") from exc

    if raw_image.ndim != 2 or probability_image.ndim != 2:
        raise HTTPException(status_code=400, detail="Preview data must be 2D.")
    if raw_image.shape != probability_image.shape:
        raise HTTPException(status_code=400, detail="Raw and probability-map preview shapes do not match.")

    raw_uint8, display_low, display_high = _normalize_raw_preview(raw_image)
    probability_float = np.asarray(probability_image, dtype=np.float32)
    probability_float = np.nan_to_num(probability_float, nan=-1.0, neginf=-1.0, posinf=1.0)
    height, width = (int(dim) for dim in raw_uint8.shape)
    response = {
        "timepoint": payload.timepoint,
        "view": payload.view,
        "zIndex": z_index,
        "width": width,
        "height": height,
        "shape": _shape_zyx_dict(raw_shape),
        "raw": {
            "dtype": "uint8",
            "data": _preview_array_base64(raw_uint8),
            "displayLow": display_low,
            "displayHigh": display_high,
        },
        "probability": {
            "dtype": "float32",
            "data": _preview_array_base64(probability_float),
        },
    }
    _preview_cache_put(cache_key, response)
    return response


def validate_tracking_input_folder(raw_path: str) -> dict[str, Any]:
    error, timepoint_names = _post_processing_timepoints_or_error(raw_path)
    if error is not None or timepoint_names is None:
        return error or _invalid_process_features_input("missing_required_files", "Invalid input folder.")

    subfolder_count = len(timepoint_names)
    if subfolder_count == 0:
        return _invalid_process_features_input("no_subfolders", "Input folder contains no valid timepoints.")
    if subfolder_count < 2:
        return _invalid_process_features_input(
            "insufficient_subfolders",
            "Tracking requires at least 2 timepoints.",
        )

    return {
        "valid": True,
        "message": f"Valid feature folder. Found {subfolder_count} timepoint{'s' if subfolder_count != 1 else ''}.",
        "subfolderCount": subfolder_count,
        "subfolderNames": timepoint_names,
    }


def validate_tracking_segmentation_folder(raw_input_path: str, raw_segmentation_path: str) -> dict[str, Any]:
    input_validation = validate_tracking_input_folder(raw_input_path)
    if not input_validation["valid"]:
        return input_validation

    segmentation_path = _resolve_allowed_inference_path(raw_segmentation_path)
    if not segmentation_path.exists():
        return _invalid_process_features_input("missing", "Segmentation folder does not exist.")
    if not segmentation_path.is_dir():
        return _invalid_process_features_input("not_directory", "Segmentation path is not a folder.")

    subfolder_names = [str(name) for name in input_validation.get("subfolderNames", [])]
    for subfolder_name in subfolder_names:
        mask_path = segmentation_path / f"{subfolder_name}.tif"
        if not mask_path.is_file():
            return _invalid_process_features_input(
                "missing_required_files",
                f"Segmentation folder is missing {mask_path.name}.",
            )

    subfolder_count = int(input_validation["subfolderCount"])
    return {
        "valid": True,
        "message": f"Valid segmentation folder. Found {subfolder_count} mask file{'s' if subfolder_count != 1 else ''}.",
        "subfolderCount": subfolder_count,
        "subfolderNames": subfolder_names,
    }


def validate_inference_input_folder(raw_path: str) -> dict[str, Any]:
    input_path = _resolve_allowed_inference_path(raw_path)

    if not input_path.exists():
        return _invalid_inference_input("missing", "Input folder does not exist.")
    if not input_path.is_dir():
        return _invalid_inference_input("not_directory", "Input path is not a folder.")

    tiff_paths = _list_inference_tiff_paths(input_path)
    if not tiff_paths:
        return _invalid_inference_input("no_tiff_files", "Input folder contains no .tif or .tiff files.")
    if has_duplicate_timepoint_names(tiff_paths):
        return _invalid_inference_input(
            "duplicate_timepoint_names",
            "Input TIFF files must have unique names after removing the extension.",
        )

    reference_shape: tuple[int, ...] | None = None
    reference_shape_xyz: dict[str, int] | None = None

    for tiff_path in tiff_paths:
        try:
            # Read TIFF metadata only; this validates shape without loading the volume into memory.
            with tifffile.TiffFile(tiff_path) as tif:
                if not tif.series:
                    return _invalid_inference_input(
                        "shape_mismatch",
                        f"{tiff_path.name} does not contain a readable 3D TIFF volume.",
                    )
                shape = tuple(int(dim) for dim in tif.series[0].shape)
        except Exception as exc:
            return _invalid_inference_input(
                "shape_mismatch",
                f"Could not read TIFF metadata from {tiff_path.name}: {exc}",
            )

        if len(shape) != 3:
            return _invalid_inference_input("shape_mismatch", f"{tiff_path.name} is not a 3D TIFF volume.")

        current_shape_xyz = _shape_xyz(shape)
        if reference_shape is None:
            reference_shape = shape
            reference_shape_xyz = current_shape_xyz
            continue

        if shape != reference_shape:
            expected = reference_shape_xyz or _shape_xyz(reference_shape)
            return _invalid_inference_input(
                "shape_mismatch",
                (
                    "TIFF files do not all have the same 3D shape. "
                    f"Expected x={expected['x']}, y={expected['y']}, z={expected['z']}."
                ),
            )

    return {
        "valid": True,
        "message": "Valid dataset.",
        "fileCount": len(tiff_paths),
        "shape": reference_shape_xyz,
    }


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SAVE_PATH_RE = re.compile(r"Saving to\s+(.+)$")
SAVED_FEATURES_RE = re.compile(r"Saved features to\s+(.+)$")
PROCESS_FEATURES_PROCESSING_RE = re.compile(
    r"^\[(?:process-features|segmentation|tracking|probmap)\] Processing (.+?) \((\d+)/(\d+)\)$"
)
PROCESS_FEATURES_COMPLETED_RE = re.compile(r"^\[(?:process-features|segmentation|tracking|probmap)\] Completed (.+)$")
TRACKING_MATCHING_RE = re.compile(r"^\[tracking\] Matching (.+?) -> (.+?) \((\d+)/(\d+)\)$")
TRACKING_MATCHED_RE = re.compile(r"^\[tracking\] Matched (.+?) -> (.+?) \((\d+)/(\d+)\)$")
DEFAULT_INFERENCE_OMP_NUM_THREADS = 4
NORM_PER_VOL_MIN_RE = re.compile(r"^Global hist min:\s*(.+?)\s*$", re.MULTILINE)
NORM_PER_VOL_MAX_RE = re.compile(r"^Global hist max:\s*(.+?)\s*$", re.MULTILINE)
NORMALIZATION_MODE_PER_VOLUME = "per_volume"
NORMALIZATION_MODE_GLOBAL_AUTO = "global_auto"
NORMALIZATION_MODE_GLOBAL_MANUAL = "global_manual"
PROCESS_FEATURES_SAVE_FORMATS = {".npy", ".tif"}
SEGMENTATION_MODE_VORONOI_OTSU = "voronoi_otsu"
SEGMENTATION_MODE_PROBABILITY_MAP = "probability_map"
SEGMENTATION_MODE_LEGACY_PROBABILITY_MAP = "legacy_probability_map"
SEGMENTATION_MODE_OPTIONS = {
    SEGMENTATION_MODE_VORONOI_OTSU,
    SEGMENTATION_MODE_PROBABILITY_MAP,
    SEGMENTATION_MODE_LEGACY_PROBABILITY_MAP,
}
PROBABILITY_MAP_DENSITY_METHODS = {"kde", "gpu-hist"}
NORMALIZATION_MODE_OPTIONS = {
    NORMALIZATION_MODE_PER_VOLUME,
    NORMALIZATION_MODE_GLOBAL_AUTO,
    NORMALIZATION_MODE_GLOBAL_MANUAL,
}


def _invalid_inference_run(reason_code: str, message: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reasonCode": reason_code,
        "message": message,
        "requiresOverwriteConfirmation": False,
    }


def _invalid_process_features_run(reason_code: str, message: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reasonCode": reason_code,
        "message": message,
    }


def _validate_requested_gpu(gpu_index: int | None) -> tuple[dict[str, Any] | None, int | None]:
    if gpu_index is None:
        return _invalid_process_features_run("missing_gpu_selection", "Select one GPU."), None

    requested_gpu = int(gpu_index)
    gpu_status = get_nvidia_gpu_memory()
    available_gpus = {int(gpu["index"]) for gpu in gpu_status.get("gpus", [])}
    if requested_gpu not in available_gpus:
        return _invalid_process_features_run("invalid_gpu_selection", "Selected GPU is not available on this server."), None
    return None, requested_gpu


def _validate_post_processing_output_path(
    raw_output_path: str | None,
    *,
    input_path: Path,
    label: str,
) -> tuple[dict[str, Any] | None, Path | None]:
    text = (raw_output_path or "").strip()
    if not text:
        return None, input_path
    output_path = _resolve_allowed_inference_path(raw_output_path)
    if output_path.exists() and not output_path.is_dir():
        return _invalid_process_features_run("output_not_directory", f"{label} path is not a folder."), None
    return None, output_path


def _inference_overwrite_confirmation(output_path: Path, count: int, preview: list[str]) -> dict[str, Any]:
    message = (
        "Inference-managed outputs already exist. Confirm overwrite to replace lr_feats/, raw/, tmp/, "
        "and norm_per_vol.txt while preserving other files in the folder."
    )
    return {
        "valid": True,
        "message": message,
        "requiresOverwriteConfirmation": True,
        "outputPath": str(output_path),
        "outputEntryCount": count,
        "outputEntriesPreview": preview,
    }


def _summarize_existing_inference_outputs(output_path: Path) -> tuple[int, list[str]]:
    existing_names: list[str] = []
    for managed_path in inference_managed_output_paths(output_path):
        if not managed_path.exists():
            continue
        suffix = "/" if managed_path.is_dir() else ""
        existing_names.append(f"{managed_path.name}{suffix}")
    return len(existing_names), existing_names[:10]


def _clear_inference_managed_outputs(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    for managed_path in inference_managed_output_paths(output_path):
        if managed_path.is_dir():
            shutil.rmtree(managed_path, ignore_errors=True)
        elif managed_path.exists():
            managed_path.unlink()


def _coerce_start(value: int | None) -> int:
    return 0 if value is None else value


def _coerce_script_crop_end(value: int | None) -> int:
    return 0 if value is None else value + 1


def _coerce_script_file_end(value: int | None) -> int | None:
    return None if value is None else value + 1


def _validate_inclusive_axis_bounds(start: int, end: int | None, *, size: int, axis_label: str) -> str | None:
    last_index = size - 1
    if start < 0 or start > last_index:
        return f"{axis_label} start must be between 0 and {last_index}."
    if end is not None and (end < 0 or end > last_index):
        return f"{axis_label} end must be between 0 and {last_index}."
    return None


def _build_process_features_launch_config(
    payload: RunProcessFeaturesRequest,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    input_validation = validate_process_features_input_folder(payload.input_path)
    if not input_validation["valid"]:
        return _invalid_process_features_run(input_validation["reasonCode"], input_validation["message"]), None

    if not payload.save_pca and not payload.save_high_resolution_features:
        return _invalid_process_features_run(
            "no_outputs_selected",
            "Choose at least one output: Save PCA and/or Save high-resolution features.",
        ), None

    if payload.gpu_index is None:
        return _invalid_process_features_run("missing_gpu_selection", "Select one GPU."), None

    requested_gpu = int(payload.gpu_index)
    gpu_status = get_nvidia_gpu_memory()
    available_gpus = {int(gpu["index"]) for gpu in gpu_status.get("gpus", [])}
    if requested_gpu not in available_gpus:
        return _invalid_process_features_run("invalid_gpu_selection", "Selected GPU is not available on this server."), None

    if payload.high_resolution_save_format not in PROCESS_FEATURES_SAVE_FORMATS:
        return _invalid_process_features_run(
            "invalid_high_resolution_format",
            "High-resolution feature save format must be one of: .npy, .tif.",
        ), None

    if payload.pca_save_format not in PROCESS_FEATURES_SAVE_FORMATS:
        return _invalid_process_features_run("invalid_pca_format", "PCA save format must be one of: .npy, .tif."), None

    input_path = _resolve_allowed_inference_path(payload.input_path)
    output_error, output_path = _validate_post_processing_output_path(
        payload.output_path,
        input_path=input_path,
        label="Output",
    )
    if output_error is not None:
        return output_error, None

    subfolder_count = int(input_validation["subfolderCount"])
    timepoint_names = list(input_validation.get("subfolderNames", []))
    file_start = _coerce_start(payload.file_range.start)
    file_end_inclusive = payload.file_range.end
    if file_start < 0 or file_start >= subfolder_count:
        return _invalid_process_features_run(
            "invalid_file_range",
            f"Start file must be between 0 and {subfolder_count - 1}.",
        ), None
    if file_end_inclusive is not None and (file_end_inclusive < 0 or file_end_inclusive >= subfolder_count):
        return _invalid_process_features_run(
            "invalid_file_range",
            f"End file must be between 0 and {subfolder_count - 1}.",
        ), None

    file_end = _coerce_script_file_end(file_end_inclusive)
    effective_file_end = subfolder_count if file_end is None else file_end
    if effective_file_end <= file_start:
        return _invalid_process_features_run("empty_file_selection", "Chosen files leave zero timepoints to process."), None

    selected_timepoint_names = timepoint_names[file_start:effective_file_end]
    selected_subfolder_count = len(selected_timepoint_names)
    return (
        {
            "valid": True,
            "message": "Validation passed.",
            "subfolderCount": selected_subfolder_count,
            "selectedFileCount": selected_subfolder_count,
        },
        {
            "input_path": input_path,
            "output_path": output_path,
            "gpu_index": requested_gpu,
            "save_pca": bool(payload.save_pca),
            "pca_components": int(payload.pca_components),
            "pca_save_format": payload.pca_save_format,
            "global_pca": bool(payload.global_pca),
            "save_high_resolution_features": bool(payload.save_high_resolution_features),
            "high_resolution_save_format": payload.high_resolution_save_format,
            "subfolder_count": selected_subfolder_count,
            "input_subfolder_count": subfolder_count,
            "file_start": file_start,
            "file_end": file_end,
            "selected_timepoint_names": selected_timepoint_names,
        },
    )


def _build_probability_map_density_config(
    payload: RunForegroundProbabilityMapRequest | RunSegmentationRequest,
    *,
    input_validation: dict[str, Any],
    output_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if payload.density_method not in PROBABILITY_MAP_DENSITY_METHODS:
        return _invalid_process_features_run(
            "invalid_density_method",
            "Probability-map density method must be one of: kde or gpu-hist.",
        ), None

    run_density_estimation = bool(payload.run_density_estimation)
    training_timepoint = (payload.training_timepoint or "").strip() or None
    subfolder_names = set(input_validation.get("subfolderNames", []))
    seg_tif_path: Path | None = None
    valid_mask_tif_path: Path | None = None

    if run_density_estimation:
        densities_path = probability_map_densities_path(output_path)
        if training_timepoint is None:
            return _invalid_process_features_run(
                "missing_training_timepoint",
                "Choose one training timepoint for Stage 1.",
            ), None
        if training_timepoint not in subfolder_names:
            return _invalid_process_features_run(
                "invalid_training_timepoint",
                "Selected Stage 1 training timepoint is not part of the validated input folder.",
            ), None
        seg_tif_raw = (payload.seg_tif or "").strip()
        if not seg_tif_raw:
            return _invalid_process_features_run("missing_seg_tif", "Choose an annotated FG/BG mask."), None
        seg_tif_path = _resolve_allowed_inference_path(seg_tif_raw)
        if not seg_tif_path.is_file():
            return _invalid_process_features_run("missing_seg_tif", "Annotated FG/BG mask does not exist."), None
        valid_mask_raw = (payload.valid_mask_tif or "").strip()
        if valid_mask_raw:
            valid_mask_tif_path = _resolve_allowed_inference_path(valid_mask_raw)
            if not valid_mask_tif_path.is_file():
                return _invalid_process_features_run(
                    "missing_valid_mask_tif",
                    "Valid voxels mask does not exist.",
                ), None
    else:
        densities_raw = (payload.densities_path or "").strip()
        if not densities_raw:
            return _invalid_process_features_run("missing_probmap_densities", "Choose a Stage 1 output file."), None
        densities_path = _resolve_allowed_inference_path(densities_raw)
        if not densities_path.is_file():
            return _invalid_process_features_run("missing_probmap_densities", "Stage 1 output file does not exist."), None

    return None, {
        "run_density_estimation": run_density_estimation,
        "training_timepoint": training_timepoint,
        "seg_tif": seg_tif_path,
        "valid_mask_tif": valid_mask_tif_path,
        "densities_path": densities_path,
        "density_method": payload.density_method,
        "feature_batch": int(payload.feature_batch),
        "kde_points": int(payload.kde_points),
        "kde_max_samples": int(payload.kde_max_samples),
        "kde_bandwidth": float(payload.kde_bandwidth) if payload.kde_bandwidth is not None else None,
        "hist_sigma_bins": float(payload.hist_sigma_bins),
        "seed": int(payload.seed),
    }


def _build_segmentation_launch_config(
    payload: RunSegmentationRequest,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    segmentation_mode = (payload.mode or "").strip() or (
        SEGMENTATION_MODE_VORONOI_OTSU if payload.enable_voronoi_otsu else ""
    )
    if segmentation_mode not in SEGMENTATION_MODE_OPTIONS:
        return _invalid_process_features_run("invalid_segmentation_mode", "Segmentation mode is invalid."), None

    input_validation = (
        validate_probability_map_input_folder(payload.input_path)
        if segmentation_mode == SEGMENTATION_MODE_PROBABILITY_MAP
        else validate_segmentation_input_folder(payload.input_path)
    )
    if not input_validation["valid"]:
        return _invalid_process_features_run(input_validation["reasonCode"], input_validation["message"]), None

    input_path = _resolve_allowed_inference_path(payload.input_path)
    output_error, output_path = _validate_post_processing_output_path(
        payload.output_path,
        input_path=input_path,
        label="Segmentation output",
    )
    if output_error is not None or output_path is None:
        return output_error or _invalid_process_features_run("output_not_directory", "Segmentation output is invalid."), None
    subfolder_count = int(input_validation["subfolderCount"])

    if segmentation_mode == SEGMENTATION_MODE_VORONOI_OTSU:
        gpu_error, requested_gpu = _validate_requested_gpu(payload.gpu_index)
        if gpu_error is not None or requested_gpu is None:
            return gpu_error or _invalid_process_features_run("missing_gpu_selection", "Select one GPU."), None

        launch_config: dict[str, Any] = {
            "input_path": input_path,
            "output_path": output_path,
            "gpu_index": requested_gpu,
            "mode": segmentation_mode,
            "subfolder_count": subfolder_count,
            "gaussian_blur_sigma": int(payload.gaussian_blur_sigma),
            "rolling_ball_radius": float(payload.rolling_ball_radius),
            "enable_voronoi_otsu": True,
        }
        return (
            {
                "valid": True,
                "message": "Validation passed.",
                "subfolderCount": subfolder_count,
            },
            launch_config,
        )

    if segmentation_mode == SEGMENTATION_MODE_PROBABILITY_MAP:
        launch_config = {
            "input_path": input_path,
            "output_path": output_path,
            "mode": segmentation_mode,
            "subfolder_count": subfolder_count,
            "probmap_threshold": float(payload.probmap_threshold),
            "run_connected_components": bool(payload.run_connected_components),
            "progress_total": subfolder_count,
        }
        return (
            {
                "valid": True,
                "message": "Validation passed.",
                "subfolderCount": subfolder_count,
                "probmapPath": input_validation.get("probmapPath"),
                "probmapExists": bool(input_validation.get("probmapExists")),
            },
            launch_config,
        )

    gpu_error, requested_gpu = _validate_requested_gpu(payload.gpu_index)
    if gpu_error is not None or requested_gpu is None:
        return gpu_error or _invalid_process_features_run("missing_gpu_selection", "Select one GPU."), None

    run_stage_2 = bool(payload.run_stage_2)
    if not payload.run_density_estimation and not run_stage_2:
        return _invalid_process_features_run(
            "no_probability_map_stages_selected",
            "Choose at least one stage: Run stage 1 and/or Run stage 2.",
        ), None

    density_error, density_config = _build_probability_map_density_config(
        payload,
        input_validation=input_validation,
        output_path=output_path,
    )
    if density_error is not None or density_config is None:
        return density_error or _invalid_process_features_run("invalid_density_settings", "Invalid density settings."), None

    progress_total = (2 if density_config["run_density_estimation"] else 0) + (subfolder_count if run_stage_2 else 0)
    launch_config: dict[str, Any] = {
        "input_path": input_path,
        "output_path": output_path,
        "gpu_index": requested_gpu,
        "mode": segmentation_mode,
        "subfolder_count": subfolder_count,
        "run_stage_2": run_stage_2,
        "bg_prob_threshold": float(payload.bg_prob_threshold),
        "fg_prob_threshold": float(payload.fg_prob_threshold),
        "progress_total": progress_total,
        "stage_2_output": "legacy",
        **density_config,
    }
    return (
        {
            "valid": True,
            "message": "Validation passed.",
            "subfolderCount": subfolder_count,
            "probmapDensitiesPath": str(density_config["densities_path"]),
            "probmapDensitiesExists": density_config["densities_path"].is_file(),
        },
        launch_config,
    )


def _build_foreground_probability_map_launch_config(
    payload: RunForegroundProbabilityMapRequest,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    input_validation = validate_segmentation_input_folder(payload.input_path)
    if not input_validation["valid"]:
        return _invalid_process_features_run(input_validation["reasonCode"], input_validation["message"]), None

    gpu_error, requested_gpu = _validate_requested_gpu(payload.gpu_index)
    if gpu_error is not None or requested_gpu is None:
        return gpu_error or _invalid_process_features_run("missing_gpu_selection", "Select one GPU."), None

    input_path = _resolve_allowed_inference_path(payload.input_path)
    output_error, output_path = _validate_post_processing_output_path(
        payload.output_path,
        input_path=input_path,
        label="Output",
    )
    if output_error is not None or output_path is None:
        return output_error or _invalid_process_features_run("output_not_directory", "Output is invalid."), None

    density_error, density_config = _build_probability_map_density_config(
        payload,
        input_validation=input_validation,
        output_path=output_path,
    )
    if density_error is not None or density_config is None:
        return density_error or _invalid_process_features_run("invalid_density_settings", "Invalid density settings."), None

    subfolder_count = int(input_validation["subfolderCount"])
    progress_total = (2 if density_config["run_density_estimation"] else 0) + subfolder_count
    launch_config: dict[str, Any] = {
        "input_path": input_path,
        "output_path": output_path,
        "gpu_index": requested_gpu,
        "mode": "foreground_probability_map",
        "subfolder_count": subfolder_count,
        "run_stage_2": True,
        "bg_prob_threshold": 0.4,
        "fg_prob_threshold": 0.95,
        "progress_total": progress_total,
        "stage_2_output": "probmap",
        **density_config,
    }
    return (
        {
            "valid": True,
            "message": "Validation passed.",
            "subfolderCount": subfolder_count,
            "probmapDensitiesPath": str(density_config["densities_path"]),
            "probmapDensitiesExists": density_config["densities_path"].is_file(),
        },
        launch_config,
    )


def _build_tracking_launch_config(
    payload: RunTrackingRequest,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    input_validation = validate_tracking_input_folder(payload.input_path)
    if not input_validation["valid"]:
        return _invalid_process_features_run(input_validation["reasonCode"], input_validation["message"]), None

    segmentation_validation = validate_tracking_segmentation_folder(payload.input_path, payload.segmentation_path)
    if not segmentation_validation["valid"]:
        return _invalid_process_features_run(
            segmentation_validation["reasonCode"],
            segmentation_validation["message"],
        ), None

    input_path = _resolve_allowed_inference_path(payload.input_path)
    segmentation_path = _resolve_allowed_inference_path(payload.segmentation_path)
    output_error, output_path = _validate_post_processing_output_path(
        payload.output_path,
        input_path=input_path,
        label="Output",
    )
    if output_error is not None:
        return output_error, None
    subfolder_count = int(input_validation["subfolderCount"])
    try:
        vote_thresholds = _parse_tracking_vote_thresholds(payload.vote_thresholds)
    except ValueError as exc:
        return _invalid_process_features_run("invalid_vote_thresholds", str(exc)), None
    try:
        output_filename = _normalize_tracking_output_filename(payload.output_filename)
    except ValueError as exc:
        return _invalid_process_features_run("invalid_output_filename", str(exc)), None
    return (
        {
            "valid": True,
            "message": "Validation passed.",
            "subfolderCount": subfolder_count,
        },
        {
            "input_path": input_path,
            "segmentation_path": segmentation_path,
            "output_path": output_path,
            "output_filename": output_filename,
            "max_distance_xy": float(payload.max_distance_xy),
            "max_distance_z": float(payload.max_distance_z),
            "z_distance_weight": float(payload.z_distance_weight),
            "min_distance_to_remove_cand": float(payload.min_distance_to_remove_cand),
            "vote_thresholds": vote_thresholds,
            "subfolder_count": subfolder_count,
            "pair_count": max(0, subfolder_count - 1),
            "progress_total": subfolder_count + max(0, subfolder_count - 1),
            "dice_threshold": float(payload.dice_threshold),
            "corr_threshold": float(payload.corr_threshold),
            "invert_z": bool(payload.invert_z),
            "save_extended_results": bool(payload.save_extended_results),
            "ignore_features": bool(payload.ignore_features),
            "disable_centroid_fallback": bool(payload.disable_centroid_fallback),
            "aggressive_feature_matching": bool(payload.aggressive_feature_matching),
            "min_feature_votes": int(payload.min_feature_votes),
        },
    )


def _resolve_omp_num_threads() -> str:
    omp_num_threads = DEFAULT_INFERENCE_OMP_NUM_THREADS
    raw_omp_num_threads = os.environ.get("OMP_NUM_THREADS", "").strip()
    if raw_omp_num_threads:
        try:
            omp_num_threads = max(DEFAULT_INFERENCE_OMP_NUM_THREADS, int(raw_omp_num_threads))
        except ValueError:
            omp_num_threads = DEFAULT_INFERENCE_OMP_NUM_THREADS
    return str(omp_num_threads)


def _validate_normalization_payload(
    payload: RunInferenceRequest,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    normalization_mode = payload.normalization_mode.strip() or NORMALIZATION_MODE_PER_VOLUME
    if normalization_mode not in NORMALIZATION_MODE_OPTIONS:
        return (
            _invalid_inference_run("invalid_normalization_mode", "Normalization mode is invalid."),
            None,
        )

    global_hist_min = payload.global_hist_min
    global_hist_max = payload.global_hist_max

    if normalization_mode == NORMALIZATION_MODE_PER_VOLUME:
        if global_hist_min is not None or global_hist_max is not None:
            return (
                _invalid_inference_run(
                    "unexpected_global_hist_values",
                    "Global histogram values are only valid for manual global normalization.",
                ),
                None,
            )
        return None, {
            "normalization_mode": normalization_mode,
            "global_hist_min": None,
            "global_hist_max": None,
        }

    if normalization_mode == NORMALIZATION_MODE_GLOBAL_AUTO:
        if global_hist_min is not None or global_hist_max is not None:
            return (
                _invalid_inference_run(
                    "unexpected_global_hist_values",
                    "Do not provide manual global histogram values when auto-compute is selected.",
                ),
                None,
            )
        return None, {
            "normalization_mode": normalization_mode,
            "global_hist_min": None,
            "global_hist_max": None,
        }

    if global_hist_min is None or global_hist_max is None:
        return (
            _invalid_inference_run(
                "missing_global_hist_values",
                "Manual global normalization requires both global histogram values.",
            ),
            None,
        )
    if global_hist_max <= global_hist_min:
        return (
            _invalid_inference_run(
                "invalid_global_hist_values",
                "Global histogram max must be greater than global histogram min.",
            ),
            None,
        )

    return None, {
        "normalization_mode": normalization_mode,
        "global_hist_min": float(global_hist_min),
        "global_hist_max": float(global_hist_max),
    }


def _summarize_directory(path: Path) -> tuple[int, list[str]]:
    if not path.exists() or not path.is_dir():
        return 0, []

    entries = sorted((entry.name for entry in path.iterdir()), key=lambda value: (value.casefold(), value))
    return len(entries), entries[:8]


def _clear_directory_contents(path: Path) -> None:
    if not path.exists():
        return

    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
            continue
        if child.is_dir():
            shutil.rmtree(child)


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)


def _extract_zip_to_directory(archive_path: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if not infos:
            raise RuntimeError("Archive is empty.")

        for info in infos:
            member = Path(info.filename)
            if member.is_absolute():
                raise RuntimeError(f"Archive contains an absolute path: {info.filename!r}")
            if any(part == ".." for part in member.parts):
                raise RuntimeError(f"Archive contains an unsafe path: {info.filename!r}")

        archive.extractall(extract_dir)


def _select_extracted_dataset_path(extract_dir: Path, dataset_name: str) -> Path:
    named_path = extract_dir / dataset_name
    if named_path.is_dir():
        return named_path

    visible_children = [child for child in extract_dir.iterdir() if child.name not in {"__MACOSX"}]
    if len(visible_children) == 1 and visible_children[0].is_dir():
        return visible_children[0]
    return extract_dir


def _invalid_data_download(reason_code: str, message: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reasonCode": reason_code,
        "message": message,
    }


def _build_data_download_launch_config(
    payload: RunDataDownloadRequest,
    *,
    require_overwrite_confirmation: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        manifest_datasets = _load_public_data_manifest()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    selected_names: list[str] = []
    seen_names: set[str] = set()
    for raw_name in payload.datasets:
        try:
            name = _validate_public_data_dataset_name(raw_name)
        except ValueError:
            return _invalid_data_download("invalid_dataset_name", f"Invalid dataset selection: {raw_name!r}"), None
        if name in seen_names:
            continue
        seen_names.add(name)
        selected_names.append(name)

    if not selected_names:
        return _invalid_data_download("missing_selection", "Select at least one dataset to download."), None

    manifest_by_name = {item["name"]: item for item in manifest_datasets}
    missing_names = [name for name in selected_names if name not in manifest_by_name]
    if missing_names:
        missing_preview = ", ".join(missing_names[:3])
        if len(missing_names) > 3:
            missing_preview = f"{missing_preview}, ..."
        return _invalid_data_download(
            "unknown_dataset",
            f"Some selected datasets are not available in the public data manifest: {missing_preview}.",
        ), None

    repo_root = get_repo_root()
    raw_data_dir = get_public_data_dir()
    if not _is_relative_to(raw_data_dir, repo_root):
        raise HTTPException(status_code=500, detail="Public data directory must stay inside the repo root.")

    selected_datasets = [manifest_by_name[name] for name in selected_names]
    existing_items = [
        {
            "name": item["name"],
            "path": str(raw_data_dir / item["name"]),
        }
        for item in selected_datasets
        if (raw_data_dir / item["name"]).exists() or (raw_data_dir / item["name"]).is_symlink()
    ]

    if existing_items and payload.existing_mode is None and require_overwrite_confirmation:
        existing_names = [item["name"] for item in existing_items]
        return {
            "valid": True,
            "requiresOverwriteConfirmation": True,
            "message": (
                f"{len(existing_names)} selected dataset{'s already exist' if len(existing_names) != 1 else ' already exists'} "
                "under data/raw_data. Do you want to skip them or overwrite them?"
            ),
            "existingDatasetCount": len(existing_names),
            "existingDatasetNames": existing_names,
            "existingDatasetPaths": [item["path"] for item in existing_items],
        }, None

    existing_mode = payload.existing_mode or "overwrite"
    overwrite_existing = existing_mode == "overwrite"
    if existing_mode == "skip":
        selected_datasets = [
            item
            for item in selected_datasets
            if not (raw_data_dir / item["name"]).exists() and not (raw_data_dir / item["name"]).is_symlink()
        ]

    if not selected_datasets:
        return (
            _invalid_data_download(
                "nothing_to_download",
                "All selected datasets already exist in data/raw_data. Choose overwrite to replace them.",
            ),
            None,
        )

    selected_dataset_names = {item["name"] for item in selected_datasets}
    skipped_names = [name for name in selected_names if name not in selected_dataset_names]
    return None, {
        "manifest_url": get_public_data_manifest_url(),
        "download_root": raw_data_dir,
        "selected_datasets": selected_datasets,
        "selected_names": [item["name"] for item in selected_datasets],
        "skipped_names": skipped_names,
        "overwrite_existing": overwrite_existing,
    }


def _resolve_backbone_weight_path(raw_value: str) -> Path | None:
    selected = (raw_value or "").strip()
    if not selected:
        return None

    repo_root = get_repo_root()
    allowed = {item["value"] for item in list_backbone_weights()}
    if selected not in allowed:
        return None

    resolved = (repo_root / selected).resolve()
    if not _is_relative_to(resolved, repo_root):
        return None
    if not resolved.is_file():
        return None
    return resolved


def _resolve_backbone_model_config(raw_value: str) -> tuple[str, dict[str, Any]] | None:
    selected = (raw_value or "").strip().lower()
    config = INFERENCE_BACKBONE_MODEL_CONFIGS.get(selected)
    if config is None:
        return None
    return selected, dict(config)


def _format_hydra_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _coerce_axis_triplet_xyz(value: float | InferenceAxisRequest | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, InferenceAxisRequest):
        if value.x is None or value.y is None or value.z is None:
            return None
        return (float(value.x), float(value.y), float(value.z))
    scalar = float(value)
    return (scalar, scalar, scalar)


def _format_upsample_factor_override(value: Any) -> str:
    if isinstance(value, (int, float)):
        return str(float(value))
    if isinstance(value, InferenceAxisRequest):
        axis_xyz = _coerce_axis_triplet_xyz(value)
        if axis_xyz is None:
            raise ValueError("Upsample factor values are required.")
    else:
        axis_xyz = tuple(float(item) for item in value)
        if len(axis_xyz) != 3:
            raise ValueError("Upsample factor must be a scalar or three axis values.")

    x_factor, y_factor, z_factor = axis_xyz
    return f"[{z_factor},{y_factor},{x_factor}]"


def _build_inference_launch_config(
    payload: RunInferenceRequest,
    *,
    require_overwrite_confirmation: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    input_validation = validate_inference_input_folder(payload.input_path)
    if not input_validation["valid"]:
        return _invalid_inference_run(input_validation["reasonCode"], input_validation["message"]), None

    input_path = _resolve_allowed_inference_path(payload.input_path)
    output_path = _resolve_allowed_inference_path(payload.output_path)
    if output_path.exists() and not output_path.is_dir():
        return _invalid_inference_run("output_not_directory", "Output path is not a folder."), None

    if _is_relative_to(input_path, output_path):
        return _invalid_inference_run(
            "output_contains_input",
            "Output folder cannot be the input folder, or a parent of it.",
        ), None

    backbone_path = _resolve_backbone_weight_path(payload.backbone_weight)
    if backbone_path is None:
        return _invalid_inference_run("missing_backbone_weight", "Select a backbone weights file."), None

    backbone_model_config = _resolve_backbone_model_config(payload.backbone_model)
    if backbone_model_config is None:
        return _invalid_inference_run("invalid_backbone_model", "Backbone model type is invalid."), None
    backbone_model, backbone_config_overrides = backbone_model_config

    requested_gpus = sorted(set(int(index) for index in payload.gpu_indices))
    if not requested_gpus:
        return _invalid_inference_run("missing_gpu_selection", "Select at least one GPU."), None

    available_gpu_indices = {
        int(gpu["index"]) for gpu in get_nvidia_gpu_memory().get("gpus", []) if "index" in gpu
    }
    if any(index not in available_gpu_indices for index in requested_gpus):
        return _invalid_inference_run("invalid_gpu_selection", "Selected GPUs are not available on the server."), None

    raw_upsample_factor = payload.upsample_factor
    upsample_factor_xyz = _coerce_axis_triplet_xyz(raw_upsample_factor)
    if upsample_factor_xyz is None or any(factor < 1.0 for factor in upsample_factor_xyz):
        return _invalid_inference_run("invalid_upsample_factor", "Upsample factor must be greater than or equal to 1."), None

    anisotropy_x = payload.anisotropy.x
    anisotropy_y = payload.anisotropy.y
    anisotropy_z = payload.anisotropy.z
    if anisotropy_x is None or anisotropy_y is None or anisotropy_z is None:
        return _invalid_inference_run("invalid_anisotropy", "Anisotropy correction values are required."), None
    if anisotropy_x <= 0 or anisotropy_y <= 0 or anisotropy_z <= 0:
        return _invalid_inference_run("invalid_anisotropy", "Anisotropy correction values must be greater than 0."), None

    route_mapping = {"full": "default", "streaming": "streaming"}
    inference_route = route_mapping.get(payload.route)
    if inference_route is None:
        return _invalid_inference_run("invalid_route", "Inference route is invalid."), None

    dtype_mapping = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    dtype = dtype_mapping.get(payload.precision)
    if dtype is None:
        return _invalid_inference_run("invalid_precision", "Precision is invalid."), None

    padding_mode = payload.padding_mode.lower()
    if padding_mode not in INFERENCE_PADDING_MODES:
        return _invalid_inference_run("invalid_padding_mode", "Padding mode is invalid."), None

    normalization_error, normalization_config = _validate_normalization_payload(payload)
    if normalization_error is not None or normalization_config is None:
        return normalization_error, None

    shape = input_validation["shape"]
    raw_shape = (int(shape["z"]), int(shape["y"]), int(shape["x"]))
    crop_start_x = _coerce_start(payload.crop_bounds.x_start)
    crop_start_y = _coerce_start(payload.crop_bounds.y_start)
    crop_start_z = _coerce_start(payload.crop_bounds.z_start)
    crop_end_x_inclusive = payload.crop_bounds.x_end
    crop_end_y_inclusive = payload.crop_bounds.y_end
    crop_end_z_inclusive = payload.crop_bounds.z_end
    crop_error = _validate_inclusive_axis_bounds(crop_start_x, crop_end_x_inclusive, size=raw_shape[2], axis_label="X")
    if crop_error is None:
        crop_error = _validate_inclusive_axis_bounds(
            crop_start_y,
            crop_end_y_inclusive,
            size=raw_shape[1],
            axis_label="Y",
        )
    if crop_error is None:
        crop_error = _validate_inclusive_axis_bounds(
            crop_start_z,
            crop_end_z_inclusive,
            size=raw_shape[0],
            axis_label="Z",
        )
    if crop_error is not None:
        return _invalid_inference_run("invalid_crop", f"Crop parameters are invalid: {crop_error}"), None

    crop_end_x = _coerce_script_crop_end(crop_end_x_inclusive)
    crop_end_y = _coerce_script_crop_end(crop_end_y_inclusive)
    crop_end_z = _coerce_script_crop_end(crop_end_z_inclusive)

    effective_crop_params = (
        crop_start_z,
        crop_end_z if crop_end_z > 0 else raw_shape[0],
        crop_start_y,
        crop_end_y if crop_end_y > 0 else raw_shape[1],
        crop_start_x,
        crop_end_x if crop_end_x > 0 else raw_shape[2],
    )
    try:
        _validate_inference_crop_params(effective_crop_params, raw_shape)
    except AssertionError as exc:
        return _invalid_inference_run("invalid_crop", f"Crop parameters are invalid: {exc}"), None

    file_start = _coerce_start(payload.file_range.start)
    file_count = int(input_validation["fileCount"])
    file_end_inclusive = payload.file_range.end
    if file_start < 0 or file_start >= file_count:
        return _invalid_inference_run(
            "invalid_file_range",
            f"Start file must be between 0 and {file_count - 1}.",
        ), None
    if file_end_inclusive is not None and (file_end_inclusive < 0 or file_end_inclusive >= file_count):
        return _invalid_inference_run(
            "invalid_file_range",
            f"End file must be between 0 and {file_count - 1}.",
        ), None
    file_end = _coerce_script_file_end(file_end_inclusive)
    effective_file_end = file_count if file_end is None else file_end
    if effective_file_end <= file_start:
        return _invalid_inference_run("empty_file_selection", "Chosen files leave zero files to process."), None

    selected_input_paths = _selected_inference_tiff_paths(input_path, file_start, effective_file_end)
    if not selected_input_paths:
        return _invalid_inference_run("empty_file_selection", "Chosen files leave zero files to process."), None
    selected_stems = [path.stem for path in selected_input_paths]

    overwrite_warning: dict[str, Any] | None = None
    output_entry_count, output_preview = _summarize_existing_inference_outputs(output_path)
    if output_entry_count > 0 and not payload.overwrite:
        overwrite_warning = _inference_overwrite_confirmation(output_path, output_entry_count, output_preview)
        if require_overwrite_confirmation:
            return overwrite_warning, None

    launch_upsample_factor: float | tuple[float, float, float]
    if isinstance(raw_upsample_factor, InferenceAxisRequest):
        launch_upsample_factor = upsample_factor_xyz
    else:
        launch_upsample_factor = float(raw_upsample_factor)

    return (
        {
            "valid": True,
            "message": "Validation passed.",
            "requiresOverwriteConfirmation": False,
            "selectedFileCount": len(selected_stems),
        },
        {
            "input_path": input_path,
            "output_path": output_path,
            "backbone_path": backbone_path,
            "backbone_model": backbone_model,
            "backbone_config_overrides": backbone_config_overrides,
            "gpu_indices": requested_gpus,
            "upsample_factor": launch_upsample_factor,
            "anisotropy_xyz": (
                float(anisotropy_x),
                float(anisotropy_y),
                float(anisotropy_z),
            ),
            "file_start": file_start,
            "file_end": file_end,
            "crop_params": [
                crop_start_z,
                crop_end_z,
                crop_start_y,
                crop_end_y,
                crop_start_x,
                crop_end_x,
            ],
            "effective_crop_params": effective_crop_params,
            "inference_route": inference_route,
            "dtype": dtype,
            "padding_mode": padding_mode,
            "normalization_mode": normalization_config["normalization_mode"],
            "global_hist_min": normalization_config["global_hist_min"],
            "global_hist_max": normalization_config["global_hist_max"],
            "selected_file_count": len(selected_stems),
            "selected_stems": selected_stems,
            "overwrite": bool(payload.overwrite),
            "overwrite_warning": overwrite_warning,
        },
    )


def _build_inference_command(launch_config: dict[str, Any]) -> list[str]:
    anisotropy_x, anisotropy_y, anisotropy_z = launch_config["anisotropy_xyz"]
    crop_params = launch_config["crop_params"]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={len(launch_config['gpu_indices'])}",
        "scripts/inference/inference.py",
        f"file_path={launch_config['input_path']}",
        f"save_path={launch_config['output_path']}",
        f"backbone_path={launch_config['backbone_path']}",
        f"file_start={launch_config['file_start']}",
        f"crop_params=[{crop_params[0]},{crop_params[1]},{crop_params[2]},{crop_params[3]},{crop_params[4]},{crop_params[5]}]",
        f"upsample_factor={_format_upsample_factor_override(launch_config['upsample_factor'])}",
        f"isotropic_scale_factor=[{anisotropy_z},{anisotropy_y},{anisotropy_x}]",
        f"inference_route={launch_config['inference_route']}",
        f"dtype={launch_config['dtype']}",
        f"padding_mode={launch_config.get('padding_mode', 'reflect')}",
    ]
    backbone_config_overrides = launch_config.get(
        "backbone_config_overrides",
        INFERENCE_BACKBONE_MODEL_CONFIGS[INFERENCE_BACKBONE_MODEL_NOPE],
    )
    for key, value in backbone_config_overrides.items():
        command.append(f"{key}={_format_hydra_override_value(value)}")
    global_hist_min = launch_config.get("global_hist_min")
    global_hist_max = launch_config.get("global_hist_max")
    if global_hist_min is not None and global_hist_max is not None:
        command.append(f"global_hist_min={global_hist_min}")
        command.append(f"global_hist_max={global_hist_max}")
    if launch_config["file_end"] is not None:
        command.append(f"file_end={launch_config['file_end']}")
    return command


def _build_process_features_command(launch_config: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "scripts/post_processing/process_features.py",
        "--input-path",
        str(launch_config["input_path"]),
        "--output-path",
        str(launch_config["output_path"]),
        "--pca-components",
        str(launch_config["pca_components"]),
        "--pca-format",
        launch_config["pca_save_format"],
        "--global-pca",
        "true" if launch_config.get("global_pca", True) else "false",
        "--high-resolution-format",
        launch_config["high_resolution_save_format"],
        "--file-start",
        str(launch_config["file_start"]),
    ]
    if launch_config["file_end"] is not None:
        command.extend(["--file-end", str(launch_config["file_end"])])
    if launch_config["save_pca"]:
        command.append("--save-pca")
    if launch_config["save_high_resolution_features"]:
        command.append("--save-high-resolution-features")
    return command


def _build_probability_map_command(launch_config: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "scripts/post_processing/probability_map.py",
        "--input-path",
        str(launch_config["input_path"]),
        "--output-path",
        str(launch_config["output_path"]),
        "--densities-path",
        str(launch_config["densities_path"]),
        "--device",
        "cuda:0",
        "--feature-batch",
        str(launch_config["feature_batch"]),
        "--kde-points",
        str(launch_config["kde_points"]),
        "--kde-max-samples",
        str(launch_config["kde_max_samples"]),
        "--density-method",
        launch_config["density_method"],
        "--hist-sigma-bins",
        str(launch_config["hist_sigma_bins"]),
        "--bg-prob-threshold",
        str(launch_config["bg_prob_threshold"]),
        "--fg-prob-threshold",
        str(launch_config["fg_prob_threshold"]),
        "--stage-2-output",
        launch_config.get("stage_2_output", "legacy"),
        "--seed",
        str(launch_config["seed"]),
    ]
    if launch_config.get("run_stage_2", True):
        command.append("--run-stage-2")
    else:
        command.append("--skip-stage-2")
    if launch_config.get("kde_bandwidth") is not None:
        command.extend(["--kde-bandwidth", str(launch_config["kde_bandwidth"])])
    if launch_config.get("run_density_estimation"):
        command.extend(["--run-density-estimation", "--training-timepoint", launch_config["training_timepoint"]])
        if launch_config.get("seg_tif") is not None:
            command.extend(["--seg-tif", str(launch_config["seg_tif"])])
        if launch_config.get("valid_mask_tif") is not None:
            command.extend(["--valid-mask-tif", str(launch_config["valid_mask_tif"])])
    return command


def _build_segmentation_command(launch_config: dict[str, Any]) -> list[str]:
    if launch_config["mode"] == SEGMENTATION_MODE_LEGACY_PROBABILITY_MAP:
        return _build_probability_map_command(launch_config)

    if launch_config["mode"] == SEGMENTATION_MODE_PROBABILITY_MAP:
        command = [
            sys.executable,
            "scripts/post_processing/probmap_segmentation.py",
            "--input-path",
            str(launch_config["input_path"]),
            "--output-path",
            str(launch_config["output_path"]),
            "--threshold",
            str(launch_config["probmap_threshold"]),
        ]
        command.append("--run-ccl" if launch_config.get("run_connected_components", True) else "--skip-ccl")
        return command

    command = [
        sys.executable,
        "scripts/post_processing/segmentation.py",
        "--input-path",
        str(launch_config["input_path"]),
        "--output-path",
        str(launch_config["output_path"]),
        "--gaussian-blur-sigma",
        str(launch_config["gaussian_blur_sigma"]),
        "--rolling-ball-radius",
        str(launch_config["rolling_ball_radius"]),
    ]
    if launch_config["enable_voronoi_otsu"]:
        command.append("--enable-voronoi-otsu")
    return command


def _build_foreground_probability_map_command(launch_config: dict[str, Any]) -> list[str]:
    return _build_probability_map_command(launch_config)


def _build_tracking_command(launch_config: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "scripts/post_processing/tracking.py",
        "--input-path",
        str(launch_config["input_path"]),
        "--segmentation-path",
        str(launch_config["segmentation_path"]),
        "--output-path",
        str(launch_config["output_path"]),
        "--output-filename",
        str(launch_config["output_filename"]),
        "--max-distance-xy",
        str(launch_config["max_distance_xy"]),
        "--max-distance-z",
        str(launch_config["max_distance_z"]),
        "--z-distance-weight",
        str(launch_config["z_distance_weight"]),
        "--min-distance-to-remove-cand",
        str(launch_config["min_distance_to_remove_cand"]),
        "--dice-threshold",
        str(launch_config["dice_threshold"]),
        "--corr-threshold",
        str(launch_config["corr_threshold"]),
        "--min-feature-votes",
        str(launch_config["min_feature_votes"]),
    ]
    vote_thresholds = launch_config.get("vote_thresholds")
    if vote_thresholds:
        command.extend(["--vote-thresholds", ",".join(str(value) for value in vote_thresholds)])
    else:
        command.extend(["--vote-thresholds", ""])
    if launch_config.get("invert_z", False):
        command.append("--invert-z")
    if launch_config.get("save_extended_results", False):
        command.append("--save-extended-results")
    if launch_config.get("ignore_features", False):
        command.append("--ignore-features")
    if launch_config.get("disable_centroid_fallback", False):
        command.append("--disable-centroid-fallback")
    if launch_config.get("aggressive_feature_matching", False):
        command.append("--aggressive-feature-matching")
    return command


def _build_norm_per_vol_command(launch_config: dict[str, Any]) -> list[str]:
    anisotropy_x, anisotropy_y, anisotropy_z = launch_config["anisotropy_xyz"]
    crop_params = launch_config["crop_params"]
    command = [
        sys.executable,
        "scripts/inference/norm_per_vol.py",
        f"file_path={launch_config['input_path']}",
        f"save_path={launch_config['output_path']}",
        f"file_start={launch_config['file_start']}",
        f"crop_params=[{crop_params[0]},{crop_params[1]},{crop_params[2]},{crop_params[3]},{crop_params[4]},{crop_params[5]}]",
        f"isotropic_scale_factor=[{anisotropy_z},{anisotropy_y},{anisotropy_x}]",
    ]
    if launch_config["file_end"] is not None:
        command.append(f"file_end={launch_config['file_end']}")
    return command


def _build_inference_command_env(launch_config: dict[str, Any]) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in launch_config["gpu_indices"]),
        "OMP_NUM_THREADS": _resolve_omp_num_threads(),
        "PYTHONUNBUFFERED": "1",
    }


def _build_process_features_command_env(launch_config: dict[str, Any]) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(launch_config["gpu_index"]),
        "OMP_NUM_THREADS": _resolve_omp_num_threads(),
        "PYTHONUNBUFFERED": "1",
    }


def _build_tracking_command_env() -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": _resolve_omp_num_threads(),
        "PYTHONUNBUFFERED": "1",
    }


def _build_norm_per_vol_command_env() -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": _resolve_omp_num_threads(),
        "PYTHONUNBUFFERED": "1",
    }


def _render_inference_preview_command(launch_config: dict[str, Any], *, cwd: Path) -> str:
    normalization_mode = launch_config.get("normalization_mode")
    inference_env = _build_inference_command_env(launch_config)

    if normalization_mode != NORMALIZATION_MODE_GLOBAL_AUTO:
        return _render_shell_command(_build_inference_command(launch_config), cwd=cwd, env=inference_env)

    norm_command = _build_norm_per_vol_command(launch_config)
    norm_env = _build_norm_per_vol_command_env()
    norm_text = _render_shell_command(norm_command, env=norm_env)

    preview_launch = dict(launch_config)
    preview_launch["global_hist_min"] = "<computed-from-norm_per_vol>"
    preview_launch["global_hist_max"] = "<computed-from-norm_per_vol>"
    inference_text = _render_shell_command(_build_inference_command(preview_launch), env=inference_env)
    return f"cd {shlex.quote(os.fspath(cwd))} && {norm_text} && \\\n{inference_text}"


def _render_shell_command(
    command: list[str],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    command_text = shlex.join(command)
    env = env or {}
    if env:
        env_prefix = " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())
        command_text = f"{env_prefix} {command_text}"
    if cwd is None:
        return command_text
    return f"cd {shlex.quote(os.fspath(cwd))} && {command_text}"


def _norm_per_vol_stats_path(output_path: Path) -> Path:
    return inference_norm_per_vol_stats_path(output_path)


def _parse_norm_per_vol_stats_text(text: str) -> tuple[float, float]:
    min_match = NORM_PER_VOL_MIN_RE.search(text)
    max_match = NORM_PER_VOL_MAX_RE.search(text)
    if min_match is None or max_match is None:
        raise ValueError("norm_per_vol.txt is missing global histogram values.")

    global_hist_min = float(min_match.group(1))
    global_hist_max = float(max_match.group(1))
    if global_hist_max <= global_hist_min:
        raise ValueError("norm_per_vol.txt contains invalid global histogram bounds.")
    return global_hist_min, global_hist_max


def _read_norm_per_vol_stats(path: Path) -> tuple[float, float]:
    return _parse_norm_per_vol_stats_text(path.read_text(encoding="utf-8"))


def _clean_process_line(raw_line: str) -> str:
    cleaned = ANSI_ESCAPE_RE.sub("", raw_line.replace("\r", "\n")).rstrip()
    return cleaned if cleaned.strip() else ""


def _canonicalize_runtime_path(path: Path | str) -> str:
    return os.path.realpath(os.fspath(path))


def _expected_feature_paths(output_path: Path, selected_stems: list[str]) -> list[Path]:
    return [inference_lr_feats_path(output_path, stem) for stem in selected_stems]


def _existing_expected_feature_paths(expected_feature_paths: list[Path]) -> list[Path]:
    return [path for path in expected_feature_paths if path.is_file()]


def _run_norm_per_vol_prepass(
    job: jobs_api.JobState,
    launch_config: dict[str, Any],
    *,
    repo_root: Path,
    log_handle: TextIO | None,
) -> tuple[float, float] | None:
    output_path = Path(launch_config["output_path"])
    stats_path = _norm_per_vol_stats_path(output_path)
    command = _build_norm_per_vol_command(launch_config)
    command_env = _build_norm_per_vol_command_env()
    command_text = _render_shell_command(command, cwd=repo_root, env=command_env)

    _append_job_log_line(job, f"[server] Global normalization prepass command: {command_text}", log_handle)

    with job.lock:
        if job.stop_requested:
            _mark_job_halted_locked(job)
            return None
        job.current = "Computing normalization stats"

    env = os.environ.copy()
    env.update(command_env)
    _append_job_log_line(job, "[server] Launching normalization prepass.", log_handle)
    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    with job.lock:
        job.process = process
        if job.stop_requested:
            job.current = "Stopping"

    if job.stop_requested:
        jobs_api.terminate_job_process(job, force=False)

    last_output_line: str | None = None
    if process.stdout is not None:
        for raw_line in process.stdout:
            for part in raw_line.split("\r"):
                line = _clean_process_line(part)
                if not line:
                    continue
                last_output_line = line
                _append_job_log_line(job, line, log_handle)

    return_code = process.wait()
    with job.lock:
        job.process = None
        if job.stop_requested:
            _mark_job_halted_locked(job)
            return None

    if return_code != 0:
        raise RuntimeError(last_output_line or f"norm_per_vol.py exited with code {return_code}.")
    if not stats_path.is_file():
        raise RuntimeError("Normalization prepass did not produce norm_per_vol.txt.")

    global_hist_min, global_hist_max = _read_norm_per_vol_stats(stats_path)
    _append_job_log_line(
        job,
        (
            "[server] Parsed global normalization stats: "
            f"global_hist_min={global_hist_min}, global_hist_max={global_hist_max}"
        ),
        log_handle,
    )
    return global_hist_min, global_hist_max


def _mark_job_halted_locked(job: jobs_api.JobState) -> None:
    job.status = "halted"
    job.current = "Stopped"
    job.finished_at_ms = jobs_api._now_ms()
    job.process = None


def _open_job_log(job: jobs_api.JobState) -> TextIO | None:
    try:
        log_dir = get_job_logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = (log_dir / f"{job.job_id}.log").resolve()
        handle = log_path.open("w", encoding="utf-8")
    except OSError:
        with job.lock:
            job.log_path = None
            job.log_available = False
        return None

    with job.lock:
        job.log_path = str(log_path)
    return handle


def _append_job_log_line(job: jobs_api.JobState, line: str, log_handle: TextIO | None) -> None:
    text = line.rstrip("\r\n")
    if not text:
        return

    if log_handle is not None:
        try:
            log_handle.write(f"{text}\n")
            log_handle.flush()
        except OSError:
            pass

    with job.lock:
        job.log_tail.append(text)
        job.log_line_count += 1
        job.log_available = True


def _update_job_progress_from_output(
    job: jobs_api.JobState,
    line: str,
    expected_saving_paths: set[str],
    expected_feature_paths: set[str],
    saved_feature_paths: set[str],
) -> None:
    saved_match = SAVED_FEATURES_RE.search(line)
    if saved_match:
        saved_path_text = saved_match.group(1).strip()
        saved_feature_key = _canonicalize_runtime_path(saved_path_text)
        if saved_feature_key not in expected_feature_paths:
            return
        save_name = Path(saved_path_text).stem or saved_path_text
        with job.lock:
            if job.status != "running" or job.stop_requested or saved_feature_key in saved_feature_paths:
                return
            saved_feature_paths.add(saved_feature_key)
            job.processed = min(job.total, len(saved_feature_paths)) if job.total > 0 else len(saved_feature_paths)
            job.current = save_name
        return

    saving_match = SAVE_PATH_RE.search(line)
    if saving_match:
        save_path_text = saving_match.group(1).strip()
        save_path_key = _canonicalize_runtime_path(save_path_text)
        if save_path_key not in expected_saving_paths:
            return
        save_name = Path(save_path_text).stem or save_path_text
        with job.lock:
            if job.status == "running" and not job.stop_requested:
                job.current = f"Processing {save_name}"
        return

    if "Processing " in line and " files" in line:
        with job.lock:
            if job.status == "running" and not job.stop_requested:
                job.current = "Running"


def _update_process_features_job_progress_from_output(
    job: jobs_api.JobState,
    line: str,
    progress_state: dict[str, Any],
) -> int:
    completed_subfolders = progress_state.setdefault("completed_subfolders", set())
    processing_match = PROCESS_FEATURES_PROCESSING_RE.search(line)
    if processing_match:
        subfolder_name = processing_match.group(1).strip()
        with job.lock:
            if job.status == "running" and not job.stop_requested:
                job.current = f"Processing {subfolder_name}"
        return len(completed_subfolders)

    completed_match = PROCESS_FEATURES_COMPLETED_RE.search(line)
    if not completed_match:
        return len(completed_subfolders)

    subfolder_name = completed_match.group(1).strip()
    with job.lock:
        if job.status != "running" or job.stop_requested or subfolder_name in completed_subfolders:
            return len(completed_subfolders)
        completed_subfolders.add(subfolder_name)
        job.processed = min(job.total, len(completed_subfolders)) if job.total > 0 else len(completed_subfolders)
        job.current = subfolder_name
    return len(completed_subfolders)


def _update_tracking_job_progress_from_output(
    job: jobs_api.JobState,
    line: str,
    progress_state: dict[str, Any],
) -> int:
    completed_subfolders = progress_state.setdefault("completed_subfolders", set())
    completed_pairs = progress_state.setdefault("completed_pairs", set())
    total_completed = len(completed_subfolders) + len(completed_pairs)

    matching_match = TRACKING_MATCHING_RE.search(line)
    if matching_match:
        ref_name = matching_match.group(1).strip()
        cand_name = matching_match.group(2).strip()
        with job.lock:
            if job.status == "running" and not job.stop_requested:
                job.current = f"Matching {ref_name} -> {cand_name}"
        return total_completed

    matched_match = TRACKING_MATCHED_RE.search(line)
    if matched_match:
        ref_name = matched_match.group(1).strip()
        cand_name = matched_match.group(2).strip()
        pair_key = f"{ref_name}->{cand_name}"
        with job.lock:
            if job.status != "running" or job.stop_requested or pair_key in completed_pairs:
                return len(completed_subfolders) + len(completed_pairs)
            completed_pairs.add(pair_key)
            total_completed = len(completed_subfolders) + len(completed_pairs)
            job.processed = min(job.total, total_completed) if job.total > 0 else total_completed
            job.current = f"Matched {ref_name} -> {cand_name}"
        return len(completed_subfolders) + len(completed_pairs)

    processing_match = PROCESS_FEATURES_PROCESSING_RE.search(line)
    if processing_match:
        subfolder_name = processing_match.group(1).strip()
        with job.lock:
            if job.status == "running" and not job.stop_requested:
                job.current = f"Preparing {subfolder_name}"
        return total_completed

    completed_match = PROCESS_FEATURES_COMPLETED_RE.search(line)
    if not completed_match:
        return total_completed

    subfolder_name = completed_match.group(1).strip()
    with job.lock:
        if job.status != "running" or job.stop_requested or subfolder_name in completed_subfolders:
            return len(completed_subfolders) + len(completed_pairs)
        completed_subfolders.add(subfolder_name)
        total_completed = len(completed_subfolders) + len(completed_pairs)
        job.processed = min(job.total, total_completed) if job.total > 0 else total_completed
        job.current = f"Prepared {subfolder_name}"
    return len(completed_subfolders) + len(completed_pairs)


def _run_inference_job(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    output_path = Path(launch_config["output_path"])
    expected_feature_paths = _expected_feature_paths(output_path, launch_config["selected_stems"])
    expected_feature_path_keys = {_canonicalize_runtime_path(path) for path in expected_feature_paths}
    expected_saving_path_keys = expected_feature_path_keys
    saved_feature_paths: set[str] = set()
    repo_root = get_repo_root()
    log_handle = _open_job_log(job)
    preview_command_text = _render_inference_preview_command(launch_config, cwd=repo_root)

    with job.lock:
        job.command = preview_command_text
        job.working_dir = str(repo_root)

    _append_job_log_line(job, f"[server] Working directory: {repo_root}", log_handle)
    _append_job_log_line(job, f"[server] Planned command: {preview_command_text}", log_handle)

    try:
        with job.lock:
            if job.stop_requested:
                _mark_job_halted_locked(job)
                stop_before_launch = True
            else:
                stop_before_launch = False
                job.current = "Preparing output folder"
        if stop_before_launch:
            _append_job_log_line(job, "[server] Job stopped before launch.", log_handle)
            return

        if launch_config["overwrite"]:
            _clear_inference_managed_outputs(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        if launch_config.get("normalization_mode") == NORMALIZATION_MODE_GLOBAL_AUTO:
            prepass_result = _run_norm_per_vol_prepass(
                job,
                launch_config,
                repo_root=repo_root,
                log_handle=log_handle,
            )
            if prepass_result is None:
                _append_job_log_line(job, "[server] Job stopped during normalization prepass.", log_handle)
                return

            global_hist_min, global_hist_max = prepass_result
            launch_config["global_hist_min"] = global_hist_min
            launch_config["global_hist_max"] = global_hist_max
            launch_config["normalization_mode"] = NORMALIZATION_MODE_GLOBAL_MANUAL
            with job.lock:
                job.command = _render_inference_preview_command(launch_config, cwd=repo_root)

        with job.lock:
            if job.stop_requested:
                _mark_job_halted_locked(job)
                stop_before_process = True
            else:
                stop_before_process = False
                job.current = "Starting"
        if stop_before_process:
            _append_job_log_line(job, "[server] Job stopped before process start.", log_handle)
            return

        command_env = _build_inference_command_env(launch_config)
        command = _build_inference_command(launch_config)
        command_text = _render_shell_command(command, cwd=repo_root, env=command_env)
        with job.lock:
            job.command = command_text

        env = os.environ.copy()
        env.update(command_env)
        _append_job_log_line(job, f"[server] Command: {command_text}", log_handle)
        _append_job_log_line(job, "[server] Launching inference process.", log_handle)
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        with job.lock:
            job.process = process
            if job.stop_requested:
                job.current = "Stopping"

        if job.stop_requested:
            jobs_api.terminate_job_process(job, force=False)

        last_output_line: str | None = None
        if process.stdout is not None:
            for raw_line in process.stdout:
                for part in raw_line.split("\r"):
                    line = _clean_process_line(part)
                    if not line:
                        continue
                    last_output_line = line
                    _append_job_log_line(job, line, log_handle)
                    _update_job_progress_from_output(
                        job,
                        line,
                        expected_saving_path_keys,
                        expected_feature_path_keys,
                        saved_feature_paths,
                    )

        return_code = process.wait()
        existing_feature_paths = _existing_expected_feature_paths(expected_feature_paths)
        saved_feature_count = len(existing_feature_paths)
        final_log_line: str | None = None
        with job.lock:
            job.process = None
            job.exit_code = return_code
            if job.stop_requested:
                job.status = "halted"
                job.current = "Stopped"
                job.finished_at_ms = jobs_api._now_ms()
                final_log_line = "[server] Job stopped by user."
            elif return_code == 0 and saved_feature_count == job.total:
                job.status = "completed"
                if job.total > 0:
                    job.processed = saved_feature_count
                job.current = "Done"
                job.finished_at_ms = jobs_api._now_ms()
                final_log_line = "[server] Inference completed successfully."
            else:
                job.status = "failed"
                job.current = "Failed"
                if return_code == 0:
                    missing_stems = [
                        path.stem for path in expected_feature_paths if not path.is_file()
                    ]
                    missing_preview = ", ".join(missing_stems[:3])
                    if len(missing_stems) > 3:
                        missing_preview = f"{missing_preview}, ..."
                    missing_suffix = f" Missing outputs: {missing_preview}." if missing_preview else ""
                    job.error = (
                        f"Inference exited successfully but produced {saved_feature_count}/{job.total} expected "
                        f"feature files.{missing_suffix}"
                    )
                    job.processed = saved_feature_count
                    final_log_line = f"[server] {job.error}"
                else:
                    job.error = last_output_line or f"Inference exited with code {return_code}."
                    final_log_line = f"[server] Process exited with code {return_code}."
                job.finished_at_ms = jobs_api._now_ms()
        if final_log_line:
            _append_job_log_line(job, final_log_line, log_handle)
    except Exception as exc:
        _append_job_log_line(job, f"[server] Runner error: {exc}", log_handle)
        with job.lock:
            job.process = None
            if job.stop_requested:
                job.status = "halted"
                job.current = "Stopped"
            else:
                job.status = "failed"
                job.current = "Failed"
                job.error = str(exc)
            job.finished_at_ms = jobs_api._now_ms()
    finally:
        if log_handle is not None:
            log_handle.close()


def _run_process_features_job(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    _run_single_gpu_post_processing_job(
        job,
        launch_config,
        command=_build_process_features_command(launch_config),
        command_env=_build_process_features_command_env(launch_config),
        launch_log_line="[server] Launching process-features job.",
        success_log_line="[server] Process-features job completed successfully.",
        job_name="Process-features",
    )


def _run_segmentation_job(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    _run_single_gpu_post_processing_job(
        job,
        launch_config,
        command=_build_segmentation_command(launch_config),
        command_env=(
            _build_tracking_command_env()
            if launch_config.get("mode") == SEGMENTATION_MODE_PROBABILITY_MAP
            else _build_process_features_command_env(launch_config)
        ),
        launch_log_line="[server] Launching segmentation job.",
        success_log_line="[server] Segmentation job completed successfully.",
        job_name="Segmentation",
    )


def _run_foreground_probability_map_job(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    _run_single_gpu_post_processing_job(
        job,
        launch_config,
        command=_build_foreground_probability_map_command(launch_config),
        command_env=_build_process_features_command_env(launch_config),
        launch_log_line="[server] Launching foreground probability-map job.",
        success_log_line="[server] Foreground probability-map job completed successfully.",
        job_name="Foreground probability-map",
    )


def _run_tracking_job(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    _run_single_gpu_post_processing_job(
        job,
        launch_config,
        command=_build_tracking_command(launch_config),
        command_env=_build_tracking_command_env(),
        launch_log_line="[server] Launching tracking job.",
        success_log_line="[server] Tracking job completed successfully.",
        job_name="Tracking",
        progress_updater=_update_tracking_job_progress_from_output,
    )


def _run_data_download_job(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    repo_root = get_repo_root()
    download_root = Path(launch_config["download_root"])
    selected_datasets = list(launch_config["selected_datasets"])
    overwrite_existing = bool(launch_config["overwrite_existing"])
    log_handle = _open_job_log(job)

    with job.lock:
        job.command = f"Download public data from {launch_config['manifest_url']} to {download_root}"
        job.working_dir = str(repo_root)

    _append_job_log_line(job, f"[server] Working directory: {repo_root}", log_handle)
    _append_job_log_line(job, f"[server] Manifest: {launch_config['manifest_url']}", log_handle)
    _append_job_log_line(job, f"[server] Download root: {download_root}", log_handle)
    if launch_config.get("skipped_names"):
        skipped_text = ", ".join(launch_config["skipped_names"])
        _append_job_log_line(job, f"[server] Skipping existing datasets: {skipped_text}", log_handle)

    try:
        download_root.mkdir(parents=True, exist_ok=True)
        temp_parent = download_root.parent

        with job.lock:
            if job.stop_requested:
                _mark_job_halted_locked(job)
                stop_before_start = True
            else:
                stop_before_start = False
                job.current = "Starting"
        if stop_before_start:
            _append_job_log_line(job, "[server] Job stopped before launch.", log_handle)
            return

        for dataset in selected_datasets:
            dataset_name = str(dataset["name"])
            archive_url = str(dataset["archiveUrl"])
            target_path = download_root / dataset_name

            with job.lock:
                if job.stop_requested:
                    _mark_job_halted_locked(job)
                    _append_job_log_line(job, "[server] Job stopped by user.", log_handle)
                    return
                job.current = f"Downloading {dataset_name}"

            _append_job_log_line(job, f"[server] Downloading {dataset_name} from {archive_url}", log_handle)

            with tempfile.TemporaryDirectory(prefix=f"spatialdino-data-{dataset_name}-", dir=temp_parent) as tmp:
                tmp_path = Path(tmp)
                archive_path = tmp_path / f"{dataset_name}.zip"
                extract_dir = tmp_path / "extract"
                extract_dir.mkdir(parents=True, exist_ok=True)

                _download_url_to_file(archive_url, archive_path)

                with job.lock:
                    if job.stop_requested:
                        _mark_job_halted_locked(job)
                        _append_job_log_line(job, "[server] Job stopped by user.", log_handle)
                        return
                    job.current = f"Extracting {dataset_name}"

                _append_job_log_line(job, f"[server] Extracting {dataset_name}", log_handle)
                _extract_zip_to_directory(archive_path, extract_dir)

                extracted_source = _select_extracted_dataset_path(extract_dir, dataset_name)
                if not extracted_source.exists():
                    raise RuntimeError(f"Archive for {dataset_name} produced no extractable data.")

                if extracted_source == extract_dir:
                    wrapped_dir = tmp_path / dataset_name
                    wrapped_dir.mkdir(parents=True, exist_ok=True)
                    moved_any = False
                    for child in list(extract_dir.iterdir()):
                        shutil.move(str(child), str(wrapped_dir / child.name))
                        moved_any = True
                    if not moved_any:
                        raise RuntimeError(f"Archive for {dataset_name} is empty after extraction.")
                    extracted_source = wrapped_dir

                if target_path.exists() or target_path.is_symlink():
                    if overwrite_existing:
                        _remove_path(target_path)
                    else:
                        raise RuntimeError(
                            f"Target path already exists for {dataset_name}: {target_path}. Use overwrite to replace it."
                        )

                shutil.move(str(extracted_source), str(target_path))

            with job.lock:
                job.processed = min(job.total, job.processed + 1) if job.total > 0 else job.processed + 1
                job.current = dataset_name
            _append_job_log_line(job, f"[server] Saved dataset {dataset_name} to {target_path}", log_handle)

        with job.lock:
            if job.stop_requested:
                job.status = "halted"
                job.current = "Stopped"
            else:
                job.status = "completed"
                job.current = "Done"
            job.finished_at_ms = jobs_api._now_ms()
        _append_job_log_line(job, "[server] Public data download completed successfully.", log_handle)
    except Exception as exc:
        _append_job_log_line(job, f"[server] Runner error: {exc}", log_handle)
        with job.lock:
            if job.stop_requested:
                job.status = "halted"
                job.current = "Stopped"
            else:
                job.status = "failed"
                job.current = "Failed"
                job.error = str(exc)
            job.finished_at_ms = jobs_api._now_ms()
    finally:
        if log_handle is not None:
            log_handle.close()


def _run_single_gpu_post_processing_job(
    job: jobs_api.JobState,
    launch_config: dict[str, Any],
    *,
    command: list[str],
    command_env: dict[str, str],
    launch_log_line: str,
    success_log_line: str,
    job_name: str,
    progress_updater: Callable[[jobs_api.JobState, str, dict[str, Any]], int] = _update_process_features_job_progress_from_output,
    progress_state: dict[str, Any] | None = None,
) -> None:
    repo_root = get_repo_root()
    log_handle = _open_job_log(job)
    command_text = _render_shell_command(command, cwd=repo_root, env=command_env)
    progress_state = {} if progress_state is None else progress_state
    completed_count = 0

    with job.lock:
        job.command = command_text
        job.working_dir = str(repo_root)

    _append_job_log_line(job, f"[server] Working directory: {repo_root}", log_handle)
    _append_job_log_line(job, f"[server] Command: {command_text}", log_handle)

    try:
        with job.lock:
            if job.stop_requested:
                _mark_job_halted_locked(job)
                stop_before_launch = True
            else:
                stop_before_launch = False
                job.current = "Starting"
        if stop_before_launch:
            _append_job_log_line(job, "[server] Job stopped before launch.", log_handle)
            return

        env = os.environ.copy()
        env.update(command_env)
        _append_job_log_line(job, launch_log_line, log_handle)
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        with job.lock:
            job.process = process
            if job.stop_requested:
                job.current = "Stopping"

        if job.stop_requested:
            jobs_api.terminate_job_process(job, force=False)

        last_output_line: str | None = None
        if process.stdout is not None:
            for raw_line in process.stdout:
                for part in raw_line.split("\r"):
                    line = _clean_process_line(part)
                    if not line:
                        continue
                    last_output_line = line
                    _append_job_log_line(job, line, log_handle)
                    completed_count = progress_updater(job, line, progress_state)

        return_code = process.wait()
        final_log_line: str | None = None
        with job.lock:
            job.process = None
            job.exit_code = return_code
            if job.stop_requested:
                job.status = "halted"
                job.current = "Stopped"
                job.finished_at_ms = jobs_api._now_ms()
                final_log_line = "[server] Job stopped by user."
            elif return_code == 0 and completed_count == job.total:
                job.status = "completed"
                job.processed = completed_count
                job.current = "Done"
                job.finished_at_ms = jobs_api._now_ms()
                final_log_line = success_log_line
            else:
                job.status = "failed"
                job.current = "Failed"
                if return_code == 0:
                    job.error = f"{job_name} job exited successfully but only completed {completed_count}/{job.total} progress steps."
                else:
                    job.error = last_output_line or f"{job_name} job exited with code {return_code}."
                job.finished_at_ms = jobs_api._now_ms()
                final_log_line = f"[server] {job.error}"
        if final_log_line:
            _append_job_log_line(job, final_log_line, log_handle)
    except Exception as exc:
        _append_job_log_line(job, f"[server] Runner error: {exc}", log_handle)
        with job.lock:
            job.process = None
            if job.stop_requested:
                job.status = "halted"
                job.current = "Stopped"
            else:
                job.status = "failed"
                job.current = "Failed"
                job.error = str(exc)
            job.finished_at_ms = jobs_api._now_ms()
    finally:
        if log_handle is not None:
            log_handle.close()


def _launch_inference_job_thread(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    thread = threading.Thread(
        target=_run_inference_job,
        args=(job, launch_config),
        daemon=True,
        name=f"inference-job-{job.job_id}",
    )
    thread.start()


def _launch_process_features_job_thread(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    thread = threading.Thread(
        target=_run_process_features_job,
        args=(job, launch_config),
        daemon=True,
        name=f"process-features-job-{job.job_id}",
    )
    thread.start()


def _launch_segmentation_job_thread(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    thread = threading.Thread(
        target=_run_segmentation_job,
        args=(job, launch_config),
        daemon=True,
        name=f"segmentation-job-{job.job_id}",
    )
    thread.start()


def _launch_foreground_probability_map_job_thread(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    thread = threading.Thread(
        target=_run_foreground_probability_map_job,
        args=(job, launch_config),
        daemon=True,
        name=f"foreground-probability-map-job-{job.job_id}",
    )
    thread.start()


def _launch_tracking_job_thread(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    thread = threading.Thread(
        target=_run_tracking_job,
        args=(job, launch_config),
        daemon=True,
        name=f"tracking-job-{job.job_id}",
    )
    thread.start()


def _launch_data_download_job_thread(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    thread = threading.Thread(
        target=_run_data_download_job,
        args=(job, launch_config),
        daemon=True,
        name=f"data-download-job-{job.job_id}",
    )
    thread.start()


def load_index_html(dist_dir: Path) -> str | None:
    index_path = dist_dir / "index.html"
    if not index_path.is_file():
        return None
    return index_path.read_text(encoding="utf-8")


def render_index_html(index_html: str, *, client_host: str | None = None) -> str:
    hostname = html.escape(get_server_hostname())
    session_label = html.escape(classify_client_session(client_host))
    return (
        index_html.replace("__SPATIALDINO_SERVER_HOSTNAME__", hostname).replace(
            "__SPATIALDINO_SESSION_LABEL__", session_label
        )
    )


app = FastAPI(title="spatialDINO")

api = APIRouter(prefix="/api")


@api.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api.get("/session")
def session_info(request: Request) -> dict[str, str]:
    client_host = request.client.host if request.client else None
    return {
        "serverHostname": get_server_hostname(),
        "sessionLabel": classify_client_session(client_host),
    }


@api.get("/data/options")
def data_options() -> dict[str, object]:
    try:
        datasets = _load_public_data_manifest()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "manifestUrl": get_public_data_manifest_url(),
        "downloadRoot": str(get_public_data_dir()),
        "datasets": [{"name": item["name"]} for item in datasets],
    }


@api.post("/data/download")
def run_data_download(
    payload: RunDataDownloadRequest,
    x_spatialdino_clientid: str | None = Header(None),
) -> dict[str, Any]:
    client_id = jobs_api._require_client_id(x_spatialdino_clientid)
    validation, launch_config = _build_data_download_launch_config(payload)
    if launch_config is None:
        return {"submitted": False, **validation}

    download_root = Path(launch_config["download_root"])
    label = "Public data download"
    job = jobs_api.JobState(
        job_id=str(uuid.uuid4()),
        owner_client_id=client_id,
        type="data-download",
        status="running",
        processed=0,
        total=len(launch_config["selected_datasets"]),
        current="Queued",
        label=label,
        save_dir=str(download_root),
        datasets=[
            {"source_dir": str(item["archiveUrl"]), "save_to": str(item["name"])}
            for item in launch_config["selected_datasets"]
        ],
    )
    jobs_api.register_job(job)

    try:
        _launch_data_download_job_thread(job, launch_config)
    except Exception as exc:
        jobs_api.unregister_job(job.job_id)
        return {
            "submitted": False,
            **_invalid_data_download("submit_failed", f"Could not start public data download: {exc}"),
        }

    skipped_names = list(launch_config.get("skipped_names", []))
    if skipped_names:
        return {
            "submitted": True,
            "jobId": job.job_id,
            "message": (
                f"Public data download submitted. Skipping {len(skipped_names)} existing dataset"
                f"{'' if len(skipped_names) == 1 else 's'}."
            ),
        }
    return {"submitted": True, "jobId": job.job_id, "message": "Public data download submitted."}


@api.get("/inference/options")
def inference_options() -> dict[str, object]:
    gpu_status = get_nvidia_gpu_memory()
    return {
        "gpus": [{"index": gpu["index"], "name": gpu["name"]} for gpu in gpu_status.get("gpus", [])],
        "gpuError": gpu_status.get("error"),
        "nvidiaSmiAvailable": bool(gpu_status.get("nvidiaSmiAvailable")),
        "backboneWeights": list_backbone_weights(),
    }


@api.post("/inference/download-backbone")
def download_inference_backbone(payload: DownloadBackboneWeightsRequest) -> dict[str, Any]:
    repo_root = get_repo_root()
    models_dir = get_backbone_weights_dir()
    target_path = get_default_inference_backbone_path()

    if not _is_relative_to(models_dir, repo_root) or not _is_relative_to(target_path, repo_root):
        raise HTTPException(status_code=500, detail="Backbone weights path must stay inside the repo root.")

    with _default_inference_backbone_download_lock:
        if models_dir.exists() and not models_dir.is_dir():
            raise HTTPException(status_code=500, detail="Backbone weights directory is not a folder.")

        already_exists = target_path.exists()
        if already_exists:
            if not target_path.is_file():
                raise HTTPException(status_code=500, detail="Backbone weights target exists but is not a file.")
            if not payload.overwrite:
                return {
                    "downloaded": False,
                    "requiresOverwriteConfirmation": True,
                    "message": "backbone.pth already exists. Do you want to overwrite it?",
                    "targetPath": str(target_path),
                    "backboneWeight": DEFAULT_INFERENCE_BACKBONE_RELATIVE_PATH,
                }

        try:
            models_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not create the weights directory: {exc}") from exc

        try:
            _download_url_to_file(DEFAULT_INFERENCE_BACKBONE_URL, target_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not download backbone weights: {exc}") from exc

    return {
        "downloaded": True,
        "message": f"Saved backbone weights to {target_path}.",
        "targetPath": str(target_path),
        "backboneWeight": DEFAULT_INFERENCE_BACKBONE_RELATIVE_PATH,
        "alreadyExisted": already_exists,
    }


@api.post("/inference/validate-input")
def validate_inference_input(payload: ValidateInferenceInputRequest) -> dict[str, Any]:
    return validate_inference_input_folder(payload.path)


@api.post("/post-processing/process-features/validate-input")
def validate_process_features_input(payload: ValidateProcessFeaturesInputRequest) -> dict[str, Any]:
    return validate_process_features_input_folder(payload.path)


@api.post("/post-processing/segmentation/validate-input")
def validate_segmentation_input(payload: ValidateProcessFeaturesInputRequest) -> dict[str, Any]:
    return validate_segmentation_input_folder(payload.path)


@api.post("/post-processing/probability-map/validate-input")
def validate_probability_map_input(payload: ValidateProcessFeaturesInputRequest) -> dict[str, Any]:
    return validate_probability_map_input_folder(payload.path)


@api.post("/post-processing/probability-map/preview/metadata")
def probability_map_preview_metadata_endpoint(payload: ProbabilityMapPreviewMetadataRequest) -> dict[str, Any]:
    return probability_map_preview_metadata(payload.input_path)


@api.post("/post-processing/probability-map/preview/image")
def probability_map_preview_image_endpoint(payload: ProbabilityMapPreviewImageRequest) -> dict[str, Any]:
    return probability_map_preview_image(payload)


@api.post("/post-processing/tracking/validate-input")
def validate_tracking_input(payload: ValidateTrackingInputRequest) -> dict[str, Any]:
    return validate_tracking_input_folder(payload.path)


@api.post("/post-processing/tracking/validate-segmentation-folder")
def validate_tracking_segmentation(payload: ValidateTrackingSegmentationFolderRequest) -> dict[str, Any]:
    return validate_tracking_segmentation_folder(payload.input_path, payload.segmentation_path)


@api.post("/post-processing/process-features/run")
def run_process_features(
    payload: RunProcessFeaturesRequest,
    x_spatialdino_clientid: str | None = Header(None),
) -> dict[str, Any]:
    client_id = jobs_api._require_client_id(x_spatialdino_clientid)
    validation, launch_config = _build_process_features_launch_config(payload)
    if launch_config is None:
        return {"submitted": False, **validation}

    input_path = Path(launch_config["input_path"])
    output_path = Path(launch_config["output_path"])
    label = f"Process features {input_path.name}".strip()
    job = jobs_api.JobState(
        job_id=str(uuid.uuid4()),
        owner_client_id=client_id,
        type="process-features",
        status="running",
        processed=0,
        total=int(launch_config["subfolder_count"]),
        current="Queued",
        label=label,
        save_dir=str(output_path.parent),
        datasets=[{"source_dir": str(input_path), "save_to": output_path.name or str(output_path)}],
    )
    jobs_api.register_job(job)

    try:
        _launch_process_features_job_thread(job, launch_config)
    except Exception as exc:
        jobs_api.unregister_job(job.job_id)
        return {
            "submitted": False,
            **_invalid_process_features_run("submit_failed", f"Could not start process-features job: {exc}"),
        }

    return {"submitted": True, "jobId": job.job_id, "message": "Process-features job submitted."}


@api.post("/post-processing/foreground-probability-map/run")
def run_foreground_probability_map(
    payload: RunForegroundProbabilityMapRequest,
    x_spatialdino_clientid: str | None = Header(None),
) -> dict[str, Any]:
    client_id = jobs_api._require_client_id(x_spatialdino_clientid)
    validation, launch_config = _build_foreground_probability_map_launch_config(payload)
    if launch_config is None:
        return {"submitted": False, **validation}

    input_path = Path(launch_config["input_path"])
    output_path = Path(launch_config["output_path"])
    label = f"Foreground probability map {input_path.name}".strip()
    job = jobs_api.JobState(
        job_id=str(uuid.uuid4()),
        owner_client_id=client_id,
        type="foreground-probability-map",
        status="running",
        processed=0,
        total=int(launch_config.get("progress_total", launch_config["subfolder_count"])),
        current="Queued",
        label=label,
        save_dir=str(output_path.parent),
        datasets=[{"source_dir": str(input_path), "save_to": output_path.name or str(output_path)}],
    )
    jobs_api.register_job(job)

    try:
        _launch_foreground_probability_map_job_thread(job, launch_config)
    except Exception as exc:
        jobs_api.unregister_job(job.job_id)
        return {
            "submitted": False,
            **_invalid_process_features_run("submit_failed", f"Could not start foreground probability-map job: {exc}"),
        }

    return {"submitted": True, "jobId": job.job_id, "message": "Foreground probability-map job submitted."}


@api.post("/post-processing/segmentation/run")
def run_segmentation(
    payload: RunSegmentationRequest,
    x_spatialdino_clientid: str | None = Header(None),
) -> dict[str, Any]:
    client_id = jobs_api._require_client_id(x_spatialdino_clientid)
    validation, launch_config = _build_segmentation_launch_config(payload)
    if launch_config is None:
        return {"submitted": False, **validation}

    input_path = Path(launch_config["input_path"])
    output_path = Path(launch_config["output_path"])
    label = f"Segmentation {input_path.name}".strip()
    job = jobs_api.JobState(
        job_id=str(uuid.uuid4()),
        owner_client_id=client_id,
        type="segmentation",
        status="running",
        processed=0,
        total=int(launch_config.get("progress_total", launch_config["subfolder_count"])),
        current="Queued",
        label=label,
        save_dir=str(output_path.parent),
        datasets=[{"source_dir": str(input_path), "save_to": output_path.name or str(output_path)}],
    )
    jobs_api.register_job(job)

    try:
        _launch_segmentation_job_thread(job, launch_config)
    except Exception as exc:
        jobs_api.unregister_job(job.job_id)
        return {
            "submitted": False,
            **_invalid_process_features_run("submit_failed", f"Could not start segmentation job: {exc}"),
        }

    return {"submitted": True, "jobId": job.job_id, "message": "Segmentation job submitted."}


@api.post("/post-processing/tracking/run")
def run_tracking(
    payload: RunTrackingRequest,
    x_spatialdino_clientid: str | None = Header(None),
) -> dict[str, Any]:
    client_id = jobs_api._require_client_id(x_spatialdino_clientid)
    validation, launch_config = _build_tracking_launch_config(payload)
    if launch_config is None:
        return {"submitted": False, **validation}

    input_path = Path(launch_config["input_path"])
    output_path = Path(launch_config["output_path"])
    label = f"Tracking {input_path.name}".strip()
    job = jobs_api.JobState(
        job_id=str(uuid.uuid4()),
        owner_client_id=client_id,
        type="tracking",
        status="running",
        processed=0,
        total=int(launch_config["progress_total"]),
        current="Queued",
        label=label,
        save_dir=str(output_path.parent),
        datasets=[{"source_dir": str(input_path), "save_to": output_path.name or str(output_path)}],
    )
    jobs_api.register_job(job)

    try:
        _launch_tracking_job_thread(job, launch_config)
    except Exception as exc:
        jobs_api.unregister_job(job.job_id)
        return {
            "submitted": False,
            **_invalid_process_features_run("submit_failed", f"Could not start tracking job: {exc}"),
        }

    return {"submitted": True, "jobId": job.job_id, "message": "Tracking job submitted."}


@api.post("/inference/run")
def run_inference(
    payload: RunInferenceRequest,
    x_spatialdino_clientid: str | None = Header(None),
) -> dict[str, Any]:
    client_id = jobs_api._require_client_id(x_spatialdino_clientid)
    validation, launch_config = _build_inference_launch_config(payload)
    if launch_config is None:
        return {"submitted": False, **validation}

    output_path = Path(launch_config["output_path"])
    label = f"Inference {Path(launch_config['input_path']).name}".strip()
    save_dir = str(output_path.parent)
    save_to = output_path.name or str(output_path)
    job = jobs_api.JobState(
        job_id=str(uuid.uuid4()),
        owner_client_id=client_id,
        type="inference",
        status="running",
        processed=0,
        total=int(launch_config["selected_file_count"]),
        current="Queued",
        label=label,
        save_dir=save_dir,
        datasets=[{"source_dir": str(launch_config["input_path"]), "save_to": save_to}],
        roi={
            "x0": int(launch_config["effective_crop_params"][4]),
            "x1": int(launch_config["effective_crop_params"][5]),
            "y0": int(launch_config["effective_crop_params"][2]),
            "y1": int(launch_config["effective_crop_params"][3]),
            "z0": int(launch_config["effective_crop_params"][0]),
            "z1": int(launch_config["effective_crop_params"][1]),
        },
        overwrite=bool(payload.overwrite),
    )
    jobs_api.register_job(job)

    try:
        _launch_inference_job_thread(job, launch_config)
    except Exception as exc:
        jobs_api.unregister_job(job.job_id)
        return {
            "submitted": False,
            **_invalid_inference_run("submit_failed", f"Could not start inference: {exc}"),
        }

    return {"submitted": True, "jobId": job.job_id, "message": "Inference job submitted."}


@api.post("/inference/command-preview")
def inference_command_preview(payload: RunInferenceRequest) -> dict[str, Any]:
    validation, launch_config = _build_inference_launch_config(payload, require_overwrite_confirmation=False)
    if launch_config is None:
        return validation

    repo_root = get_repo_root()
    overwrite_warning = launch_config.get("overwrite_warning")
    return {
        "valid": True,
        "workingDirectory": str(repo_root),
        "command": _render_inference_preview_command(launch_config, cwd=repo_root),
        "requiresOverwriteConfirmation": bool(overwrite_warning),
        "overwriteMessage": overwrite_warning["message"] if overwrite_warning else None,
    }


@api.get("/status/cpu")
async def status_cpu() -> dict:
    return await get_cpu_activity()


@api.get("/status/gpus")
def status_gpus() -> dict:
    return get_nvidia_gpu_memory()


api.include_router(fs_router)
api.include_router(jobs_router)

app.include_router(api)


@app.get("/")
def root(request: Request) -> HTMLResponse:
    dist_dir = get_dist_dir()
    index_html = load_index_html(dist_dir)
    if index_html is None:
        msg = (
            "spatialDINO backend is running, but the frontend is not built.\n"
            "Build the frontend with `cd apps/web && npm install && npm run build`.\n"
        )
        return HTMLResponse(f"<pre>{html.escape(msg)}</pre>", status_code=503)
    client_host = request.client.host if request.client else None
    return HTMLResponse(render_index_html(index_html, client_host=client_host))


@app.get("/{path:path}")
def spa_fallback(path: str, request: Request):
    if path.startswith("api/"):
        raise HTTPException(status_code=404)

    dist_dir = get_dist_dir()
    if not dist_dir.is_dir():
        raise HTTPException(status_code=404)

    requested = (dist_dir / path).resolve()
    try:
        requested.relative_to(dist_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404) from exc

    if requested.is_file():
        return FileResponse(requested)

    index_html = load_index_html(dist_dir)
    if index_html is None:
        raise HTTPException(status_code=404)
    client_host = request.client.host if request.client else None
    return HTMLResponse(render_index_html(index_html, client_host=client_host))
