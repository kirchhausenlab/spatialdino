from __future__ import annotations

import html
import ipaddress
import os
import re
import shlex
import shutil
import socket
import subprocess
from functools import lru_cache
from pathlib import Path
import sys
import threading
from typing import Any, TextIO
import urllib.request
import uuid

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import tifffile

from spatialdino_server.fs_api import router as fs_router
from spatialdino_server.fs_roots import _configured_fs_roots_from_env
from spatialdino_server import jobs_api
from spatialdino_server.jobs_api import router as jobs_router
from spatialdino_server.status import get_cpu_activity, get_nvidia_gpu_memory


JOB_LOGS_DIRNAME = ".spatialdino_job_logs"
DEFAULT_INFERENCE_BACKBONE_FILENAME = "backbone.pth"
DEFAULT_INFERENCE_BACKBONE_RELATIVE_PATH = f"models/{DEFAULT_INFERENCE_BACKBONE_FILENAME}"
DEFAULT_INFERENCE_BACKBONE_URL = (
    "https://spatialdino.s3.us-east-1.amazonaws.com/models/spatial_dino/step%3D244999/backbone.pth"
)
_default_inference_backbone_download_lock = threading.Lock()


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


class ValidateInferenceInputRequest(BaseModel):
    path: str = Field(..., min_length=1)


class ValidateProcessFeaturesInputRequest(BaseModel):
    path: str = Field(..., min_length=1)


class DownloadBackboneWeightsRequest(BaseModel):
    overwrite: bool = False


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
    gpu_indices: list[int] = Field(default_factory=list)
    upsample_factor: float | None = None
    route: str = Field("full", min_length=1)
    precision: str = Field("bfloat16", min_length=1)
    crop_bounds: InferenceCropBoundsRequest = Field(default_factory=InferenceCropBoundsRequest)
    anisotropy: InferenceAxisRequest = Field(default_factory=InferenceAxisRequest)
    file_range: InferenceFileRangeRequest = Field(default_factory=InferenceFileRangeRequest)
    normalization_mode: str = Field("per_volume", min_length=1)
    global_hist_min: float | None = None
    global_hist_max: float | None = None
    overwrite: bool = False


class RunProcessFeaturesRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    gpu_index: int | None = None
    save_high_resolution_features: bool = False
    high_resolution_save_format: str = Field(".tif", min_length=1)
    save_pca: bool = False
    pca_components: int = Field(3, ge=1)
    pca_save_format: str = Field(".tif", min_length=1)


class RunSegmentationRequest(BaseModel):
    input_path: str = Field(..., min_length=1)
    gpu_index: int | None = None
    enable_voronoi_otsu: bool = False
    gaussian_blur_sigma: int = Field(3, ge=0)
    rolling_ball_radius: float = Field(10.0, ge=0.0)


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
    tiff_paths: list[Path] = []
    with os.scandir(input_path) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith((".tif", ".tiff")):
                continue
            tiff_paths.append(Path(entry.path))

    tiff_paths.sort(key=lambda path: (path.name.casefold(), path.name))
    return tiff_paths


def _selected_inference_tiff_paths(input_path: Path, file_start: int, file_end: int | None) -> list[Path]:
    return _list_inference_tiff_paths(input_path)[file_start:file_end]


def _list_child_directories(input_path: Path) -> list[Path]:
    child_dirs: list[Path] = []
    with os.scandir(input_path) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            child_dirs.append(Path(entry.path))

    child_dirs.sort(key=lambda path: (path.name.casefold(), path.name))
    return child_dirs


def _invalid_process_features_input(reason_code: str, message: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reasonCode": reason_code,
        "message": message,
    }


def validate_process_features_input_folder(raw_path: str) -> dict[str, Any]:
    input_path = _resolve_allowed_inference_path(raw_path)

    if not input_path.exists():
        return _invalid_process_features_input("missing", "Input folder does not exist.")
    if not input_path.is_dir():
        return _invalid_process_features_input("not_directory", "Input path is not a folder.")

    subfolders = _list_child_directories(input_path)
    if not subfolders:
        return _invalid_process_features_input("no_subfolders", "Input folder contains no subfolders.")

    for subfolder in subfolders:
        missing_files: list[str] = []
        if not subfolder.joinpath("lr_feats.npy").is_file():
            missing_files.append("lr_feats.npy")
        if not subfolder.joinpath("volume_unnorm.tif").is_file():
            missing_files.append("volume_unnorm.tif")

        if missing_files:
            if len(missing_files) == 1:
                missing_text = missing_files[0]
            else:
                missing_text = ", ".join(missing_files[:-1]) + f", and {missing_files[-1]}"
            return _invalid_process_features_input(
                "missing_required_files",
                f"Subfolder {subfolder.name} is missing {missing_text}.",
            )

    subfolder_count = len(subfolders)
    return {
        "valid": True,
        "message": f"Valid feature folder. Found {subfolder_count} subfolder{'s' if subfolder_count != 1 else ''}.",
        "subfolderCount": subfolder_count,
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
PROCESS_FEATURES_PROCESSING_RE = re.compile(r"^\[(?:process-features|segmentation)\] Processing (.+?) \((\d+)/(\d+)\)$")
PROCESS_FEATURES_COMPLETED_RE = re.compile(r"^\[(?:process-features|segmentation)\] Completed (.+)$")
DEFAULT_INFERENCE_OMP_NUM_THREADS = 4
NORM_PER_VOL_MIN_RE = re.compile(r"^Global hist min:\s*(.+?)\s*$", re.MULTILINE)
NORM_PER_VOL_MAX_RE = re.compile(r"^Global hist max:\s*(.+?)\s*$", re.MULTILINE)
NORMALIZATION_MODE_PER_VOLUME = "per_volume"
NORMALIZATION_MODE_GLOBAL_AUTO = "global_auto"
NORMALIZATION_MODE_GLOBAL_MANUAL = "global_manual"
PROCESS_FEATURES_SAVE_FORMATS = {".npy", ".tif"}
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


def _overwrite_confirmation(output_path: Path, count: int, preview: list[str]) -> dict[str, Any]:
    message = "Output folder is not empty. Confirm overwrite to erase its contents and continue."
    return {
        "valid": True,
        "message": message,
        "requiresOverwriteConfirmation": True,
        "outputPath": str(output_path),
        "outputEntryCount": count,
        "outputEntriesPreview": preview,
    }


def _coerce_start(value: int | None) -> int:
    return 0 if value is None else value


def _coerce_crop_end(value: int | None) -> int:
    return 0 if value is None else value


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
    subfolder_count = int(input_validation["subfolderCount"])
    return (
        {
            "valid": True,
            "message": "Validation passed.",
            "subfolderCount": subfolder_count,
        },
        {
            "input_path": input_path,
            "gpu_index": requested_gpu,
            "save_pca": bool(payload.save_pca),
            "pca_components": int(payload.pca_components),
            "pca_save_format": payload.pca_save_format,
            "save_high_resolution_features": bool(payload.save_high_resolution_features),
            "high_resolution_save_format": payload.high_resolution_save_format,
            "subfolder_count": subfolder_count,
        },
    )


def _build_segmentation_launch_config(
    payload: RunSegmentationRequest,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    input_validation = validate_process_features_input_folder(payload.input_path)
    if not input_validation["valid"]:
        return _invalid_process_features_run(input_validation["reasonCode"], input_validation["message"]), None

    if not payload.enable_voronoi_otsu:
        return _invalid_process_features_run(
            "segmentation_disabled",
            "Enable Voronoi-Otsu segmentation.",
        ), None

    if payload.gpu_index is None:
        return _invalid_process_features_run("missing_gpu_selection", "Select one GPU."), None

    requested_gpu = int(payload.gpu_index)
    gpu_status = get_nvidia_gpu_memory()
    available_gpus = {int(gpu["index"]) for gpu in gpu_status.get("gpus", [])}
    if requested_gpu not in available_gpus:
        return _invalid_process_features_run("invalid_gpu_selection", "Selected GPU is not available on this server."), None

    input_path = _resolve_allowed_inference_path(payload.input_path)
    subfolder_count = int(input_validation["subfolderCount"])
    return (
        {
            "valid": True,
            "message": "Validation passed.",
            "subfolderCount": subfolder_count,
        },
        {
            "input_path": input_path,
            "gpu_index": requested_gpu,
            "enable_voronoi_otsu": bool(payload.enable_voronoi_otsu),
            "gaussian_blur_sigma": int(payload.gaussian_blur_sigma),
            "rolling_ball_radius": float(payload.rolling_ball_radius),
            "subfolder_count": subfolder_count,
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

    requested_gpus = sorted(set(int(index) for index in payload.gpu_indices))
    if not requested_gpus:
        return _invalid_inference_run("missing_gpu_selection", "Select at least one GPU."), None

    available_gpu_indices = {
        int(gpu["index"]) for gpu in get_nvidia_gpu_memory().get("gpus", []) if "index" in gpu
    }
    if any(index not in available_gpu_indices for index in requested_gpus):
        return _invalid_inference_run("invalid_gpu_selection", "Selected GPUs are not available on the server."), None

    upsample_factor = payload.upsample_factor
    if upsample_factor is None or upsample_factor < 1.0:
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

    normalization_error, normalization_config = _validate_normalization_payload(payload)
    if normalization_error is not None or normalization_config is None:
        return normalization_error, None

    shape = input_validation["shape"]
    raw_shape = (int(shape["z"]), int(shape["y"]), int(shape["x"]))
    crop_start_x = _coerce_start(payload.crop_bounds.x_start)
    crop_start_y = _coerce_start(payload.crop_bounds.y_start)
    crop_start_z = _coerce_start(payload.crop_bounds.z_start)
    crop_end_x = _coerce_crop_end(payload.crop_bounds.x_end)
    crop_end_y = _coerce_crop_end(payload.crop_bounds.y_end)
    crop_end_z = _coerce_crop_end(payload.crop_bounds.z_end)

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
    file_end = payload.file_range.end
    file_count = int(input_validation["fileCount"])
    effective_file_end = file_count if file_end is None else file_end
    if file_start < 0 or file_start > file_count:
        return _invalid_inference_run("invalid_file_range", "Start file must be between 0 and the number of files."), None
    if file_end is not None and (file_end < 0 or file_end > file_count):
        return _invalid_inference_run("invalid_file_range", "End file must be between 0 and the number of files."), None
    if effective_file_end <= file_start:
        return _invalid_inference_run("empty_file_selection", "Chosen files leave zero files to process."), None

    selected_input_paths = _selected_inference_tiff_paths(input_path, file_start, file_end)
    if not selected_input_paths:
        return _invalid_inference_run("empty_file_selection", "Chosen files leave zero files to process."), None
    selected_stems = [path.stem for path in selected_input_paths]

    overwrite_warning: dict[str, Any] | None = None
    output_entry_count, output_preview = _summarize_directory(output_path)
    if output_entry_count > 0 and not payload.overwrite:
        overwrite_warning = _overwrite_confirmation(output_path, output_entry_count, output_preview)
        if require_overwrite_confirmation:
            return overwrite_warning, None

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
            "gpu_indices": requested_gpus,
            "upsample_factor": float(upsample_factor),
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
        f"upsample_factor={launch_config['upsample_factor']}",
        f"isotropic_scale_factor=[{anisotropy_z},{anisotropy_y},{anisotropy_x}]",
        f"inference_route={launch_config['inference_route']}",
        f"dtype={launch_config['dtype']}",
    ]
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
        "--pca-components",
        str(launch_config["pca_components"]),
        "--pca-format",
        launch_config["pca_save_format"],
        "--high-resolution-format",
        launch_config["high_resolution_save_format"],
    ]
    if launch_config["save_pca"]:
        command.append("--save-pca")
    if launch_config["save_high_resolution_features"]:
        command.append("--save-high-resolution-features")
    return command


def _build_segmentation_command(launch_config: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "scripts/post_processing/segmentation.py",
        "--input-path",
        str(launch_config["input_path"]),
        "--gaussian-blur-sigma",
        str(launch_config["gaussian_blur_sigma"]),
        "--rolling-ball-radius",
        str(launch_config["rolling_ball_radius"]),
    ]
    if launch_config["enable_voronoi_otsu"]:
        command.append("--enable-voronoi-otsu")
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
    return output_path / "norm_per_vol.txt"


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
    return [output_path / stem / "lr_feats.npy" for stem in selected_stems]


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
    expected_output_dirs: set[str],
    expected_feature_paths: set[str],
    saved_feature_paths: set[str],
) -> None:
    saved_match = SAVED_FEATURES_RE.search(line)
    if saved_match:
        saved_path_text = saved_match.group(1).strip()
        saved_feature_key = _canonicalize_runtime_path(saved_path_text)
        if saved_feature_key not in expected_feature_paths:
            return
        save_name = Path(saved_path_text).parent.name or saved_path_text
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
        if save_path_key not in expected_output_dirs:
            return
        save_name = Path(save_path_text).name or save_path_text
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
    completed_subfolders: set[str],
) -> None:
    processing_match = PROCESS_FEATURES_PROCESSING_RE.search(line)
    if processing_match:
        subfolder_name = processing_match.group(1).strip()
        with job.lock:
            if job.status == "running" and not job.stop_requested:
                job.current = f"Processing {subfolder_name}"
        return

    completed_match = PROCESS_FEATURES_COMPLETED_RE.search(line)
    if not completed_match:
        return

    subfolder_name = completed_match.group(1).strip()
    with job.lock:
        if job.status != "running" or job.stop_requested or subfolder_name in completed_subfolders:
            return
        completed_subfolders.add(subfolder_name)
        job.processed = min(job.total, len(completed_subfolders)) if job.total > 0 else len(completed_subfolders)
        job.current = subfolder_name


def _run_inference_job(job: jobs_api.JobState, launch_config: dict[str, Any]) -> None:
    output_path = Path(launch_config["output_path"])
    expected_feature_paths = _expected_feature_paths(output_path, launch_config["selected_stems"])
    expected_feature_path_keys = {_canonicalize_runtime_path(path) for path in expected_feature_paths}
    expected_output_dir_keys = {_canonicalize_runtime_path(path.parent) for path in expected_feature_paths}
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
            output_path.mkdir(parents=True, exist_ok=True)
            _clear_directory_contents(output_path)

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
                        expected_output_dir_keys,
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
                        path.parent.name for path in expected_feature_paths if not path.is_file()
                    ]
                    missing_preview = ", ".join(missing_stems[:3])
                    if len(missing_stems) > 3:
                        missing_preview = f"{missing_preview}, ..."
                    missing_suffix = f" Missing outputs: {missing_preview}." if missing_preview else ""
                    job.error = (
                        f"Inference exited successfully but produced {saved_feature_count}/{job.total} expected "
                        f"lr_feats.npy files.{missing_suffix}"
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
        command_env=_build_process_features_command_env(launch_config),
        launch_log_line="[server] Launching segmentation job.",
        success_log_line="[server] Segmentation job completed successfully.",
        job_name="Segmentation",
    )


def _run_single_gpu_post_processing_job(
    job: jobs_api.JobState,
    launch_config: dict[str, Any],
    *,
    command: list[str],
    command_env: dict[str, str],
    launch_log_line: str,
    success_log_line: str,
    job_name: str,
) -> None:
    repo_root = get_repo_root()
    log_handle = _open_job_log(job)
    command_text = _render_shell_command(command, cwd=repo_root, env=command_env)
    completed_subfolders: set[str] = set()

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
                    _update_process_features_job_progress_from_output(job, line, completed_subfolders)

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
            elif return_code == 0 and len(completed_subfolders) == job.total:
                job.status = "completed"
                job.processed = len(completed_subfolders)
                job.current = "Done"
                job.finished_at_ms = jobs_api._now_ms()
                final_log_line = success_log_line
            else:
                job.status = "failed"
                job.current = "Failed"
                if return_code == 0:
                    job.error = f"{job_name} job exited successfully but only completed {len(completed_subfolders)}/{job.total} subfolders."
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
        save_dir=str(input_path),
        datasets=[{"source_dir": str(input_path), "save_to": input_path.name or str(input_path)}],
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
    label = f"Segmentation {input_path.name}".strip()
    job = jobs_api.JobState(
        job_id=str(uuid.uuid4()),
        owner_client_id=client_id,
        type="segmentation",
        status="running",
        processed=0,
        total=int(launch_config["subfolder_count"]),
        current="Queued",
        label=label,
        save_dir=str(input_path),
        datasets=[{"source_dir": str(input_path), "save_to": input_path.name or str(input_path)}],
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
