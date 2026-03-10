from __future__ import annotations

import html
import ipaddress
import os
import socket
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import tifffile

from spatialdino_server.fs_api import router as fs_router
from spatialdino_server.fs_roots import _configured_fs_roots_from_env
from spatialdino_server.jobs_api import router as jobs_router
from spatialdino_server.status import get_cpu_activity, get_nvidia_gpu_memory


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
    models_dir = (repo_root / "models").resolve()
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


class ValidateInferenceInputRequest(BaseModel):
    path: str = Field(..., min_length=1)


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


def validate_inference_input_folder(raw_path: str) -> dict[str, Any]:
    input_path = _resolve_allowed_inference_path(raw_path)

    if not input_path.exists():
        return _invalid_inference_input("missing", "Input folder does not exist.")
    if not input_path.is_dir():
        return _invalid_inference_input("not_directory", "Input path is not a folder.")

    tiff_paths: list[Path] = []
    with os.scandir(input_path) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith((".tif", ".tiff")):
                continue
            tiff_paths.append(Path(entry.path))

    tiff_paths.sort(key=lambda path: (path.name.casefold(), path.name))
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


@api.post("/inference/validate-input")
def validate_inference_input(payload: ValidateInferenceInputRequest) -> dict[str, Any]:
    return validate_inference_input_folder(payload.path)


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
