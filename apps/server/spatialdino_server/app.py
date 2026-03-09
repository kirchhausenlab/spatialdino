from __future__ import annotations

import html
import ipaddress
import os
import socket
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from spatialdino_server.fs_api import router as fs_router
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
