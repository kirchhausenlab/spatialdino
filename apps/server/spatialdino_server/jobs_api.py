from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/jobs")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _require_client_id(value: str | None) -> str:
    text = (value or "").strip()
    if len(text) < 8 or len(text) > 128:
        raise HTTPException(status_code=400, detail="Missing X-SpatialDINO-ClientId header.")
    if "\x00" in text:
        raise HTTPException(status_code=400, detail="Invalid client id.")
    return text


class CancelJobRequest(BaseModel):
    job_id: str = Field(..., min_length=1)


class ClearJobsRequest(BaseModel):
    keep_running: bool = True


class RemoveJobRequest(BaseModel):
    job_id: str = Field(..., min_length=1)


@dataclass
class JobState:
    job_id: str
    owner_client_id: str
    type: str = "job"
    status: str = "running"
    created_at_ms: int = field(default_factory=_now_ms)
    finished_at_ms: int | None = None
    processed: int = 0
    total: int = 0
    error: str | None = None
    current: str | None = None
    label: str | None = None
    save_dir: str | None = None
    datasets: list[dict[str, str]] = field(default_factory=list)
    roi: dict[str, int] | None = None
    mask_cval: int | None = None
    added_padding: int | None = None
    invert_lut: bool | None = None
    copy_metadata_risky: bool | None = None
    round_down_shapes: bool | None = None
    overwrite: bool | None = None
    stop_requested: bool = False
    process: subprocess.Popen[str] | None = field(default=None, repr=False, compare=False)
    lock: threading.Lock = field(default_factory=threading.Lock)


_jobs_lock = threading.Lock()
_jobs: dict[str, JobState] = {}


def register_job(job: JobState) -> None:
    with _jobs_lock:
        _jobs[job.job_id] = job


def unregister_job(job_id: str) -> None:
    with _jobs_lock:
        _jobs.pop(job_id, None)


def terminate_job_process(job: JobState, *, force: bool = False) -> bool:
    with job.lock:
        process = job.process

    if not process or process.poll() is not None:
        return False

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return False
    return True


def _serialize_job(job: JobState) -> dict[str, Any]:
    return {
        "jobId": job.job_id,
        "type": job.type,
        "status": job.status,
        "createdAtMs": job.created_at_ms,
        "finishedAtMs": job.finished_at_ms,
        "processed": job.processed,
        "total": job.total,
        "error": job.error,
        "current": job.current,
        "label": job.label,
        "saveDir": job.save_dir,
        "datasets": job.datasets,
        "roi": job.roi,
        "maskCval": job.mask_cval,
        "addedPadding": job.added_padding,
        "invertLut": job.invert_lut,
        "copyMetadataRisky": job.copy_metadata_risky,
        "roundDownShapes": job.round_down_shapes,
        "overwrite": job.overwrite,
    }


@router.get("/list")
def list_jobs(x_spatialdino_clientid: str | None = Header(None)) -> dict[str, Any]:
    client_id = _require_client_id(x_spatialdino_clientid)
    now_ms = _now_ms()

    with _jobs_lock:
        values = list(_jobs.values())

    jobs = []
    for job in values:
        if job.owner_client_id != client_id:
            continue
        with job.lock:
            jobs.append(_serialize_job(job))

    jobs.sort(key=lambda item: item["createdAtMs"], reverse=True)
    return {"jobs": jobs, "nowMs": now_ms}


@router.post("/cancel")
def cancel_job(payload: CancelJobRequest, x_spatialdino_clientid: str | None = Header(None)) -> dict[str, Any]:
    client_id = _require_client_id(x_spatialdino_clientid)
    with _jobs_lock:
        job = _jobs.get(payload.job_id)
    if not job or job.owner_client_id != client_id:
        raise HTTPException(status_code=404, detail="Unknown job id.")

    with job.lock:
        if job.status != "running":
            return {"ok": True, "status": job.status}
        job.stop_requested = True
        job.current = "Stopping"

    terminate_job_process(job, force=False)
    return {"ok": True, "status": "stopping"}


@router.post("/clear")
def clear_jobs(payload: ClearJobsRequest, x_spatialdino_clientid: str | None = Header(None)) -> dict[str, Any]:
    client_id = _require_client_id(x_spatialdino_clientid)
    removed = 0
    with _jobs_lock:
        for job_id, job in list(_jobs.items()):
            if job.owner_client_id != client_id:
                continue
            with job.lock:
                running = job.status == "running"
            if payload.keep_running and running:
                continue
            if running:
                continue
            _jobs.pop(job_id, None)
            removed += 1
    return {"ok": True, "removed": removed}


@router.post("/remove")
def remove_job(payload: RemoveJobRequest, x_spatialdino_clientid: str | None = Header(None)) -> dict[str, Any]:
    client_id = _require_client_id(x_spatialdino_clientid)
    with _jobs_lock:
        job = _jobs.get(payload.job_id)
    if not job or job.owner_client_id != client_id:
        raise HTTPException(status_code=404, detail="Unknown job id.")

    with job.lock:
        if job.status == "running":
            raise HTTPException(status_code=409, detail="Cannot remove a running job (stop it first).")

    with _jobs_lock:
        _jobs.pop(payload.job_id, None)
    return {"ok": True}
