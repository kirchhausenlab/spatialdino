from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CpuSample:
    total: int
    idle: int


def _read_proc_stat(path: Path = Path("/proc/stat")) -> dict[str, CpuSample]:
    samples: dict[str, CpuSample] = {}
    text = path.read_text(encoding="utf-8", errors="replace")

    for line in text.splitlines():
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        cpu_id = parts[0]
        if cpu_id == "cpu":
            continue
        if len(parts) < 5:
            continue

        try:
            values = [int(value) for value in parts[1:]]
        except ValueError:
            continue

        idle = values[3]
        iowait = values[4] if len(values) > 4 else 0

        total = sum(values)
        idle_total = idle + iowait
        samples[cpu_id] = CpuSample(total=total, idle=idle_total)

    return samples


def _utilization_pct(prev: CpuSample, cur: CpuSample) -> float:
    total_delta = cur.total - prev.total
    idle_delta = cur.idle - prev.idle
    if total_delta <= 0:
        return 0.0
    busy = total_delta - idle_delta
    return max(0.0, min(100.0, busy / total_delta * 100.0))


async def get_cpu_activity(sample_window_s: float = 0.2, active_threshold_pct: float = 10.0) -> dict:
    first = _read_proc_stat()
    await asyncio.sleep(sample_window_s)
    second = _read_proc_stat()

    cpu_ids = sorted(set(first.keys()) & set(second.keys()))
    per_core = []
    for cpu_id in cpu_ids:
        pct = _utilization_pct(first[cpu_id], second[cpu_id])
        per_core.append({"cpu": cpu_id, "utilizationPct": round(pct, 1)})

    total_cores = len(per_core)
    active_cores = sum(1 for core in per_core if core["utilizationPct"] > active_threshold_pct)
    avg_utilization = round(sum(core["utilizationPct"] for core in per_core) / total_cores, 1) if total_cores else 0.0

    return {
        "totalCores": total_cores,
        "activeCores": active_cores,
        "averageUtilizationPct": avg_utilization,
        "sampleWindowMs": int(sample_window_s * 1000),
    }


def get_nvidia_gpu_memory() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=2)
    except FileNotFoundError:
        return {"nvidiaSmiAvailable": False, "gpus": []}
    except subprocess.TimeoutExpired:
        return {"nvidiaSmiAvailable": True, "gpus": [], "error": "nvidia-smi timed out"}

    if result.returncode != 0:
        err = (result.stderr or "").strip() or (result.stdout or "").strip()
        return {"nvidiaSmiAvailable": True, "gpus": [], "error": err or f"nvidia-smi exited {result.returncode}"}

    gpus = []
    for raw_line in (result.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        index_s, name, used_s, total_s = parts[0], parts[1], parts[2], parts[3]
        try:
            index = int(index_s)
            used_mib = int(used_s)
            total_mib = int(total_s)
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "name": name,
                "memoryUsedMiB": used_mib,
                "memoryTotalMiB": total_mib,
            }
        )

    gpus.sort(key=lambda gpu: gpu["index"])
    return {"nvidiaSmiAvailable": True, "gpus": gpus}
