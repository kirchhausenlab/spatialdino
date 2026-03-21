from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from spatialdino_server import status


class ReadProcStatTests(unittest.TestCase):
    def test_read_proc_stat_parses_cpu_rows(self) -> None:
        text = """\
cpu  1 2 3 4 5 6 7 8
cpu0 10 1 2 3 4 5 6 7
cpu1 20 2 3 4 5 6 7 8
intr 123
cpuX bad values here
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stat"
            path.write_text(text, encoding="utf-8")
            parsed = status._read_proc_stat(path)

        self.assertEqual(set(parsed.keys()), {"cpu0", "cpu1"})
        self.assertEqual(parsed["cpu0"].total, 38)
        self.assertEqual(parsed["cpu0"].idle, 7)
        self.assertEqual(parsed["cpu1"].total, 55)
        self.assertEqual(parsed["cpu1"].idle, 9)

    def test_utilization_pct_clamps_to_bounds(self) -> None:
        self.assertEqual(
            status._utilization_pct(status.CpuSample(10, 1), status.CpuSample(10, 1)),
            0.0,
        )
        self.assertEqual(
            status._utilization_pct(status.CpuSample(10, 1), status.CpuSample(20, 1)),
            100.0,
        )
        self.assertEqual(
            status._utilization_pct(status.CpuSample(10, 1), status.CpuSample(20, 11)),
            0.0,
        )


class CpuActivityTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_cpu_activity_aggregates_active_cores(self) -> None:
        first = {
            "cpu0": status.CpuSample(total=100, idle=20),
            "cpu1": status.CpuSample(total=100, idle=90),
        }
        second = {
            "cpu0": status.CpuSample(total=200, idle=40),
            "cpu1": status.CpuSample(total=200, idle=195),
        }
        with (
            patch(
                "spatialdino_server.status._read_proc_stat", side_effect=[first, second]
            ),
            patch(
                "spatialdino_server.status.asyncio.sleep", new=AsyncMock()
            ) as sleep_mock,
        ):
            payload = await status.get_cpu_activity(
                sample_window_s=0.5, active_threshold_pct=10.0
            )

        sleep_mock.assert_awaited_once_with(0.5)
        self.assertEqual(payload["totalCores"], 2)
        self.assertEqual(payload["activeCores"], 1)
        self.assertEqual(payload["averageUtilizationPct"], 40.0)
        self.assertEqual(payload["sampleWindowMs"], 500)


class NvidiaGpuMemoryTests(unittest.TestCase):
    def test_handles_missing_nvidia_smi(self) -> None:
        with patch(
            "spatialdino_server.status.subprocess.run", side_effect=FileNotFoundError
        ):
            payload = status.get_nvidia_gpu_memory()
        self.assertEqual(payload, {"nvidiaSmiAvailable": False, "gpus": []})

    def test_handles_timeout(self) -> None:
        with patch(
            "spatialdino_server.status.subprocess.run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", timeout=2),
        ):
            payload = status.get_nvidia_gpu_memory()
        self.assertEqual(payload["nvidiaSmiAvailable"], True)
        self.assertEqual(payload["gpus"], [])
        self.assertEqual(payload["error"], "nvidia-smi timed out")

    def test_handles_nonzero_exit(self) -> None:
        completed = SimpleNamespace(returncode=7, stdout="", stderr="boom")
        with patch("spatialdino_server.status.subprocess.run", return_value=completed):
            payload = status.get_nvidia_gpu_memory()
        self.assertEqual(payload["nvidiaSmiAvailable"], True)
        self.assertEqual(payload["gpus"], [])
        self.assertEqual(payload["error"], "boom")

    def test_parses_and_sorts_gpu_rows(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stderr="",
            stdout="1, GPU-B, 20, 100\n0, GPU-A, 10, 80\ninvalid,row\n",
        )
        with patch("spatialdino_server.status.subprocess.run", return_value=completed):
            payload = status.get_nvidia_gpu_memory()

        self.assertEqual(payload["nvidiaSmiAvailable"], True)
        self.assertEqual(
            payload["gpus"],
            [
                {
                    "index": 0,
                    "name": "GPU-A",
                    "memoryUsedMiB": 10,
                    "memoryTotalMiB": 80,
                },
                {
                    "index": 1,
                    "name": "GPU-B",
                    "memoryUsedMiB": 20,
                    "memoryTotalMiB": 100,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
