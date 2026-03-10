from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from spatialdino_server import jobs_api


def make_job(**overrides: object) -> jobs_api.JobState:
    job = jobs_api.JobState(
        job_id="job-1",
        owner_client_id="client-1234",
        type="roi-save",
        status="completed",
        created_at_ms=100,
        finished_at_ms=200,
        processed=12,
        total=12,
        current="Done",
        label="ROI Save",
        save_dir="/tmp/output",
        datasets=[{"source_dir": "/tmp/input", "save_to": "roi"}],
        roi={"x0": 0, "x1": 1, "y0": 2, "y1": 3, "z0": 4, "z1": 5},
        added_padding=0,
        invert_lut=True,
        copy_metadata_risky=False,
        overwrite=False,
    )
    for key, value in overrides.items():
        setattr(job, key, value)
    return job


class JobsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        with jobs_api._jobs_lock:
            jobs_api._jobs.clear()

    def tearDown(self) -> None:
        with jobs_api._jobs_lock:
            jobs_api._jobs.clear()

    def test_list_jobs_filters_by_client_and_sorts_newest_first(self) -> None:
        with jobs_api._jobs_lock:
            jobs_api._jobs["old"] = make_job(job_id="old", created_at_ms=100)
            jobs_api._jobs["new"] = make_job(job_id="new", created_at_ms=300, label="Newest")
            jobs_api._jobs["other"] = make_job(job_id="other", owner_client_id="other-9999", created_at_ms=500)

        payload = jobs_api.list_jobs("client-1234")

        self.assertEqual([job["jobId"] for job in payload["jobs"]], ["new", "old"])
        self.assertEqual(payload["jobs"][0]["label"], "Newest")
        self.assertIn("nowMs", payload)

    def test_clear_jobs_keeps_running_entries_when_requested(self) -> None:
        with jobs_api._jobs_lock:
            jobs_api._jobs["done"] = make_job(job_id="done", status="completed")
            jobs_api._jobs["running"] = make_job(job_id="running", status="running", finished_at_ms=None)

        result = jobs_api.clear_jobs(jobs_api.ClearJobsRequest(keep_running=True), "client-1234")

        self.assertEqual(result, {"ok": True, "removed": 1})
        with jobs_api._jobs_lock:
            self.assertEqual(set(jobs_api._jobs.keys()), {"running"})

    def test_cancel_requests_stop_and_remove_deletes_job_after_halt(self) -> None:
        with jobs_api._jobs_lock:
            jobs_api._jobs["job-1"] = make_job(job_id="job-1", status="running", finished_at_ms=None, current="Busy")

        with patch("spatialdino_server.jobs_api.terminate_job_process") as terminate_process:
            cancel_result = jobs_api.cancel_job(jobs_api.CancelJobRequest(job_id="job-1"), "client-1234")

        self.assertEqual(cancel_result, {"ok": True, "status": "stopping"})
        terminate_process.assert_called_once()
        with jobs_api._jobs_lock:
            job = jobs_api._jobs["job-1"]
            self.assertEqual(job.status, "running")
            self.assertEqual(job.current, "Stopping")
            self.assertEqual(job.finished_at_ms, None)
            job.status = "halted"
            job.current = "Stopped"
            job.finished_at_ms = 1234

        remove_result = jobs_api.remove_job(jobs_api.RemoveJobRequest(job_id="job-1"), "client-1234")

        self.assertEqual(remove_result, {"ok": True})
        with jobs_api._jobs_lock:
            self.assertEqual(jobs_api._jobs, {})

    def test_remove_unknown_job_raises_404(self) -> None:
        with self.assertRaises(HTTPException) as context:
            jobs_api.remove_job(jobs_api.RemoveJobRequest(job_id="missing"), "client-1234")

        self.assertEqual(context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
