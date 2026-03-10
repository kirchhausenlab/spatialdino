from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from starlette.requests import Request
import tifffile

from spatialdino_server import app as app_module
from spatialdino_server import jobs_api


def make_request(client_host: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "client": (client_host, 12345),
            "server": ("testserver", 8000),
            "scheme": "http",
            "http_version": "1.1",
        }
    )


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        with jobs_api._jobs_lock:
            jobs_api._jobs.clear()

    def tearDown(self) -> None:
        with jobs_api._jobs_lock:
            jobs_api._jobs.clear()

    def test_root_serves_built_frontend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            (dist / "index.html").write_text(
                "<p>__SPATIALDINO_SERVER_HOSTNAME__ __SPATIALDINO_SESSION_LABEL__</p>",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"SPATIALDINO_DIST_DIR": str(dist), "SPATIALDINO_SERVER_HOSTNAME": "render-host"},
                clear=False,
            ):
                with patch("spatialdino_server.app.classify_client_session", return_value="Local"):
                    response = app_module.root(make_request("127.0.0.1"))

            self.assertEqual(response.status_code, 200)  # type: ignore[attr-defined]
            body = response.body.decode("utf-8")
            self.assertIn("render-host", body)
            self.assertIn("Local", body)

    def test_spa_fallback_serves_index_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            (dist / "index.html").write_text("<p>fallback</p>", encoding="utf-8")

            with patch.dict(os.environ, {"SPATIALDINO_DIST_DIR": str(dist)}, clear=False):
                response = app_module.spa_fallback("inference", make_request("127.0.0.1"))

            self.assertEqual(response.status_code, 200)  # type: ignore[attr-defined]
            self.assertIn("fallback", response.body.decode("utf-8"))

    def test_classify_client_session_marks_loopback_as_local(self) -> None:
        with patch("spatialdino_server.app.get_server_ip_addresses", return_value=frozenset({"10.0.0.5"})):
            self.assertEqual(app_module.classify_client_session("127.0.0.1"), "Local")

    def test_classify_client_session_marks_server_ip_as_local(self) -> None:
        with patch("spatialdino_server.app.get_server_ip_addresses", return_value=frozenset({"10.0.0.5"})):
            self.assertEqual(app_module.classify_client_session("10.0.0.5"), "Local")

    def test_classify_client_session_marks_private_ip_as_local_network(self) -> None:
        with patch("spatialdino_server.app.get_server_ip_addresses", return_value=frozenset({"10.0.0.5"})):
            self.assertEqual(app_module.classify_client_session("192.168.1.20"), "Local network")

    def test_classify_client_session_marks_public_ip_as_remote(self) -> None:
        with patch("spatialdino_server.app.get_server_ip_addresses", return_value=frozenset({"10.0.0.5"})):
            self.assertEqual(app_module.classify_client_session("8.8.8.8"), "Remote")

    def test_session_info_returns_hostname_and_label(self) -> None:
        with patch.dict(os.environ, {"SPATIALDINO_SERVER_HOSTNAME": "render-host"}, clear=False):
            with patch("spatialdino_server.app.classify_client_session", return_value="Local network"):
                payload = app_module.session_info(make_request("192.168.1.20"))

        self.assertEqual(payload, {"serverHostname": "render-host", "sessionLabel": "Local network"})

    def test_inference_options_lists_backbone_weights_and_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            models_dir = repo_root / "models"
            models_dir.mkdir()
            (models_dir / "b.pth").write_text("", encoding="utf-8")
            (models_dir / "a.pt").write_text("", encoding="utf-8")
            (models_dir / "ignore.txt").write_text("", encoding="utf-8")

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=repo_root),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={
                        "nvidiaSmiAvailable": True,
                        "gpus": [
                            {"index": 1, "name": "GPU-1", "memoryUsedMiB": 0, "memoryTotalMiB": 1},
                            {"index": 0, "name": "GPU-0", "memoryUsedMiB": 0, "memoryTotalMiB": 1},
                        ],
                    },
                ),
            ):
                payload = app_module.inference_options()

        self.assertEqual(
            payload["backboneWeights"],
            [
                {"label": "a.pt", "value": "models/a.pt"},
                {"label": "b.pth", "value": "models/b.pth"},
            ],
        )
        self.assertEqual(
            payload["gpus"],
            [
                {"index": 1, "name": "GPU-1"},
                {"index": 0, "name": "GPU-0"},
            ],
        )
        self.assertIsNone(payload["gpuError"])
        self.assertEqual(payload["nvidiaSmiAvailable"], True)

    def test_validate_inference_input_folder_accepts_uniform_3d_tiffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            tifffile.imwrite(input_dir / "a.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            tifffile.imwrite(input_dir / "b.tiff", np.ones((2, 3, 4), dtype=np.uint8))

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_inference_input_folder(str(input_dir))

        self.assertEqual(
            payload,
            {
                "valid": True,
                "message": "Valid dataset.",
                "fileCount": 2,
                "shape": {"x": 4, "y": 3, "z": 2},
            },
        )

    def test_validate_inference_input_folder_rejects_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing"

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_inference_input_folder(str(missing))

        self.assertEqual(payload["valid"], False)
        self.assertEqual(payload["reasonCode"], "missing")
        self.assertEqual(payload["message"], "Input folder does not exist.")

    def test_validate_inference_input_folder_rejects_non_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_file = root / "input.txt"
            input_file.write_text("x", encoding="utf-8")

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_inference_input_folder(str(input_file))

        self.assertEqual(payload["valid"], False)
        self.assertEqual(payload["reasonCode"], "not_directory")
        self.assertEqual(payload["message"], "Input path is not a folder.")

    def test_validate_inference_input_folder_rejects_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_inference_input_folder(str(input_dir))

        self.assertEqual(payload["valid"], False)
        self.assertEqual(payload["reasonCode"], "no_tiff_files")
        self.assertEqual(payload["message"], "Input folder contains no .tif or .tiff files.")

    def test_validate_inference_input_folder_rejects_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            tifffile.imwrite(input_dir / "a.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            tifffile.imwrite(input_dir / "b.tif", np.zeros((5, 3, 4), dtype=np.uint8))

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_inference_input_folder(str(input_dir))

        self.assertEqual(payload["valid"], False)
        self.assertEqual(payload["reasonCode"], "shape_mismatch")
        self.assertIn("TIFF files do not all have the same 3D shape.", payload["message"])

    def test_run_inference_requests_overwrite_confirmation_for_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            models_dir = root / "models"
            input_dir.mkdir()
            output_dir.mkdir()
            models_dir.mkdir()
            tifffile.imwrite(input_dir / "stack0001.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            (output_dir / "existing.txt").write_text("x", encoding="utf-8")
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 4, "y_start": 0, "y_end": 3, "z_start": 0, "z_end": 2},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 1},
                overwrite=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                response = app_module.run_inference(payload, "client-1234")

        self.assertEqual(response["submitted"], False)
        self.assertEqual(response["valid"], True)
        self.assertEqual(response["requiresOverwriteConfirmation"], True)
        self.assertEqual(response["outputEntryCount"], 1)
        self.assertEqual(response["outputEntriesPreview"], ["existing.txt"])
        with jobs_api._jobs_lock:
            self.assertEqual(jobs_api._jobs, {})

    def test_run_inference_submits_job_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            models_dir = root / "models"
            input_dir.mkdir()
            output_dir.mkdir()
            models_dir.mkdir()
            tifffile.imwrite(input_dir / "stack0001.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 4, "y_start": 0, "y_end": 3, "z_start": 0, "z_end": 2},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 1},
                overwrite=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
                patch("spatialdino_server.app._launch_inference_job_thread") as launch_thread,
            ):
                response = app_module.run_inference(payload, "client-1234")

        self.assertEqual(response["submitted"], True)
        self.assertIn("jobId", response)
        launch_thread.assert_called_once()
        launch_config = launch_thread.call_args.args[1]
        self.assertEqual(launch_config["selected_stems"], ["stack0001"])
        with jobs_api._jobs_lock:
            self.assertEqual(len(jobs_api._jobs), 1)
            job = next(iter(jobs_api._jobs.values()))
            self.assertEqual(job.type, "inference")
            self.assertEqual(job.total, 1)
            self.assertEqual(job.datasets, [{"source_dir": str(input_dir), "save_to": "output"}])

    def test_inference_command_preview_returns_shell_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            models_dir = root / "models"
            input_dir.mkdir()
            output_dir.mkdir()
            models_dir.mkdir()
            tifffile.imwrite(input_dir / "stack0001.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[1, 0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 4, "y_start": 0, "y_end": 3, "z_start": 0, "z_end": 2},
                anisotropy={"x": 1.0, "y": 2.0, "z": 3.0},
                file_range={"start": 0, "end": 1},
                overwrite=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={
                        "nvidiaSmiAvailable": True,
                        "gpus": [{"index": 0, "name": "GPU-0"}, {"index": 1, "name": "GPU-1"}],
                    },
                ),
            ):
                response = app_module.inference_command_preview(payload)

        self.assertEqual(response["valid"], True)
        self.assertEqual(response["workingDirectory"], str(root))
        self.assertEqual(response["requiresOverwriteConfirmation"], False)
        self.assertIn(f"cd {root}", response["command"])
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", response["command"])
        self.assertIn("PYTHONUNBUFFERED=1", response["command"])
        self.assertIn(app_module.sys.executable, response["command"])
        self.assertIn("--nproc_per_node=2", response["command"])
        self.assertIn(f"file_path={input_dir}", response["command"])
        self.assertIn(f"save_path={output_dir}", response["command"])
        self.assertIn("isotropic_scale_factor=[3.0,2.0,1.0]", response["command"])
        self.assertIn("inference_route=streaming", response["command"])
        self.assertIn("dtype=bf16", response["command"])

    def test_inference_command_preview_includes_manual_global_norm_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            models_dir = root / "models"
            input_dir.mkdir()
            output_dir.mkdir()
            models_dir.mkdir()
            tifffile.imwrite(input_dir / "stack0001.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 4, "y_start": 0, "y_end": 3, "z_start": 0, "z_end": 2},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 1},
                normalization_mode="global_manual",
                global_hist_min=12.5,
                global_hist_max=98.5,
                overwrite=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                response = app_module.inference_command_preview(payload)

        self.assertEqual(response["valid"], True)
        self.assertIn("global_hist_min=12.5", response["command"])
        self.assertIn("global_hist_max=98.5", response["command"])

    def test_inference_command_preview_for_auto_global_norm_shows_two_stage_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            models_dir = root / "models"
            input_dir.mkdir()
            output_dir.mkdir()
            models_dir.mkdir()
            tifffile.imwrite(input_dir / "stack0001.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 4, "y_start": 0, "y_end": 3, "z_start": 0, "z_end": 2},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 1},
                normalization_mode="global_auto",
                overwrite=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                response = app_module.inference_command_preview(payload)

        self.assertEqual(response["valid"], True)
        self.assertIn("scripts/inference/norm_per_vol.py", response["command"])
        self.assertIn("global_hist_min=<computed-from-norm_per_vol>", response["command"])
        self.assertIn("global_hist_max=<computed-from-norm_per_vol>", response["command"])

    def test_inference_command_preview_warns_when_output_is_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            models_dir = root / "models"
            input_dir.mkdir()
            output_dir.mkdir()
            models_dir.mkdir()
            tifffile.imwrite(input_dir / "stack0001.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            (output_dir / "existing.txt").write_text("x", encoding="utf-8")
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 4, "y_start": 0, "y_end": 3, "z_start": 0, "z_end": 2},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 1},
                overwrite=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                response = app_module.inference_command_preview(payload)

        self.assertEqual(response["valid"], True)
        self.assertEqual(response["requiresOverwriteConfirmation"], True)
        self.assertEqual(
            response["overwriteMessage"],
            "Output folder is not empty. Confirm overwrite to erase its contents and continue.",
        )
        self.assertIn("CUDA_VISIBLE_DEVICES=0", response["command"])
        self.assertIn("PYTHONUNBUFFERED=1", response["command"])

    def test_run_inference_rejects_manual_global_norm_without_both_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            models_dir = root / "models"
            input_dir.mkdir()
            output_dir.mkdir()
            models_dir.mkdir()
            tifffile.imwrite(input_dir / "stack0001.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 4, "y_start": 0, "y_end": 3, "z_start": 0, "z_end": 2},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 1},
                normalization_mode="global_manual",
                global_hist_min=12.5,
                overwrite=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                response = app_module.run_inference(payload, "client-1234")

        self.assertEqual(response["submitted"], False)
        self.assertEqual(response["valid"], False)
        self.assertEqual(response["reasonCode"], "missing_global_hist_values")

    def test_build_inference_command_env_preserves_higher_omp_setting(self) -> None:
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "16"}, clear=False):
            env = app_module._build_inference_command_env({"gpu_indices": [2, 0]})

        self.assertEqual(
            env,
            {
                "CUDA_VISIBLE_DEVICES": "2,0",
                "OMP_NUM_THREADS": "16",
                "PYTHONUNBUFFERED": "1",
            },
        )

    def test_run_inference_rejects_empty_file_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            models_dir = root / "models"
            input_dir.mkdir()
            output_dir.mkdir()
            models_dir.mkdir()
            tifffile.imwrite(input_dir / "stack0001.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 4, "y_start": 0, "y_end": 3, "z_start": 0, "z_end": 2},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 1, "end": 1},
                overwrite=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                response = app_module.run_inference(payload, "client-1234")

        self.assertEqual(response["submitted"], False)
        self.assertEqual(response["valid"], False)
        self.assertEqual(response["reasonCode"], "empty_file_selection")

    def test_progress_updates_only_after_features_are_saved(self) -> None:
        job = jobs_api.JobState(
            job_id="job-1",
            owner_client_id="client-1234",
            type="inference",
            status="running",
            total=1,
        )
        expected_output_dirs = {app_module._canonicalize_runtime_path("/tmp/out/stack0001")}
        expected_feature_paths = {app_module._canonicalize_runtime_path("/tmp/out/stack0001/lr_feats.npy")}
        saved_feature_paths: set[str] = set()

        app_module._update_job_progress_from_output(
            job,
            "Saving to /tmp/out/stack0001",
            expected_output_dirs,
            expected_feature_paths,
            saved_feature_paths,
        )
        with job.lock:
            self.assertEqual(job.processed, 0)
            self.assertEqual(job.current, "Processing stack0001")

        app_module._update_job_progress_from_output(
            job,
            "Saved features to /tmp/out/stack0001/lr_feats.npy",
            expected_output_dirs,
            expected_feature_paths,
            saved_feature_paths,
        )
        with job.lock:
            self.assertEqual(job.processed, 1)
            self.assertEqual(job.current, "stack0001")

    def test_progress_ignores_unexpected_output_paths(self) -> None:
        job = jobs_api.JobState(
            job_id="job-1",
            owner_client_id="client-1234",
            type="inference",
            status="running",
            total=1,
        )
        expected_output_dirs = {app_module._canonicalize_runtime_path("/tmp/out/stack0001")}
        expected_feature_paths = {app_module._canonicalize_runtime_path("/tmp/out/stack0001/lr_feats.npy")}
        saved_feature_paths: set[str] = set()

        app_module._update_job_progress_from_output(
            job,
            "Saving to /tmp/out/stack0002",
            expected_output_dirs,
            expected_feature_paths,
            saved_feature_paths,
        )
        app_module._update_job_progress_from_output(
            job,
            "Saved features to /tmp/out/stack0002/lr_feats.npy",
            expected_output_dirs,
            expected_feature_paths,
            saved_feature_paths,
        )

        with job.lock:
            self.assertEqual(job.processed, 0)
            self.assertIsNone(job.current)

    def test_existing_expected_feature_paths_ignore_unselected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            (output_dir / "stack0001").mkdir(parents=True)
            (output_dir / "stack0002").mkdir(parents=True)
            (output_dir / "stack0002" / "lr_feats.npy").write_bytes(b"")

            expected_paths = app_module._expected_feature_paths(output_dir, ["stack0001"])
            existing_paths = app_module._existing_expected_feature_paths(expected_paths)

        self.assertEqual(existing_paths, [])

    def test_run_inference_job_fails_when_expected_features_are_missing(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = iter(())
                self.pid = 123

            def wait(self) -> int:
                return 0

            def poll(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            (output_dir / "other").mkdir(parents=True)
            (output_dir / "other" / "lr_feats.npy").write_bytes(b"")
            job = jobs_api.JobState(
                job_id="job-1",
                owner_client_id="client-1234",
                type="inference",
                status="running",
                total=1,
            )
            launch_config = {
                "output_path": output_dir,
                "selected_stems": ["stack0001"],
                "overwrite": False,
                "gpu_indices": [0],
            }

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch("spatialdino_server.app._build_inference_command", return_value=["python"]),
                patch("spatialdino_server.app.subprocess.Popen", return_value=FakeProcess()),
            ):
                app_module._run_inference_job(job, launch_config)

            with job.lock:
                self.assertEqual(job.status, "failed")
                self.assertEqual(job.current, "Failed")
                self.assertEqual(job.exit_code, 0)
                self.assertIn("0/1 expected lr_feats.npy files", job.error)
                self.assertIn("stack0001", job.error)
                self.assertTrue(job.log_available)
                self.assertIsNotNone(job.command)
                self.assertIsNotNone(job.log_path)

            self.assertTrue(Path(job.log_path).is_file())
            log_text = Path(job.log_path).read_text(encoding="utf-8")
            self.assertIn("[server] Command:", log_text)
            self.assertIn("0/1 expected lr_feats.npy files", log_text)

    def test_run_inference_job_auto_global_norm_runs_prepass_then_inference(self) -> None:
        class FakeProcess:
            def __init__(self, stdout_lines: tuple[str, ...], return_code: int, on_wait=None) -> None:
                self.stdout = iter(stdout_lines)
                self.pid = 789
                self._return_code = return_code
                self._on_wait = on_wait

            def wait(self) -> int:
                if self._on_wait is not None:
                    self._on_wait()
                return self._return_code

            def poll(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            tifffile.imwrite(input_dir / "stack0001.tif", np.zeros((2, 3, 4), dtype=np.uint8))

            job = jobs_api.JobState(
                job_id="job-auto-global",
                owner_client_id="client-1234",
                type="inference",
                status="running",
                total=1,
            )
            launch_config = {
                "input_path": input_dir,
                "output_path": output_dir,
                "backbone_path": root / "models" / "backbone.pth",
                "selected_stems": ["stack0001"],
                "selected_file_count": 1,
                "overwrite": False,
                "gpu_indices": [0],
                "file_start": 0,
                "file_end": 1,
                "crop_params": (0, 2, 0, 3, 0, 4),
                "effective_crop_params": (0, 2, 0, 3, 0, 4),
                "upsample_factor": 1.0,
                "anisotropy_xyz": (1.0, 1.0, 1.0),
                "inference_route": "streaming",
                "dtype": "bf16",
                "normalization_mode": "global_auto",
                "global_hist_min": None,
                "global_hist_max": None,
            }
            popen_calls: list[list[str]] = []

            def write_norm_stats() -> None:
                (output_dir / "norm_per_vol.txt").write_text(
                    "Global hist min: 12.0\nGlobal hist max: 34.0",
                    encoding="utf-8",
                )

            def write_inference_output() -> None:
                feats_path = output_dir / "stack0001" / "lr_feats.npy"
                feats_path.parent.mkdir(parents=True, exist_ok=True)
                feats_path.write_bytes(b"")

            def popen_side_effect(command, **kwargs):
                popen_calls.append(list(command))
                if len(popen_calls) == 1:
                    return FakeProcess(
                        ("running norm_per_vol for 1 files\n", "saved norm_per_vol to output/norm_per_vol.txt\n"),
                        0,
                        on_wait=write_norm_stats,
                    )
                return FakeProcess(
                    (
                        f"Saving to {output_dir / 'stack0001'}\n",
                        f"Saved features to {output_dir / 'stack0001' / 'lr_feats.npy'}\n",
                    ),
                    0,
                    on_wait=write_inference_output,
                )

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch("spatialdino_server.app.subprocess.Popen", side_effect=popen_side_effect),
            ):
                app_module._run_inference_job(job, launch_config)

            self.assertEqual(len(popen_calls), 2)
            self.assertIn("scripts/inference/norm_per_vol.py", popen_calls[0])
            self.assertIn("global_hist_min=12.0", popen_calls[1])
            self.assertIn("global_hist_max=34.0", popen_calls[1])

            with job.lock:
                self.assertEqual(job.status, "completed")
                self.assertEqual(job.current, "Done")
                self.assertEqual(job.processed, 1)
                self.assertIsNotNone(job.command)
                self.assertIn("global_hist_min=12.0", job.command)
                self.assertIn("global_hist_max=34.0", job.command)

            log_text = Path(job.log_path).read_text(encoding="utf-8")
            self.assertIn("Global normalization prepass command", log_text)
            self.assertIn("Parsed global normalization stats: global_hist_min=12.0, global_hist_max=34.0", log_text)
            self.assertIn("[server] Inference completed successfully.", log_text)

    def test_run_inference_job_persists_process_output_tail(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = iter(
                    (
                        "Traceback (most recent call last):\n",
                        '  File "scripts/inference/inference.py", line 10, in <module>\n',
                        "RuntimeError: boom\n",
                    )
                )
                self.pid = 456

            def wait(self) -> int:
                return 1

            def poll(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            job = jobs_api.JobState(
                job_id="job-2",
                owner_client_id="client-1234",
                type="inference",
                status="running",
                total=1,
            )
            launch_config = {
                "output_path": output_dir,
                "selected_stems": ["stack0001"],
                "overwrite": False,
                "gpu_indices": [0],
                "input_path": root / "input",
                "backbone_path": root / "models" / "backbone.pth",
                "file_start": 0,
                "file_end": 1,
                "crop_params": (0, 1, 0, 1, 0, 1),
                "upsample_factor": 1.0,
                "anisotropy_xyz": (1.0, 1.0, 1.0),
                "inference_route": "streaming",
                "dtype": "bf16",
            }

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch("spatialdino_server.app.subprocess.Popen", return_value=FakeProcess()),
            ):
                app_module._run_inference_job(job, launch_config)

            with job.lock:
                self.assertEqual(job.status, "failed")
                self.assertEqual(job.current, "Failed")
                self.assertEqual(job.exit_code, 1)
                self.assertEqual(job.error, "RuntimeError: boom")
                self.assertTrue(job.log_available)
                self.assertIn(
                    '  File "scripts/inference/inference.py", line 10, in <module>',
                    list(job.log_tail),
                )
                self.assertIn("RuntimeError: boom", list(job.log_tail))
                self.assertIsNotNone(job.log_path)

            log_text = Path(job.log_path).read_text(encoding="utf-8")
            self.assertIn("Traceback (most recent call last):", log_text)
            self.assertIn('  File "scripts/inference/inference.py", line 10, in <module>', log_text)
            self.assertIn("RuntimeError: boom", log_text)
            self.assertIn("[server] Process exited with code 1.", log_text)
