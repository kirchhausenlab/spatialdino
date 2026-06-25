from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

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


class FakeUrlopenResponse(io.BytesIO):
    def __enter__(self) -> FakeUrlopenResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def make_zip_bytes(files: dict[str, bytes]) -> bytes:
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w") as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return handle.getvalue()


def write_inference_output_timepoint(
    root: Path,
    name: str,
    *,
    lr_shape: tuple[int, ...] = (2, 2, 2, 390),
    raw_shape: tuple[int, ...] = (4, 4, 4),
    raw_dtype: object = np.uint8,
) -> None:
    lr_dir = root / "lr_feats"
    raw_dir = root / "raw"
    lr_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(lr_dir / f"{name}.npy", np.zeros(lr_shape, dtype=np.float32))
    tifffile.imwrite(raw_dir / f"{name}.tif", np.zeros(raw_shape, dtype=raw_dtype), photometric="minisblack")


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

    def test_download_inference_backbone_creates_file_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            target_path = repo_root / "models" / "backbone.pth"

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=repo_root),
                patch(
                    "spatialdino_server.app.urllib.request.urlopen",
                    return_value=FakeUrlopenResponse(b"downloaded-weights"),
                ) as urlopen,
            ):
                response = app_module.download_inference_backbone(app_module.DownloadBackboneWeightsRequest())

            self.assertEqual(response["downloaded"], True)
            self.assertEqual(response["backboneWeight"], "models/backbone.pth")
            self.assertEqual(response["targetPath"], str(target_path))
            self.assertEqual(response["alreadyExisted"], False)
            self.assertEqual(target_path.read_bytes(), b"downloaded-weights")
            urlopen.assert_called_once_with(app_module.DEFAULT_INFERENCE_BACKBONE_URL, timeout=60)

    def test_download_inference_backbone_requests_overwrite_when_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            models_dir = repo_root / "models"
            models_dir.mkdir()
            target_path = models_dir / "backbone.pth"
            target_path.write_bytes(b"existing-weights")

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=repo_root),
                patch("spatialdino_server.app.urllib.request.urlopen") as urlopen,
            ):
                response = app_module.download_inference_backbone(app_module.DownloadBackboneWeightsRequest())

            self.assertEqual(response["downloaded"], False)
            self.assertEqual(response["requiresOverwriteConfirmation"], True)
            self.assertEqual(response["targetPath"], str(target_path))
            self.assertEqual(response["backboneWeight"], "models/backbone.pth")
            self.assertEqual(target_path.read_bytes(), b"existing-weights")
            urlopen.assert_not_called()

    def test_download_inference_backbone_overwrites_existing_file_when_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            models_dir = repo_root / "models"
            models_dir.mkdir()
            target_path = models_dir / "backbone.pth"
            target_path.write_bytes(b"existing-weights")

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=repo_root),
                patch(
                    "spatialdino_server.app.urllib.request.urlopen",
                    return_value=FakeUrlopenResponse(b"replacement-weights"),
                ),
            ):
                response = app_module.download_inference_backbone(
                    app_module.DownloadBackboneWeightsRequest(overwrite=True)
                )

            self.assertEqual(response["downloaded"], True)
            self.assertEqual(response["alreadyExisted"], True)
            self.assertEqual(target_path.read_bytes(), b"replacement-weights")

    def test_data_options_lists_manifest_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            manifest = {
                "version": 1,
                "datasets": [
                    {"name": "ap2", "archiveUrl": "https://example.com/ap2.zip"},
                    {"name": "dextran", "archiveUrl": "https://example.com/dextran.zip"},
                ],
            }

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=repo_root),
                patch(
                    "spatialdino_server.app.urllib.request.urlopen",
                    return_value=FakeUrlopenResponse(json.dumps(manifest).encode("utf-8")),
                ),
            ):
                payload = app_module.data_options()

        self.assertEqual(payload["manifestUrl"], app_module.DEFAULT_PUBLIC_DATA_MANIFEST_URL)
        self.assertEqual(payload["downloadRoot"], str(repo_root / "data" / "raw_data"))
        self.assertEqual(payload["datasets"], [{"name": "ap2"}, {"name": "dextran"}])

    def test_run_data_download_requests_overwrite_confirmation_for_existing_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            existing_path = repo_root / "data" / "raw_data" / "ap2"
            existing_path.mkdir(parents=True)

            payload = app_module.RunDataDownloadRequest(datasets=["ap2", "dextran"])

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=repo_root),
                patch(
                    "spatialdino_server.app._load_public_data_manifest",
                    return_value=[
                        {"name": "ap2", "archiveUrl": "https://example.com/ap2.zip"},
                        {"name": "dextran", "archiveUrl": "https://example.com/dextran.zip"},
                    ],
                ),
            ):
                response = app_module.run_data_download(payload, "client-1234")

        self.assertEqual(response["submitted"], False)
        self.assertEqual(response["valid"], True)
        self.assertEqual(response["requiresOverwriteConfirmation"], True)
        self.assertEqual(response["existingDatasetCount"], 1)
        self.assertEqual(response["existingDatasetNames"], ["ap2"])
        self.assertEqual(response["existingDatasetPaths"], [str(existing_path)])
        with jobs_api._jobs_lock:
            self.assertEqual(jobs_api._jobs, {})

    def test_run_data_download_submits_job_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            payload = app_module.RunDataDownloadRequest(datasets=["ap2", "dextran"])

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=repo_root),
                patch(
                    "spatialdino_server.app._load_public_data_manifest",
                    return_value=[
                        {"name": "ap2", "archiveUrl": "https://example.com/ap2.zip"},
                        {"name": "dextran", "archiveUrl": "https://example.com/dextran.zip"},
                    ],
                ),
                patch("spatialdino_server.app._launch_data_download_job_thread") as launch_thread,
            ):
                response = app_module.run_data_download(payload, "client-1234")

        self.assertEqual(response["submitted"], True)
        self.assertIn("jobId", response)
        launch_thread.assert_called_once()
        launch_config = launch_thread.call_args.args[1]
        self.assertEqual(launch_config["selected_names"], ["ap2", "dextran"])
        self.assertEqual(launch_config["download_root"], repo_root / "data" / "raw_data")
        with jobs_api._jobs_lock:
            self.assertEqual(len(jobs_api._jobs), 1)
            job = next(iter(jobs_api._jobs.values()))
            self.assertEqual(job.type, "data-download")
            self.assertEqual(job.total, 2)
            self.assertEqual(
                job.datasets,
                [
                    {"source_dir": "https://example.com/ap2.zip", "save_to": "ap2"},
                    {"source_dir": "https://example.com/dextran.zip", "save_to": "dextran"},
                ],
            )

    def test_run_data_download_job_downloads_and_extracts_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            download_root = repo_root / "data" / "raw_data"
            archive_bytes = {
                "https://example.com/ap2.zip": make_zip_bytes(
                    {
                        "ap2/image_a.tif": b"ap2-image",
                        "ap2/notes/readme.txt": b"ap2-readme",
                    }
                ),
                "https://example.com/dextran.zip": make_zip_bytes(
                    {
                        "volume_001.tif": b"dextran-image",
                        "metadata/info.txt": b"dextran-info",
                    }
                ),
            }
            job = jobs_api.JobState(
                job_id="job-data-1",
                owner_client_id="client-1234",
                type="data-download",
                status="running",
                total=2,
            )
            launch_config = {
                "manifest_url": "https://example.com/manifest.json",
                "download_root": download_root,
                "selected_datasets": [
                    {"name": "ap2", "archiveUrl": "https://example.com/ap2.zip"},
                    {"name": "dextran", "archiveUrl": "https://example.com/dextran.zip"},
                ],
                "selected_names": ["ap2", "dextran"],
                "skipped_names": [],
                "overwrite_existing": False,
            }

            def fake_download(url: str, target_path: Path) -> None:
                target_path.write_bytes(archive_bytes[url])

            with (
                patch("spatialdino_server.app.get_repo_root", return_value=repo_root),
                patch("spatialdino_server.app._download_url_to_file", side_effect=fake_download),
            ):
                app_module._run_data_download_job(job, launch_config)

            self.assertEqual((download_root / "ap2" / "image_a.tif").read_bytes(), b"ap2-image")
            self.assertEqual((download_root / "ap2" / "notes" / "readme.txt").read_bytes(), b"ap2-readme")
            self.assertEqual((download_root / "dextran" / "volume_001.tif").read_bytes(), b"dextran-image")
            self.assertEqual((download_root / "dextran" / "metadata" / "info.txt").read_bytes(), b"dextran-info")

            with job.lock:
                self.assertEqual(job.status, "completed")
                self.assertEqual(job.current, "Done")
                self.assertEqual(job.processed, 2)
                self.assertTrue(job.log_available)
                self.assertIsNotNone(job.command)
                self.assertIsNotNone(job.log_path)

            log_text = Path(job.log_path).read_text(encoding="utf-8")
            self.assertIn("Downloading ap2", log_text)
            self.assertIn("Saved dataset dextran", log_text)
            self.assertIn("Public data download completed successfully.", log_text)

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

    def test_validate_inference_input_folder_ignores_hidden_tiffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            tifffile.imwrite(input_dir / "a.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            tifffile.imwrite(input_dir / "b.tiff", np.ones((2, 3, 4), dtype=np.uint8))
            tifffile.imwrite(input_dir / ".hidden_bad.tif", np.ones((5, 6, 7), dtype=np.uint8))

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_inference_input_folder(str(input_dir))

        self.assertEqual(payload["valid"], True)
        self.assertEqual(payload["fileCount"], 2)
        self.assertEqual(payload["shape"], {"x": 4, "y": 3, "z": 2})

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

    def test_validate_process_features_input_folder_accepts_valid_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, lr_shape=(2, 2), raw_shape=(2, 3, 4))

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_process_features_input_folder(str(input_dir))

        self.assertEqual(
            payload,
            {
                "valid": True,
                "message": "Valid feature folder. Found 2 timepoints.",
                "subfolderCount": 2,
                "subfolderNames": ["sample_a", "sample_b"],
            },
        )

    def test_validate_process_features_input_folder_rejects_missing_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_process_features_input_folder(str(input_dir))

        self.assertEqual(payload["valid"], False)
        self.assertEqual(payload["reasonCode"], "missing_required_files")
        self.assertEqual(payload["message"], "Input folder must contain matching lr_feats/*.npy and raw/*.tif files.")

    def test_validate_process_features_input_folder_ignores_hidden_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, lr_shape=(2, 2), raw_shape=(2, 3, 4))
            hidden_dir = input_dir / ".hidden_sample"
            hidden_dir.mkdir()

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_process_features_input_folder(str(input_dir))

        self.assertEqual(payload["valid"], True)
        self.assertEqual(payload["subfolderCount"], 2)
        self.assertEqual(payload["subfolderNames"], ["sample_a", "sample_b"])

    def test_validate_process_features_input_folder_rejects_missing_lr_feats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            raw_dir = input_dir / "raw"
            raw_dir.mkdir()
            tifffile.imwrite(raw_dir / "sample_a.tif", np.zeros((2, 3, 4), dtype=np.uint8))

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_process_features_input_folder(str(input_dir))

        self.assertEqual(payload["valid"], False)
        self.assertEqual(payload["reasonCode"], "missing_required_files")
        self.assertEqual(payload["message"], "Missing feature file for sample_a: sample_a.npy.")

    def test_validate_process_features_input_folder_rejects_missing_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            lr_dir = input_dir / "lr_feats"
            lr_dir.mkdir()
            np.save(lr_dir / "sample_a.npy", np.zeros((2, 2), dtype=np.float32))

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_process_features_input_folder(str(input_dir))

        self.assertEqual(payload["valid"], False)
        self.assertEqual(payload["reasonCode"], "missing_required_files")
        self.assertEqual(payload["message"], "Missing raw file for sample_a: sample_a.tif.")

    def test_build_process_features_launch_config_accepts_valid_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "processed-output"
            input_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a")

            payload = app_module.RunProcessFeaturesRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                save_high_resolution_features=True,
                high_resolution_save_format=".tif",
                save_pca=True,
                pca_components=3,
                pca_save_format=".tif",
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                validation, launch_config = app_module._build_process_features_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertEqual(validation["subfolderCount"], 1)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["gpu_index"], 0)
        self.assertEqual(launch_config["subfolder_count"], 1)
        self.assertEqual(launch_config["high_resolution_save_format"], ".tif")
        self.assertEqual(launch_config["pca_save_format"], ".tif")
        self.assertEqual(launch_config["global_pca"], True)
        self.assertEqual(launch_config["output_path"], output_dir)

        command = app_module._build_process_features_command(launch_config)
        self.assertIn("scripts/post_processing/process_features.py", command)
        self.assertIn("--output-path", command)
        self.assertIn(str(output_dir), command)
        self.assertIn("--global-pca", command)
        self.assertEqual(command[command.index("--global-pca") + 1], "true")

    def test_build_process_features_launch_config_accepts_per_timepoint_pca(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "processed-output"
            input_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a")

            payload = app_module.RunProcessFeaturesRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                save_pca=True,
                pca_components=3,
                pca_save_format=".tif",
                global_pca=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                validation, launch_config = app_module._build_process_features_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["global_pca"], False)

        command = app_module._build_process_features_command(launch_config)
        self.assertIn("--global-pca", command)
        self.assertEqual(command[command.index("--global-pca") + 1], "false")

    def test_build_process_features_launch_config_accepts_file_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "processed-output"
            input_dir.mkdir()
            for name in ("sample_a", "sample_b", "sample_c"):
                write_inference_output_timepoint(input_dir, name)

            payload = app_module.RunProcessFeaturesRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                file_range=app_module.InferenceFileRangeRequest(start=1, end=2),
                save_high_resolution_features=True,
                high_resolution_save_format=".tif",
                save_pca=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                validation, launch_config = app_module._build_process_features_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertEqual(validation["subfolderCount"], 2)
        self.assertEqual(validation["selectedFileCount"], 2)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["subfolder_count"], 2)
        self.assertEqual(launch_config["input_subfolder_count"], 3)
        self.assertEqual(launch_config["file_start"], 1)
        self.assertEqual(launch_config["file_end"], 3)
        self.assertEqual(launch_config["selected_timepoint_names"], ["sample_b", "sample_c"])

        command = app_module._build_process_features_command(launch_config)
        self.assertIn("--file-start", command)
        self.assertEqual(command[command.index("--file-start") + 1], "1")
        self.assertIn("--file-end", command)
        self.assertEqual(command[command.index("--file-end") + 1], "3")

    def test_build_process_features_launch_config_accepts_feature_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "processed-output"
            input_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a", lr_shape=(2, 2, 2, 390))

            payload = app_module.RunProcessFeaturesRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                save_feature_statistics=True,
                ignore_trailing_channels=True,
                trailing_channels=6,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                validation, launch_config = app_module._build_process_features_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertTrue(launch_config["save_feature_statistics"])
        self.assertTrue(launch_config["ignore_trailing_channels"])
        self.assertEqual(launch_config["trailing_channels"], 6)

        command = app_module._build_process_features_command(launch_config)
        self.assertIn("--save-feature-statistics", command)
        self.assertIn("--ignore-trailing-channels", command)
        self.assertIn("--trailing-channels", command)
        self.assertEqual(command[command.index("--trailing-channels") + 1], "6")

    def test_build_process_features_launch_config_rejects_feature_statistics_trailing_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "processed-output"
            input_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a", lr_shape=(2, 2, 2, 4))

            payload = app_module.RunProcessFeaturesRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                save_feature_statistics=True,
                ignore_trailing_channels=True,
                trailing_channels=4,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                validation, launch_config = app_module._build_process_features_launch_config(payload)

        self.assertEqual(validation["valid"], False)
        self.assertEqual(validation["reasonCode"], "invalid_trailing_channels")
        self.assertIsNone(launch_config)

    def test_build_process_features_launch_config_rejects_empty_file_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "processed-output"
            input_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name)

            payload = app_module.RunProcessFeaturesRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                file_range=app_module.InferenceFileRangeRequest(start=1, end=0),
                save_high_resolution_features=True,
                high_resolution_save_format=".tif",
                save_pca=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                validation, launch_config = app_module._build_process_features_launch_config(payload)

        self.assertEqual(validation["valid"], False)
        self.assertEqual(validation["reasonCode"], "empty_file_selection")
        self.assertIsNone(launch_config)

    def test_build_process_features_launch_config_rejects_missing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "processed-output"
            input_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a")

            payload = app_module.RunProcessFeaturesRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                save_high_resolution_features=False,
                save_pca=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                validation, launch_config = app_module._build_process_features_launch_config(payload)

        self.assertEqual(validation["valid"], False)
        self.assertEqual(validation["reasonCode"], "no_outputs_selected")
        self.assertIsNone(launch_config)

    def test_build_segmentation_launch_config_accepts_valid_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            input_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a")

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                enable_voronoi_otsu=True,
                gaussian_blur_sigma=3,
                rolling_ball_radius=10.0,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                validation, launch_config = app_module._build_segmentation_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertEqual(validation["subfolderCount"], 1)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["gpu_index"], 0)
        self.assertEqual(launch_config["subfolder_count"], 1)
        self.assertEqual(launch_config["mode"], app_module.SEGMENTATION_MODE_VORONOI_OTSU)
        self.assertEqual(launch_config["gaussian_blur_sigma"], 3)
        self.assertEqual(launch_config["rolling_ball_radius"], 10.0)
        self.assertTrue(launch_config["enable_voronoi_otsu"])
        self.assertEqual(launch_config["output_path"], output_dir)

        command = app_module._build_segmentation_command(launch_config)
        self.assertIn("scripts/post_processing/segmentation.py", command)
        self.assertIn("--output-path", command)
        self.assertIn("--enable-voronoi-otsu", command)
        self.assertIn("--gaussian-blur-sigma", command)
        self.assertIn("--rolling-ball-radius", command)

    def test_validate_segmentation_input_folder_reports_subfolder_names_and_probmap_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name)
            np.savez_compressed(input_dir / "probmap_densities.npz", x1=np.array([], dtype=object))

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                validation = app_module.validate_segmentation_input_folder(str(input_dir))

        self.assertEqual(validation["valid"], True)
        self.assertEqual(validation["subfolderNames"], ["sample_a", "sample_b"])
        self.assertTrue(validation["probmapDensitiesExists"])
        self.assertEqual(validation["probmapDensitiesPath"], str(input_dir / "probmap_densities.npz"))

    def test_validate_general_segmentation_input_discovers_optional_sources_and_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            raw_dir = input_dir / "raw"
            probmap_dir = input_dir / "probmap"
            feature_stats_dir = input_dir / "feature_stats"
            pca_dir = input_dir / "pca_3"
            raw_dir.mkdir(parents=True)
            probmap_dir.mkdir()
            feature_stats_dir.mkdir()
            pca_dir.mkdir()
            (feature_stats_dir / "feature_stats_metadata.json").write_text(
                json.dumps(
                    {
                        "components": ["mean", "max", "min", "median", "std", "l2_norm"],
                        "dtype": "float32",
                        "source": "lr_feats",
                        "computed_before_upsampling": True,
                    }
                ),
                encoding="utf-8",
            )
            for name in ("sample_a", "sample_b"):
                tifffile.imwrite(raw_dir / f"{name}.tif", np.zeros((2, 3, 4), dtype=np.uint16), photometric="minisblack")
                tifffile.imwrite(
                    probmap_dir / f"{name}.tif",
                    np.zeros((2, 3, 4), dtype=np.float32),
                    photometric="minisblack",
                )
                tifffile.imwrite(feature_stats_dir / f"{name}.tif", np.zeros((2, 3, 4, 6), dtype=np.float32))
                tifffile.imwrite(pca_dir / f"{name}.tif", np.zeros((2, 3, 4, 3), dtype=np.uint8))
            bad_pca_dir = input_dir / "pca_2"
            bad_pca_dir.mkdir()
            tifffile.imwrite(bad_pca_dir / "sample_a.tif", np.zeros((2, 3, 4, 2), dtype=np.uint8))

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                validation = app_module.validate_general_segmentation_input_folder(str(input_dir))

        self.assertEqual(validation["valid"], True)
        self.assertEqual(validation["subfolderNames"], ["sample_a", "sample_b"])
        self.assertEqual(
            [source["id"] for source in validation["availableSources"]],
            ["raw", "probmap", "feature_stats", "pca_3"],
        )
        self.assertEqual(validation["availableSources"][2]["componentCount"], 6)
        self.assertEqual(
            validation["availableSources"][2]["componentNames"],
            ["mean", "max", "min", "median", "std", "l2_norm"],
        )
        self.assertEqual(validation["availableSources"][3]["componentCount"], 3)
        self.assertEqual(validation["sourceWarnings"][0]["folderName"], "pca_2")
        self.assertIn("missing sample_b", validation["sourceWarnings"][0]["message"])

    def test_build_segmentation_launch_config_accepts_legacy_probability_map_request_with_density_estimation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            input_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name)
            seg_tif = root / "seg.tif"
            valid_mask_tif = root / "valid_mask.tif"
            tifffile.imwrite(seg_tif, np.zeros((4, 4, 4), dtype=np.uint8))
            tifffile.imwrite(valid_mask_tif, np.ones((4, 4, 4), dtype=np.uint8))

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=2,
                mode=app_module.SEGMENTATION_MODE_LEGACY_PROBABILITY_MAP,
                run_density_estimation=True,
                training_timepoint="sample_b",
                seg_tif=str(seg_tif),
                valid_mask_tif=str(valid_mask_tif),
                density_method="gpu-hist",
                feature_batch=24,
                kde_points=256,
                kde_max_samples=1000,
                hist_sigma_bins=2.0,
                bg_prob_threshold=0.3,
                fg_prob_threshold=0.9,
                seed=9,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 2, "name": "GPU-2"}]},
                ),
            ):
                validation, launch_config = app_module._build_segmentation_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["mode"], app_module.SEGMENTATION_MODE_LEGACY_PROBABILITY_MAP)
        self.assertTrue(launch_config["run_density_estimation"])
        self.assertEqual(launch_config["training_timepoint"], "sample_b")
        self.assertEqual(launch_config["feature_batch"], 24)
        self.assertEqual(launch_config["kde_points"], 256)
        self.assertEqual(launch_config["kde_max_samples"], 1000)
        self.assertEqual(launch_config["hist_sigma_bins"], 2.0)
        self.assertEqual(launch_config["bg_prob_threshold"], 0.3)
        self.assertEqual(launch_config["fg_prob_threshold"], 0.9)
        self.assertEqual(launch_config["seed"], 9)
        self.assertEqual(launch_config["progress_total"], 4)
        self.assertEqual(launch_config["stage_2_output"], "legacy")
        self.assertEqual(launch_config["densities_path"], output_dir / "probmap_densities.npz")
        self.assertEqual(launch_config["output_path"], output_dir)

        command = app_module._build_segmentation_command(launch_config)
        self.assertIn("scripts/post_processing/probability_map.py", command)
        self.assertIn("--output-path", command)
        self.assertIn("--run-density-estimation", command)
        self.assertIn("--training-timepoint", command)
        self.assertIn("--seg-tif", command)
        self.assertIn("--valid-mask-tif", command)
        self.assertIn("--density-method", command)
        self.assertIn("--bg-prob-threshold", command)
        self.assertIn("--fg-prob-threshold", command)
        self.assertIn("--stage-2-output", command)
        self.assertIn("legacy", command)

    def test_build_segmentation_launch_config_rejects_legacy_probability_map_without_saved_densities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            input_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a")

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                mode=app_module.SEGMENTATION_MODE_LEGACY_PROBABILITY_MAP,
                run_density_estimation=False,
                densities_path=None,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                validation, launch_config = app_module._build_segmentation_launch_config(payload)

        self.assertEqual(validation["valid"], False)
        self.assertEqual(validation["reasonCode"], "missing_probmap_densities")
        self.assertIsNone(launch_config)

    def test_build_segmentation_launch_config_accepts_general_probability_map_thresholding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            input_dir.mkdir()
            probmap_dir = input_dir / "probmap"
            probmap_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name)
                tifffile.imwrite(
                    probmap_dir / f"{name}.tif",
                    np.zeros((4, 4, 4), dtype=np.float32),
                    photometric="minisblack",
                )

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=None,
                mode=app_module.SEGMENTATION_MODE_GENERAL,
                source_id="probmap",
                threshold=0.6,
                invert_mask=True,
                run_connected_components=False,
            )

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                validation, launch_config = app_module._build_segmentation_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["mode"], app_module.SEGMENTATION_MODE_GENERAL)
        self.assertEqual(launch_config["source_id"], "probmap")
        self.assertEqual(launch_config["source_kind"], "probmap")
        self.assertEqual(launch_config["threshold"], 0.6)
        self.assertEqual(launch_config["component_index"], 0)
        self.assertTrue(launch_config["invert_mask"])
        self.assertFalse(launch_config["run_connected_components"])
        self.assertEqual(launch_config["progress_total"], 2)

        command = app_module._build_segmentation_command(launch_config)
        self.assertIn("scripts/post_processing/general_segmentation.py", command)
        self.assertIn("--source-kind", command)
        self.assertIn("probmap", command)
        self.assertIn("--threshold", command)
        self.assertIn("0.6", command)
        self.assertIn("--invert-mask", command)
        self.assertIn("--skip-ccl", command)

    def test_build_segmentation_launch_config_serializes_general_processing_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            tifffile.imwrite(raw_dir / "sample_a.tif", np.zeros((2, 2, 2), dtype=np.uint8), photometric="minisblack")

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=None,
                mode=app_module.SEGMENTATION_MODE_GENERAL,
                source_id="raw",
                threshold=12.0,
                data_operations=[
                    {"type": "invert_lut"},
                    {"type": "gaussian_smoothing", "sigma": 1.5},
                ],
                mask_operations=[{"type": "remove_small_objects", "size": 8}],
                instance_method="voronoi_otsu",
                voronoi_spot_sigma=2.5,
                voronoi_outline_sigma=3.0,
            )

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                validation, launch_config = app_module._build_segmentation_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(
            launch_config["data_operations"],
            [{"type": "invert_lut"}, {"type": "gaussian_smoothing", "sigma": 1.5}],
        )
        self.assertEqual(launch_config["mask_operations"], [{"type": "remove_small_objects", "size": 8}])
        self.assertEqual(launch_config["instance_method"], "voronoi_otsu")
        self.assertEqual(launch_config["data_backend"], "cpu")

        command = app_module._build_segmentation_command(launch_config)
        data_operations = json.loads(command[command.index("--data-operations-json") + 1])
        mask_operations = json.loads(command[command.index("--mask-operations-json") + 1])
        self.assertEqual(data_operations, [{"type": "invert_lut"}, {"type": "gaussian_smoothing", "sigma": 1.5}])
        self.assertEqual(mask_operations, [{"type": "remove_small_objects", "size": 8}])
        self.assertEqual(command[command.index("--data-backend") + 1], "cpu")
        self.assertEqual(command[command.index("--instance-method") + 1], "voronoi_otsu")
        self.assertEqual(command[command.index("--voronoi-spot-sigma") + 1], "2.5")
        self.assertEqual(command[command.index("--voronoi-outline-sigma") + 1], "3.0")

    def test_build_segmentation_launch_config_for_distance_transform_watershed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            tifffile.imwrite(raw_dir / "sample_a.tif", np.zeros((2, 2, 2), dtype=np.uint8), photometric="minisblack")

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=None,
                mode=app_module.SEGMENTATION_MODE_GENERAL,
                source_id="raw",
                threshold=1.0,
                instance_method="distance_transform_watershed",
                distance_transform_dynamic=1.5,
                distance_transform_connectivity=26,
                distance_transform_spacing_z=2.0,
                distance_transform_spacing_y=1.0,
                distance_transform_spacing_x=1.0,
            )

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                validation, launch_config = app_module._build_segmentation_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["instance_method"], "distance_transform_watershed")
        self.assertEqual(launch_config["distance_transform_dynamic"], 1.5)
        self.assertEqual(launch_config["distance_transform_connectivity"], 26)
        self.assertEqual(launch_config["distance_transform_spacing"], (2.0, 1.0, 1.0))

        command = app_module._build_segmentation_command(launch_config)
        self.assertEqual(command[command.index("--instance-method") + 1], "distance_transform_watershed")
        self.assertEqual(command[command.index("--distance-transform-dynamic") + 1], "1.5")
        self.assertEqual(command[command.index("--distance-transform-connectivity") + 1], "26")
        self.assertEqual(command[command.index("--distance-transform-spacing-z") + 1], "2.0")
        self.assertEqual(command[command.index("--distance-transform-spacing-y") + 1], "1.0")
        self.assertEqual(command[command.index("--distance-transform-spacing-x") + 1], "1.0")

    def test_build_segmentation_launch_config_for_intensity_prominence_watershed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            tifffile.imwrite(raw_dir / "sample_a.tif", np.zeros((2, 2, 2), dtype=np.uint8), photometric="minisblack")

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=None,
                mode=app_module.SEGMENTATION_MODE_GENERAL,
                source_id="raw",
                threshold=1.0,
                instance_method="intensity_prominence_watershed",
                intensity_prominence=0.2,
                intensity_smoothing_sigma=0.5,
                intensity_low_percentile=2.0,
                intensity_high_percentile=98.0,
                intensity_connectivity=26,
            )

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                validation, launch_config = app_module._build_segmentation_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["instance_method"], "intensity_prominence_watershed")
        self.assertEqual(launch_config["intensity_prominence"], 0.2)
        self.assertEqual(launch_config["intensity_smoothing_sigma"], 0.5)
        self.assertEqual(launch_config["intensity_percentiles"], (2.0, 98.0))
        self.assertEqual(launch_config["intensity_connectivity"], 26)

        command = app_module._build_segmentation_command(launch_config)
        self.assertEqual(command[command.index("--instance-method") + 1], "intensity_prominence_watershed")
        self.assertEqual(command[command.index("--intensity-prominence") + 1], "0.2")
        self.assertEqual(command[command.index("--intensity-smoothing-sigma") + 1], "0.5")
        self.assertEqual(command[command.index("--intensity-low-percentile") + 1], "2.0")
        self.assertEqual(command[command.index("--intensity-high-percentile") + 1], "98.0")
        self.assertEqual(command[command.index("--intensity-connectivity") + 1], "26")

    def test_build_segmentation_launch_config_uses_selected_gpu_for_general_data_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            tifffile.imwrite(raw_dir / "sample_a.tif", np.zeros((2, 2, 2), dtype=np.uint8), photometric="minisblack")

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=2,
                mode=app_module.SEGMENTATION_MODE_GENERAL,
                source_id="raw",
                threshold=12.0,
                data_operations=[{"type": "subtract_background", "radius": 10.0}],
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 2, "name": "GPU-2"}]},
                ),
            ):
                validation, launch_config = app_module._build_segmentation_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["gpu_index"], 2)
        self.assertEqual(launch_config["data_backend"], "gpu")

        command = app_module._build_segmentation_command(launch_config)
        self.assertEqual(command[command.index("--data-backend") + 1], "gpu")
        self.assertEqual(command[command.index("--gpu-index") + 1], "0")
        command_env = app_module._build_general_segmentation_command_env(launch_config)
        self.assertEqual(command_env["CUDA_VISIBLE_DEVICES"], "2")

    def test_build_segmentation_launch_config_accepts_general_raw_without_lr_feats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            for name in ("sample_a", "sample_b"):
                tifffile.imwrite(raw_dir / f"{name}.tif", np.zeros((4, 4, 4), dtype=np.uint8), photometric="minisblack")

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=None,
                mode=app_module.SEGMENTATION_MODE_GENERAL,
                source_id="raw",
                threshold=12.0,
                run_connected_components=False,
            )

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                folder_validation = app_module.validate_general_segmentation_input_folder(str(input_dir))
                validation, launch_config = app_module._build_segmentation_launch_config(payload)

        self.assertEqual(folder_validation["valid"], True)
        self.assertEqual(folder_validation["subfolderNames"], ["sample_a", "sample_b"])
        self.assertEqual([source["id"] for source in folder_validation["availableSources"]], ["raw"])
        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["subfolder_count"], 2)
        self.assertEqual(launch_config["source_kind"], "raw")
        self.assertEqual(launch_config["threshold"], 12.0)

    def test_probability_map_preview_metadata_reports_timepoint_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            probmap_dir = input_dir / "probmap"
            probmap_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, raw_shape=(3, 4, 5))
                tifffile.imwrite(
                    probmap_dir / f"{name}.tif",
                    np.zeros((3, 4, 5), dtype=np.float32),
                    photometric="minisblack",
                )

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.probability_map_preview_metadata(str(input_dir))

        self.assertEqual(payload["valid"], True)
        self.assertEqual(payload["defaultTimepoint"], "sample_a")
        self.assertEqual([item["name"] for item in payload["timepoints"]], ["sample_a", "sample_b"])
        self.assertEqual(payload["timepoints"][0]["shape"], {"z": 3, "y": 4, "x": 5})
        self.assertEqual(payload["timepoints"][0]["width"], 5)
        self.assertEqual(payload["timepoints"][0]["height"], 4)
        self.assertEqual(payload["timepoints"][0]["zCount"], 3)
        self.assertTrue(payload["timepoints"][0]["compatible"])

    def test_probability_map_preview_image_returns_slice_and_max_projection_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            probmap_dir = input_dir / "probmap"
            probmap_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a", raw_shape=(2, 2, 3))
            raw = np.array(
                [
                    [[0, 10, 20], [30, 40, 50]],
                    [[60, 70, 80], [90, 100, 110]],
                ],
                dtype=np.uint16,
            )
            probmap = np.array(
                [
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    [[0.7, 0.8, 0.9], [1.0, 0.0, 0.25]],
                ],
                dtype=np.float32,
            )
            tifffile.imwrite(input_dir / "raw" / "sample_a.tif", raw, photometric="minisblack")
            tifffile.imwrite(probmap_dir / "sample_a.tif", probmap, photometric="minisblack")

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                slice_payload = app_module.probability_map_preview_image(
                    app_module.ProbabilityMapPreviewImageRequest(
                        input_path=str(input_dir),
                        timepoint="sample_a",
                        view="slice",
                        z_index=1,
                    )
                )
                projection_payload = app_module.probability_map_preview_image(
                    app_module.ProbabilityMapPreviewImageRequest(
                        input_path=str(input_dir),
                        timepoint="sample_a",
                        view="max_projection",
                    )
                )

        self.assertEqual(slice_payload["width"], 3)
        self.assertEqual(slice_payload["height"], 2)
        self.assertEqual(slice_payload["zIndex"], 1)
        slice_probability = np.frombuffer(
            base64.b64decode(slice_payload["probability"]["data"]),
            dtype=np.float32,
        ).reshape(2, 3)
        np.testing.assert_allclose(slice_probability, probmap[1], rtol=1e-6, atol=1e-6)
        self.assertEqual(
            len(base64.b64decode(slice_payload["raw"]["data"])),
            2 * 3,
        )

        projection_probability = np.frombuffer(
            base64.b64decode(projection_payload["probability"]["data"]),
            dtype=np.float32,
        ).reshape(2, 3)
        np.testing.assert_allclose(projection_probability, probmap.max(axis=0), rtol=1e-6, atol=1e-6)
        expected_mask = probmap.max(axis=0) >= 0.5
        preview_mask = projection_probability >= 0.5
        np.testing.assert_array_equal(preview_mask, expected_mask)

    def test_general_segmentation_preview_surface_raw_slice_does_not_compute_processed_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            raw = np.array(
                [
                    [[0, 10, 20], [30, 40, 50]],
                    [[60, 70, 80], [90, 100, 110]],
                ],
                dtype=np.uint16,
            )
            tifffile.imwrite(raw_dir / "sample_a.tif", raw, photometric="minisblack")

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                with patch(
                    "spatialdino_server.app._general_processed_volume",
                    side_effect=AssertionError("processed data should not be read for raw display slices"),
                ):
                    payload = app_module.general_segmentation_preview_surface(
                        app_module.GeneralSegmentationPreviewSurfaceRequest(
                            input_path=str(input_dir),
                            source_id="raw",
                            display_contrast="full_range",
                            timepoint="sample_a",
                            view="slice",
                            z_index=1,
                        )
                    )

        display = np.frombuffer(base64.b64decode(payload["display"]["data"]), dtype=np.uint8).reshape(2, 3)
        expected_display = np.rint((raw[1] / 110.0) * 255.0).astype(np.uint8)
        np.testing.assert_array_equal(display, expected_display)
        values = np.frombuffer(base64.b64decode(payload["evaluation"]["values"]["data"]), dtype=np.float32).reshape(2, 3)
        np.testing.assert_allclose(values, raw[1].astype(np.float32), rtol=1e-6, atol=1e-6)
        self.assertNotIn("values", payload)

    def test_general_segmentation_preview_surface_projection_returns_exact_threshold_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            raw = np.array(
                [
                    [[0, 10, 20], [30, 40, 50]],
                    [[60, 70, 80], [90, 100, 110]],
                ],
                dtype=np.uint16,
            )
            tifffile.imwrite(raw_dir / "sample_a.tif", raw, photometric="minisblack")

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.general_segmentation_preview_surface(
                    app_module.GeneralSegmentationPreviewSurfaceRequest(
                        input_path=str(input_dir),
                        source_id="raw",
                        display_source="raw",
                        timepoint="sample_a",
                        view="max_projection",
                    )
                )

        max_values = np.frombuffer(base64.b64decode(payload["evaluation"]["maxValues"]["data"]), dtype=np.float32).reshape(2, 3)
        min_values = np.frombuffer(base64.b64decode(payload["evaluation"]["minValues"]["data"]), dtype=np.float32).reshape(2, 3)
        np.testing.assert_allclose(max_values, raw.max(axis=0).astype(np.float32), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(min_values, raw.min(axis=0).astype(np.float32), rtol=1e-6, atol=1e-6)

    def test_general_segmentation_preview_data_returns_raw_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            raw = np.array(
                [
                    [[0, 10, 20], [30, 40, 50]],
                    [[60, 70, 80], [90, 100, 110]],
                ],
                dtype=np.uint16,
            )
            tifffile.imwrite(raw_dir / "sample_a.tif", raw, photometric="minisblack")

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.general_segmentation_preview_data(
                    app_module.GeneralSegmentationPreviewDataRequest(
                        input_path=str(input_dir),
                        source_id="raw",
                        timepoint="sample_a",
                    )
                )

        self.assertEqual(payload["rangeMin"], 0.0)
        self.assertEqual(payload["rangeMax"], 110.0)
        self.assertEqual(payload["step"], 1.0)

    def test_general_segmentation_preview_data_runs_selected_gpu_processing_out_of_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            raw = np.arange(8, dtype=np.uint16).reshape(2, 2, 2)
            tifffile.imwrite(raw_dir / "sample_a.tif", raw, photometric="minisblack")

            def fake_gpu_processing(values, *, source_kind, data_operations, gpu_index):
                self.assertEqual(data_operations, [{"type": "gaussian_smoothing", "sigma": 0.0}])
                self.assertEqual(source_kind, "raw")
                self.assertEqual(gpu_index, 0)
                return np.asarray(values, dtype=np.float32)

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
                patch(
                    "spatialdino_server.app._run_general_preview_gpu_processing",
                    side_effect=fake_gpu_processing,
                ) as gpu_processing_mock,
                patch(
                    "spatialdino_server.app.apply_data_operations",
                    side_effect=AssertionError("GPU preview processing must not run in the server process."),
                ),
            ):
                payload = app_module.general_segmentation_preview_data(
                    app_module.GeneralSegmentationPreviewDataRequest(
                        input_path=str(input_dir),
                        gpu_index=0,
                        source_id="raw",
                        data_operations=[{"type": "gaussian_smoothing", "sigma": 0}],
                        timepoint="sample_a",
                    )
                )

        self.assertEqual(gpu_processing_mock.call_count, 1)
        self.assertEqual(payload["rangeMin"], 0.0)
        self.assertEqual(payload["rangeMax"], 7.0)

    def test_general_preview_gpu_processing_limits_visible_gpu_in_subprocess(self) -> None:
        values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        captured: dict[str, object] = {}

        def fake_run(command, *, cwd, env, text, capture_output, check):
            captured["command"] = command
            captured["cwd"] = cwd
            captured["env"] = env
            captured["text"] = text
            captured["capture_output"] = capture_output
            captured["check"] = check
            output_npy = Path(command[command.index("--output-npy") + 1])
            np.save(output_npy, values + 1, allow_pickle=False)
            return app_module.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch("spatialdino_server.app.subprocess.run", side_effect=fake_run):
            processed = app_module._run_general_preview_gpu_processing(
                values,
                source_kind="raw",
                data_operations=[{"type": "gaussian_smoothing", "sigma": 0.0}],
                gpu_index=3,
            )

        command = captured["command"]
        env = captured["env"]
        assert isinstance(command, list)
        assert isinstance(env, dict)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(command[command.index("--gpu-index") + 1], "0")
        np.testing.assert_array_equal(processed, values + 1)

    def test_general_segmentation_preview_splits_image_and_binary_mask_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            raw = np.array(
                [
                    [[0, 10, 20], [30, 40, 50]],
                    [[60, 70, 80], [90, 100, 110]],
                ],
                dtype=np.uint16,
            )
            tifffile.imwrite(raw_dir / "sample_a.tif", raw, photometric="minisblack")

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                data_payload = app_module.general_segmentation_preview_data(
                    app_module.GeneralSegmentationPreviewDataRequest(
                        input_path=str(input_dir),
                        source_id="raw",
                        data_operations=[{"type": "invert_lut"}],
                        timepoint="sample_a",
                    )
                )
                payload = app_module.general_segmentation_preview_image(
                    app_module.GeneralSegmentationPreviewImageRequest(
                        input_path=str(input_dir),
                        source_id="raw",
                        display_source="evaluation",
                        data_operations=[{"type": "invert_lut"}],
                        require_cached_data=True,
                        timepoint="sample_a",
                        view="min_projection",
                    )
                )
                mask_payload = app_module.general_segmentation_preview_mask(
                    app_module.GeneralSegmentationPreviewMaskRequest(
                        input_path=str(input_dir),
                        source_id="raw",
                        threshold=80.0,
                        data_operations=[{"type": "invert_lut"}],
                        require_cached_data=True,
                        timepoint="sample_a",
                        view="min_projection",
                    )
                )

        self.assertNotIn("values", payload)
        self.assertNotIn("mask", payload)
        mask = np.frombuffer(base64.b64decode(mask_payload["mask"]["data"]), dtype=np.uint32).reshape(2, 3)
        expected_mask = ((110.0 - raw.astype(np.float32)) >= 80.0).max(axis=0).astype(np.uint32)
        np.testing.assert_array_equal(mask, expected_mask)
        self.assertFalse(mask_payload["mask"]["instance"])
        self.assertEqual(data_payload["rangeMin"], 0.0)
        self.assertEqual(data_payload["rangeMax"], 110.0)
        self.assertAlmostEqual(data_payload["step"], 0.11)

    def test_general_segmentation_preview_instance_uses_explicit_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            raw_dir = input_dir / "raw"
            raw_dir.mkdir(parents=True)
            raw = np.array(
                [
                    [[0, 0, 0], [0, 90, 90]],
                    [[0, 0, 0], [0, 90, 90]],
                ],
                dtype=np.uint16,
            )
            tifffile.imwrite(raw_dir / "sample_a.tif", raw, photometric="minisblack")

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.general_segmentation_preview_instance(
                    app_module.GeneralSegmentationPreviewInstanceRequest(
                        input_path=str(input_dir),
                        source_id="raw",
                        threshold=50.0,
                        instance_method="connected_components",
                        timepoint="sample_a",
                        view="max_projection",
                    )
                )

        mask = np.frombuffer(base64.b64decode(payload["mask"]["data"]), dtype=np.uint32).reshape(2, 3)
        self.assertTrue(payload["mask"]["instance"])
        self.assertGreater(int(mask.max()), 0)

    def test_general_segmentation_preview_data_returns_pca_component_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            raw_dir = input_dir / "raw"
            pca_dir = input_dir / "pca_2"
            raw_dir.mkdir(parents=True)
            pca_dir.mkdir()
            tifffile.imwrite(raw_dir / "sample_a.tif", np.zeros((2, 2, 3), dtype=np.uint8), photometric="minisblack")
            pca = np.zeros((2, 2, 3, 2), dtype=np.uint8)
            pca[..., 0] = 10
            pca[..., 1] = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
            tifffile.imwrite(pca_dir / "sample_a.tif", pca)

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.general_segmentation_preview_data(
                    app_module.GeneralSegmentationPreviewDataRequest(
                        input_path=str(input_dir),
                        source_id="pca_2",
                        threshold_component_index=1,
                        timepoint="sample_a",
                    )
                )

        self.assertEqual(payload["rangeMin"], 0.0)
        self.assertEqual(payload["rangeMax"], 255.0)

    def test_general_segmentation_preview_data_returns_feature_statistics_component_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            raw_dir = input_dir / "raw"
            feature_stats_dir = input_dir / "feature_stats"
            raw_dir.mkdir(parents=True)
            feature_stats_dir.mkdir()
            tifffile.imwrite(raw_dir / "sample_a.tif", np.zeros((2, 2, 3), dtype=np.uint8), photometric="minisblack")
            (feature_stats_dir / "feature_stats_metadata.json").write_text(
                json.dumps({"components": ["mean", "max", "min", "median", "std", "l2_norm"]}),
                encoding="utf-8",
            )
            feature_stats = np.zeros((2, 2, 3, 6), dtype=np.float32)
            feature_stats[..., 0] = 10.0
            feature_stats[..., 4] = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
            tifffile.imwrite(feature_stats_dir / "sample_a.tif", feature_stats)

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.general_segmentation_preview_data(
                    app_module.GeneralSegmentationPreviewDataRequest(
                        input_path=str(input_dir),
                        source_id="feature_stats",
                        threshold_component_index=4,
                        timepoint="sample_a",
                    )
                )

        self.assertEqual(payload["rangeMin"], 0.0)
        self.assertEqual(payload["rangeMax"], 11.0)
        self.assertAlmostEqual(payload["step"], 0.011)

    def test_probability_map_preview_metadata_marks_shape_mismatch_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            probmap_dir = input_dir / "probmap"
            probmap_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a", raw_shape=(3, 4, 5))
            tifffile.imwrite(
                probmap_dir / "sample_a.tif",
                np.zeros((2, 4, 5), dtype=np.float32),
                photometric="minisblack",
            )

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.probability_map_preview_metadata(str(input_dir))

        self.assertEqual(payload["valid"], True)
        self.assertFalse(payload["timepoints"][0]["compatible"])
        self.assertIn("do not match", payload["timepoints"][0]["message"])

    def test_build_foreground_probability_map_launch_config_accepts_density_estimation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "probmap-output"
            input_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name)
            seg_tif = root / "seg.tif"
            tifffile.imwrite(seg_tif, np.zeros((4, 4, 4), dtype=np.uint8))

            payload = app_module.RunForegroundProbabilityMapRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                run_density_estimation=True,
                training_timepoint="sample_a",
                seg_tif=str(seg_tif),
                feature_batch=16,
                kde_points=128,
                kde_max_samples=500,
                seed=3,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                validation, launch_config = app_module._build_foreground_probability_map_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["mode"], "foreground_probability_map")
        self.assertEqual(launch_config["stage_2_output"], "probmap")
        self.assertEqual(launch_config["progress_total"], 4)
        self.assertEqual(launch_config["densities_path"], output_dir / "probmap_densities.npz")

        command = app_module._build_foreground_probability_map_command(launch_config)
        self.assertIn("scripts/post_processing/probability_map.py", command)
        self.assertIn("--stage-2-output", command)
        self.assertIn("probmap", command)

    def test_validate_tracking_input_folder_accepts_valid_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, raw_dtype=np.uint16)

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_tracking_input_folder(str(input_dir))

        self.assertEqual(
            payload,
            {
                "valid": True,
                "message": "Valid feature folder. Found 2 timepoints.",
                "subfolderCount": 2,
                "subfolderNames": ["sample_a", "sample_b"],
            },
        )

    def test_validate_tracking_input_folder_rejects_insufficient_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a", raw_dtype=np.uint16)

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_tracking_input_folder(str(input_dir))

        self.assertEqual(payload["valid"], False)
        self.assertEqual(payload["reasonCode"], "insufficient_subfolders")
        self.assertEqual(payload["message"], "Tracking requires at least 2 timepoints.")

    def test_validate_tracking_segmentation_folder_rejects_missing_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            segmentation_dir = root / "segmentations"
            input_dir.mkdir()
            segmentation_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, raw_dtype=np.uint16)
            tifffile.imwrite(segmentation_dir / "sample_a.tif", np.zeros((4, 4, 4), dtype=np.uint32))

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_tracking_segmentation_folder(
                    str(input_dir),
                    str(segmentation_dir),
                )

        self.assertEqual(payload["valid"], False)
        self.assertEqual(payload["reasonCode"], "missing_required_files")
        self.assertEqual(payload["message"], "Segmentation folder is missing sample_b.tif.")

    def test_validate_tracking_segmentation_folder_accepts_matching_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            segmentation_dir = root / "segmentations"
            input_dir.mkdir()
            segmentation_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, raw_dtype=np.uint16)
                tifffile.imwrite(segmentation_dir / f"{name}.tif", np.zeros((4, 4, 4), dtype=np.uint32))

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_tracking_segmentation_folder(str(input_dir), str(segmentation_dir))

        self.assertEqual(
            payload,
            {
                "valid": True,
                "message": "Valid segmentation folder. Found 2 mask files.",
                "subfolderCount": 2,
                "subfolderNames": ["sample_a", "sample_b"],
            },
        )

    def test_validate_tracking_input_folder_ignores_hidden_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            input_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, raw_dtype=np.uint16)
            hidden_dir = input_dir / ".hidden_sample"
            hidden_dir.mkdir()

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = app_module.validate_tracking_input_folder(str(input_dir))

        self.assertEqual(payload["valid"], True)
        self.assertEqual(payload["subfolderCount"], 2)

    def test_build_tracking_launch_config_accepts_valid_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            segmentation_dir = root / "segmentations"
            output_dir = root / "tracking-output"
            input_dir.mkdir()
            segmentation_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, raw_dtype=np.uint16)
                tifffile.imwrite(segmentation_dir / f"{name}.tif", np.zeros((4, 4, 4), dtype=np.uint32))

            payload = app_module.RunTrackingRequest(
                input_path=str(input_dir),
                segmentation_path=str(segmentation_dir),
                output_path=str(output_dir),
                output_filename="debug_tracks",
                max_distance_xy=24.0,
                max_distance_z=12.0,
                z_distance_weight=2.0,
                min_distance_to_remove_cand=4.0,
                vote_thresholds="300,280,260",
                dice_threshold=0.6,
                corr_threshold=0.4,
                save_extended_results=True,
                ignore_features=True,
                disable_centroid_fallback=True,
                aggressive_feature_matching=True,
                min_feature_votes=2,
            )

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                validation, launch_config = app_module._build_tracking_launch_config(payload)

        self.assertEqual(validation["valid"], True)
        self.assertEqual(validation["subfolderCount"], 2)
        self.assertIsNotNone(launch_config)
        assert launch_config is not None
        self.assertEqual(launch_config["subfolder_count"], 2)
        self.assertEqual(launch_config["pair_count"], 1)
        self.assertEqual(launch_config["progress_total"], 3)
        self.assertEqual(launch_config["max_distance_xy"], 24.0)
        self.assertEqual(launch_config["max_distance_z"], 12.0)
        self.assertEqual(launch_config["z_distance_weight"], 2.0)
        self.assertEqual(launch_config["min_distance_to_remove_cand"], 4.0)
        self.assertEqual(launch_config["vote_thresholds"], (300, 280, 260))
        self.assertEqual(launch_config["dice_threshold"], 0.6)
        self.assertEqual(launch_config["corr_threshold"], 0.4)
        self.assertFalse(launch_config["invert_z"])
        self.assertEqual(launch_config["output_filename"], "debug_tracks.csv")
        self.assertTrue(launch_config["save_extended_results"])
        self.assertTrue(launch_config["ignore_features"])
        self.assertTrue(launch_config["disable_centroid_fallback"])
        self.assertTrue(launch_config["aggressive_feature_matching"])
        self.assertEqual(launch_config["min_feature_votes"], 2)
        self.assertEqual(launch_config["segmentation_path"], segmentation_dir)
        self.assertEqual(launch_config["output_path"], output_dir)

        command = app_module._build_tracking_command(launch_config)
        self.assertIn("scripts/post_processing/tracking.py", command)
        self.assertIn(str(segmentation_dir), command)
        self.assertIn("--output-path", command)
        self.assertIn(str(output_dir), command)
        self.assertIn("--output-filename", command)
        self.assertIn("debug_tracks.csv", command)
        self.assertIn("--max-distance-xy", command)
        self.assertIn("--max-distance-z", command)
        self.assertIn("--z-distance-weight", command)
        self.assertIn("--min-distance-to-remove-cand", command)
        self.assertIn("--vote-thresholds", command)
        self.assertIn("--dice-threshold", command)
        self.assertIn("--corr-threshold", command)
        self.assertIn("--min-feature-votes", command)
        self.assertEqual(command[command.index("--min-feature-votes") + 1], "2")
        self.assertNotIn("--invert-z", command)
        self.assertIn("--save-extended-results", command)
        self.assertIn("--ignore-features", command)
        self.assertIn("--disable-centroid-fallback", command)
        self.assertIn("--aggressive-feature-matching", command)

    def test_run_tracking_submits_job_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            segmentation_dir = root / "segmentations"
            output_dir = root / "tracking-output"
            input_dir.mkdir()
            segmentation_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, raw_dtype=np.uint16)
                tifffile.imwrite(segmentation_dir / f"{name}.tif", np.zeros((4, 4, 4), dtype=np.uint32))

            payload = app_module.RunTrackingRequest(
                input_path=str(input_dir),
                segmentation_path=str(segmentation_dir),
                output_path=str(output_dir),
                max_distance_xy=20.0,
                max_distance_z=10.0,
                z_distance_weight=2.5,
                min_distance_to_remove_cand=3.5,
                vote_thresholds="320,300,280,260",
                dice_threshold=0.5,
                corr_threshold=0.55,
                invert_z=True,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app._launch_tracking_job_thread") as launch_thread,
            ):
                response = app_module.run_tracking(payload, "client-1234")

        self.assertEqual(response["submitted"], True)
        self.assertIn("jobId", response)
        launch_thread.assert_called_once()
        launch_config = launch_thread.call_args.args[1]
        self.assertEqual(launch_config["max_distance_xy"], 20.0)
        self.assertEqual(launch_config["max_distance_z"], 10.0)
        self.assertEqual(launch_config["z_distance_weight"], 2.5)
        self.assertEqual(launch_config["min_distance_to_remove_cand"], 3.5)
        self.assertEqual(launch_config["vote_thresholds"], (320, 300, 280, 260))
        self.assertEqual(launch_config["dice_threshold"], 0.5)
        self.assertEqual(launch_config["corr_threshold"], 0.55)
        self.assertTrue(launch_config["invert_z"])
        self.assertFalse(launch_config["disable_centroid_fallback"])
        self.assertFalse(launch_config["aggressive_feature_matching"])
        self.assertEqual(launch_config["min_feature_votes"], 1)
        self.assertEqual(launch_config["segmentation_path"], segmentation_dir)
        self.assertEqual(launch_config["output_path"], output_dir)
        command = app_module._build_tracking_command(launch_config)
        self.assertIn("--invert-z", command)
        self.assertNotIn("--no-invert-z", command)
        with jobs_api._jobs_lock:
            self.assertEqual(len(jobs_api._jobs), 1)
            job = next(iter(jobs_api._jobs.values()))
            self.assertEqual(job.type, "tracking")
            self.assertEqual(job.total, 3)
            self.assertEqual(job.datasets, [{"source_dir": str(input_dir), "save_to": "tracking-output"}])

    def test_update_tracking_job_progress_from_output_tracks_preparation_and_pairs(self) -> None:
        job = jobs_api.JobState(
            job_id="job-track-progress",
            owner_client_id="client-1234",
            type="tracking",
            status="running",
            total=5,
        )
        progress_state: dict[str, object] = {}

        app_module._update_tracking_job_progress_from_output(job, "[tracking] Processing sample_a (1/3)", progress_state)
        with job.lock:
            self.assertEqual(job.current, "Preparing sample_a")
            self.assertEqual(job.processed, 0)

        app_module._update_tracking_job_progress_from_output(job, "[tracking] Completed sample_a", progress_state)
        with job.lock:
            self.assertEqual(job.current, "Prepared sample_a")
            self.assertEqual(job.processed, 1)

        app_module._update_tracking_job_progress_from_output(
            job,
            "[tracking] Matching sample_a -> sample_b (1/2)",
            progress_state,
        )
        with job.lock:
            self.assertEqual(job.current, "Matching sample_a -> sample_b")
            self.assertEqual(job.processed, 1)

        app_module._update_tracking_job_progress_from_output(
            job,
            "[tracking] Matched sample_a -> sample_b (1/2)",
            progress_state,
        )
        with job.lock:
            self.assertEqual(job.current, "Matched sample_a -> sample_b")
            self.assertEqual(job.processed, 2)

    def test_update_process_features_job_progress_accepts_probability_map_lines(self) -> None:
        job = jobs_api.JobState(
            job_id="job-probmap-progress",
            owner_client_id="client-1234",
            type="segmentation",
            status="running",
            total=4,
        )
        progress_state: dict[str, object] = {}

        app_module._update_process_features_job_progress_from_output(
            job,
            "[probmap] Processing density samples (1/2)",
            progress_state,
        )
        with job.lock:
            self.assertEqual(job.current, "Processing density samples")
            self.assertEqual(job.processed, 0)

        app_module._update_process_features_job_progress_from_output(
            job,
            "[probmap] Completed density samples",
            progress_state,
        )
        with job.lock:
            self.assertEqual(job.current, "density samples")
            self.assertEqual(job.processed, 1)

        app_module._update_process_features_job_progress_from_output(
            job,
            "[probmap] Completed density fitting",
            progress_state,
        )
        with job.lock:
            self.assertEqual(job.current, "density fitting")
            self.assertEqual(job.processed, 2)

    def test_build_tracking_launch_config_rejects_invalid_vote_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            segmentation_dir = root / "segmentations"
            output_dir = root / "tracking-output"
            input_dir.mkdir()
            segmentation_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, raw_dtype=np.uint16)
                tifffile.imwrite(segmentation_dir / f"{name}.tif", np.zeros((4, 4, 4), dtype=np.uint32))

            payload = app_module.RunTrackingRequest(
                input_path=str(input_dir),
                segmentation_path=str(segmentation_dir),
                output_path=str(output_dir),
                vote_thresholds="320,0,280",
            )

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                validation, launch_config = app_module._build_tracking_launch_config(payload)

        self.assertEqual(validation["valid"], False)
        self.assertEqual(validation["reasonCode"], "invalid_vote_thresholds")
        self.assertIsNone(launch_config)

    def test_build_tracking_launch_config_rejects_invalid_output_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            segmentation_dir = root / "segmentations"
            output_dir = root / "tracking-output"
            input_dir.mkdir()
            segmentation_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name, raw_dtype=np.uint16)
                tifffile.imwrite(segmentation_dir / f"{name}.tif", np.zeros((4, 4, 4), dtype=np.uint32))

            payload = app_module.RunTrackingRequest(
                input_path=str(input_dir),
                segmentation_path=str(segmentation_dir),
                output_path=str(output_dir),
                output_filename="../tracks.csv",
            )

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                validation, launch_config = app_module._build_tracking_launch_config(payload)

        self.assertEqual(validation["valid"], False)
        self.assertEqual(validation["reasonCode"], "invalid_output_filename")
        self.assertIsNone(launch_config)

    def test_run_segmentation_submits_job_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            input_dir.mkdir()
            write_inference_output_timepoint(input_dir, "sample_a")

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                enable_voronoi_otsu=True,
                gaussian_blur_sigma=3,
                rolling_ball_radius=10.0,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
                patch("spatialdino_server.app._launch_segmentation_job_thread") as launch_thread,
            ):
                response = app_module.run_segmentation(payload, "client-1234")

        self.assertEqual(response["submitted"], True)
        self.assertIn("jobId", response)
        launch_thread.assert_called_once()
        launch_config = launch_thread.call_args.args[1]
        self.assertEqual(launch_config["gaussian_blur_sigma"], 3)
        self.assertEqual(launch_config["rolling_ball_radius"], 10.0)
        self.assertTrue(launch_config["enable_voronoi_otsu"])
        self.assertEqual(launch_config["output_path"], output_dir)
        with jobs_api._jobs_lock:
            self.assertEqual(len(jobs_api._jobs), 1)
            job = next(iter(jobs_api._jobs.values()))
            self.assertEqual(job.type, "segmentation")
            self.assertEqual(job.total, 1)
            self.assertEqual(job.datasets, [{"source_dir": str(input_dir), "save_to": "segmentation-output"}])

    def test_run_segmentation_legacy_probability_map_submits_job_with_density_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "features"
            output_dir = root / "segmentation-output"
            input_dir.mkdir()
            for name in ("sample_a", "sample_b"):
                write_inference_output_timepoint(input_dir, name)
            seg_tif = root / "seg.tif"
            tifffile.imwrite(seg_tif, np.zeros((4, 4, 4), dtype=np.uint8))

            payload = app_module.RunSegmentationRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                gpu_index=0,
                mode=app_module.SEGMENTATION_MODE_LEGACY_PROBABILITY_MAP,
                run_density_estimation=True,
                training_timepoint="sample_a",
                seg_tif=str(seg_tif),
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
                patch("spatialdino_server.app._launch_segmentation_job_thread") as launch_thread,
            ):
                response = app_module.run_segmentation(payload, "client-1234")

        self.assertEqual(response["submitted"], True)
        launch_thread.assert_called_once()
        launch_config = launch_thread.call_args.args[1]
        self.assertEqual(launch_config["mode"], app_module.SEGMENTATION_MODE_LEGACY_PROBABILITY_MAP)
        self.assertTrue(launch_config["run_density_estimation"])
        self.assertEqual(launch_config["progress_total"], 4)
        self.assertEqual(launch_config["output_path"], output_dir)
        with jobs_api._jobs_lock:
            self.assertEqual(len(jobs_api._jobs), 1)
            job = next(iter(jobs_api._jobs.values()))
            self.assertEqual(job.type, "segmentation")
            self.assertEqual(job.total, 4)
            self.assertEqual(job.datasets, [{"source_dir": str(input_dir), "save_to": "segmentation-output"}])

    def test_run_inference_requests_overwrite_confirmation_for_existing_inference_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            models_dir = root / "models"
            input_dir.mkdir()
            output_dir.mkdir()
            models_dir.mkdir()
            tifffile.imwrite(input_dir / "stack0001.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            write_inference_output_timepoint(output_dir, "stack0001")
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                padding_mode="replicate",
                crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 0},
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
        self.assertEqual(response["outputEntryCount"], 2)
        self.assertEqual(response["outputEntriesPreview"], ["lr_feats/", "raw/"])
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
                padding_mode="replicate",
                crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 0},
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
        self.assertEqual(launch_config["padding_mode"], "replicate")
        with jobs_api._jobs_lock:
            self.assertEqual(len(jobs_api._jobs), 1)
            job = next(iter(jobs_api._jobs.values()))
            self.assertEqual(job.type, "inference")
            self.assertEqual(job.total, 1)
            self.assertEqual(job.datasets, [{"source_dir": str(input_dir), "save_to": "output"}])

    def test_build_inference_launch_config_converts_inclusive_ui_bounds(self) -> None:
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
                padding_mode="edge",
                crop_bounds={"x_start": 1, "x_end": 2, "y_start": 0, "y_end": 1, "z_start": 0, "z_end": 0},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 0},
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
                validation, launch_config = app_module._build_inference_launch_config(
                    payload,
                    require_overwrite_confirmation=False,
                )

        self.assertEqual(validation["valid"], True)
        self.assertIsNotNone(launch_config)
        self.assertEqual(launch_config["file_end"], 1)
        self.assertEqual(launch_config["crop_params"], [0, 1, 0, 2, 1, 3])
        self.assertEqual(launch_config["effective_crop_params"], (0, 1, 0, 2, 1, 3))
        self.assertEqual(launch_config["padding_mode"], "edge")

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
                upsample_factor={"x": 1.0, "y": 2.0, "z": 3.0},
                route="streaming",
                precision="bfloat16",
                padding_mode="replicate",
                crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                anisotropy={"x": 1.0, "y": 2.0, "z": 3.0},
                file_range={"start": 0, "end": 0},
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
        self.assertIn("upsample_factor=[3.0,2.0,1.0]", response["command"])
        self.assertIn("isotropic_scale_factor=[3.0,2.0,1.0]", response["command"])
        self.assertIn("inference_route=streaming", response["command"])
        self.assertIn("dtype=bf16", response["command"])
        self.assertIn("padding_mode=replicate", response["command"])

    def test_run_inference_rejects_invalid_padding_mode(self) -> None:
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
                padding_mode="invalid",
                crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 0},
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
        self.assertEqual(response["reasonCode"], "invalid_padding_mode")

    def test_inference_command_preview_includes_backbone_model_overrides(self) -> None:
        cases = {
            app_module.INFERENCE_BACKBONE_MODEL_LEARNED: [
                "pos_embed_type=learned",
                "num_register_tokens=0",
                "num_tt_register_tokens=1",
                "ffn_layer=swiglufused",
            ],
            app_module.INFERENCE_BACKBONE_MODEL_NOPE: [
                "pos_embed_type=none",
                "num_register_tokens=0",
                "num_tt_register_tokens=1",
                "ffn_layer=mlp",
            ],
            app_module.INFERENCE_BACKBONE_MODEL_ROPE: [
                "pos_embed_type=rope",
                "num_register_tokens=4",
                "num_tt_register_tokens=0",
                "rope_theta=200",
                "rope_normalize_coords=true",
                "rope_coord_shift=0.15",
                "rope_coord_jitter=1.3",
                "rope_coord_rescale=1.5",
                "rope_drop_prob=0.1",
                "ffn_layer=swiglufused",
            ],
        }

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

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app.get_repo_root", return_value=root),
                patch(
                    "spatialdino_server.app.get_nvidia_gpu_memory",
                    return_value={"nvidiaSmiAvailable": True, "gpus": [{"index": 0, "name": "GPU-0"}]},
                ),
            ):
                for backbone_model, expected_overrides in cases.items():
                    with self.subTest(backbone_model=backbone_model):
                        payload = app_module.RunInferenceRequest(
                            input_path=str(input_dir),
                            output_path=str(output_dir),
                            backbone_weight="models/backbone.pth",
                            backbone_model=backbone_model,
                            gpu_indices=[0],
                            upsample_factor=3.0,
                            route="streaming",
                            precision="bfloat16",
                            crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                            anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                            file_range={"start": 0, "end": 0},
                            overwrite=False,
                        )

                        response = app_module.inference_command_preview(payload)

                        self.assertEqual(response["valid"], True)
                        for override in expected_overrides:
                            self.assertIn(override, response["command"])

    def test_run_inference_rejects_invalid_backbone_model(self) -> None:
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
                backbone_model="unknown",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 0},
                overwrite=False,
            )

            with (
                patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False),
                patch("spatialdino_server.app.get_repo_root", return_value=root),
            ):
                response = app_module.run_inference(payload, "client-1234")

        self.assertEqual(response["submitted"], False)
        self.assertEqual(response["valid"], False)
        self.assertEqual(response["reasonCode"], "invalid_backbone_model")

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
                crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 0},
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
                crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 0},
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
            write_inference_output_timepoint(output_dir, "stack0001")
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 0},
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
            "Inference-managed outputs already exist. Confirm overwrite to replace lr_feats/, raw/, tmp/, and norm_per_vol.txt while preserving other files in the folder.",
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
                crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 0, "end": 0},
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
            tifffile.imwrite(input_dir / "stack0002.tif", np.zeros((2, 3, 4), dtype=np.uint8))
            (models_dir / "backbone.pth").write_text("", encoding="utf-8")

            payload = app_module.RunInferenceRequest(
                input_path=str(input_dir),
                output_path=str(output_dir),
                backbone_weight="models/backbone.pth",
                gpu_indices=[0],
                upsample_factor=3.0,
                route="streaming",
                precision="bfloat16",
                crop_bounds={"x_start": 0, "x_end": 3, "y_start": 0, "y_end": 2, "z_start": 0, "z_end": 1},
                anisotropy={"x": 1.0, "y": 1.0, "z": 1.0},
                file_range={"start": 1, "end": 0},
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
        expected_output_dirs = {app_module._canonicalize_runtime_path("/tmp/out/lr_feats/stack0001.npy")}
        expected_feature_paths = {app_module._canonicalize_runtime_path("/tmp/out/lr_feats/stack0001.npy")}
        saved_feature_paths: set[str] = set()

        app_module._update_job_progress_from_output(
            job,
            "Saving to /tmp/out/lr_feats/stack0001.npy",
            expected_output_dirs,
            expected_feature_paths,
            saved_feature_paths,
        )
        with job.lock:
            self.assertEqual(job.processed, 0)
            self.assertEqual(job.current, "Processing stack0001")

        app_module._update_job_progress_from_output(
            job,
            "Saved features to /tmp/out/lr_feats/stack0001.npy",
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
        expected_output_dirs = {app_module._canonicalize_runtime_path("/tmp/out/lr_feats/stack0001.npy")}
        expected_feature_paths = {app_module._canonicalize_runtime_path("/tmp/out/lr_feats/stack0001.npy")}
        saved_feature_paths: set[str] = set()

        app_module._update_job_progress_from_output(
            job,
            "Saving to /tmp/out/lr_feats/stack0002.npy",
            expected_output_dirs,
            expected_feature_paths,
            saved_feature_paths,
        )
        app_module._update_job_progress_from_output(
            job,
            "Saved features to /tmp/out/lr_feats/stack0002.npy",
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
            (output_dir / "lr_feats").mkdir(parents=True)
            (output_dir / "lr_feats" / "stack0002.npy").write_bytes(b"")

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
            (output_dir / "lr_feats").mkdir(parents=True)
            (output_dir / "lr_feats" / "other.npy").write_bytes(b"")
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
                self.assertIn("0/1 expected feature files", job.error)
                self.assertIn("stack0001", job.error)
                self.assertTrue(job.log_available)
                self.assertIsNotNone(job.command)
                self.assertIsNotNone(job.log_path)

            self.assertTrue(Path(job.log_path).is_file())
            log_text = Path(job.log_path).read_text(encoding="utf-8")
            self.assertIn("[server] Command:", log_text)
            self.assertIn("0/1 expected feature files", log_text)

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
                feats_path = output_dir / "lr_feats" / "stack0001.npy"
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
                        f"Saving to {output_dir / 'lr_feats' / 'stack0001.npy'}\n",
                        f"Saved features to {output_dir / 'lr_feats' / 'stack0001.npy'}\n",
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
