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
