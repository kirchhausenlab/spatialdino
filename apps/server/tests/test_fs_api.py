from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from spatialdino_server import fs_api


class FsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        with fs_api._CACHE_LOCK:
            fs_api._CACHE.clear()

    def tearDown(self) -> None:
        with fs_api._CACHE_LOCK:
            fs_api._CACHE.clear()

    def test_list_returns_directories_under_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alpha").mkdir()
            (root / "beta").mkdir()
            (root / ".hidden").mkdir()

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                payload = fs_api.fs_list(path=str(root), page="1", pageSize="50", sort="name", order="asc")

            self.assertEqual(payload["path"], str(root))
            self.assertEqual([item["name"] for item in payload["items"]], ["alpha", "beta"])

    def test_list_rejects_paths_outside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                with self.assertRaises(HTTPException) as context:
                    fs_api.fs_list(path="/")

            self.assertEqual(context.exception.status_code, 403)

    def test_mkdir_creates_directory_and_invalidates_cached_parent_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alpha").mkdir()

            payload = fs_api.CreateDirRequest.model_validate({"parentPath": str(root), "name": "gamma"})

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                before = fs_api.fs_list(path=str(root), page="1", pageSize="50", sort="name", order="asc")
                created = fs_api.fs_mkdir(payload)
                after = fs_api.fs_list(path=str(root), page="1", pageSize="50", sort="name", order="asc")

            self.assertEqual([item["name"] for item in before["items"]], ["alpha"])
            self.assertEqual(created["path"], str(root / "gamma"))
            self.assertEqual([item["name"] for item in after["items"]], ["alpha", "gamma"])

    def test_mkdir_rejects_duplicate_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alpha").mkdir()

            payload = fs_api.CreateDirRequest.model_validate({"parentPath": str(root), "name": "alpha"})

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                with self.assertRaises(HTTPException) as context:
                    fs_api.fs_mkdir(payload)

            self.assertEqual(context.exception.status_code, 409)
            self.assertEqual(context.exception.detail, "An entry with that name already exists.")

    def test_mkdir_rejects_invalid_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            payload = fs_api.CreateDirRequest.model_validate({"parentPath": str(root), "name": "nested/path"})

            with patch.dict(os.environ, {"SPATIALDINO_FS_ROOTS": str(root)}, clear=False):
                with self.assertRaises(HTTPException) as context:
                    fs_api.fs_mkdir(payload)

            self.assertEqual(context.exception.status_code, 400)
            self.assertEqual(context.exception.detail, "Folder name must not contain path separators.")
