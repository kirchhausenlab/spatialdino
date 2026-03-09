from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from spatialdino_server import fs_api


class FsApiTests(unittest.TestCase):
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
