from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request

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
