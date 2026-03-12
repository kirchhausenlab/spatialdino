from __future__ import annotations

import unittest
from unittest.mock import patch

from spatialdino_server import cli


class CliTests(unittest.TestCase):
    def test_main_invokes_uvicorn_with_spatialdino_app(self) -> None:
        with patch("spatialdino_server.cli.uvicorn.run") as run_mock:
            cli.main(["--host", "127.0.0.1", "--port", "9000", "--reload"])

        run_mock.assert_called_once_with(
            "spatialdino_server.app:app",
            host="127.0.0.1",
            port=9000,
            reload=True,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
