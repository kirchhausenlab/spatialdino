from __future__ import annotations

import unittest
from unittest.mock import patch

from spatialdino.logging.utils import SmoothedValue


class SmoothedValueTests(unittest.TestCase):
    def test_synchronize_between_processes_is_noop_without_initialized_process_group(
        self,
    ) -> None:
        meter = SmoothedValue()
        meter.update(3.0)

        with patch(
            "spatialdino.logging.utils.dist.is_available",
            return_value=True,
        ), patch(
            "spatialdino.logging.utils.dist.is_initialized",
            return_value=False,
        ), patch(
            "spatialdino.logging.utils.dist.barrier",
            side_effect=AssertionError("barrier should not be called"),
        ), patch(
            "spatialdino.logging.utils.dist.all_reduce",
            side_effect=AssertionError("all_reduce should not be called"),
        ):
            meter.synchronize_between_processes()

        self.assertEqual(meter.count, 1)
        self.assertEqual(meter.total, 3.0)
