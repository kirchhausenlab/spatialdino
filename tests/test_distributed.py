from __future__ import annotations

import unittest

import spatialdino.distributed as dist


class _NoSyncRecorder:
    def __init__(self) -> None:
        self.calls = 0
        self.entered = 0
        self.exited = 0

    def no_sync(self):
        self.calls += 1
        recorder = self

        class _NoSyncContext:
            def __enter__(self):
                recorder.entered += 1
                return None

            def __exit__(self, exc_type, exc, tb):
                recorder.exited += 1
                return False

        return _NoSyncContext()


class DistributedUtilsTests(unittest.TestCase):
    def test_should_sync_gradients_without_accumulation(self) -> None:
        self.assertTrue(
            dist.should_sync_gradients(
                world_size=2,
                accum_iter=1,
                micro_steps_in_step=0,
            )
        )

    def test_should_sync_gradients_on_final_accumulation_microbatch(self) -> None:
        self.assertFalse(
            dist.should_sync_gradients(
                world_size=2,
                accum_iter=2,
                micro_steps_in_step=0,
            )
        )
        self.assertTrue(
            dist.should_sync_gradients(
                world_size=2,
                accum_iter=2,
                micro_steps_in_step=1,
            )
        )

    def test_should_sync_gradients_without_distributed_training(self) -> None:
        self.assertTrue(
            dist.should_sync_gradients(
                world_size=1,
                accum_iter=4,
                micro_steps_in_step=0,
            )
        )

    def test_maybe_no_sync_uses_context_only_when_sync_is_deferred(self) -> None:
        recorder = _NoSyncRecorder()

        with dist.maybe_no_sync(recorder, should_sync=False):
            pass

        self.assertEqual(recorder.calls, 1)
        self.assertEqual(recorder.entered, 1)
        self.assertEqual(recorder.exited, 1)

        with dist.maybe_no_sync(recorder, should_sync=True):
            pass

        self.assertEqual(recorder.calls, 1)
