from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from spatialdino.data.collate import mask_generator


def _load_pretrain_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train" / "pretrain.py"
    spec = importlib.util.spec_from_file_location("pretrain_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pretrain module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pretrain = _load_pretrain_module()


class PretrainTodoFixTests(unittest.TestCase):
    def test_validate_crop_counts_requires_positive_global_and_local_crops(self) -> None:
        with self.assertRaisesRegex(AssertionError, "n_global_crops"):
            pretrain._validate_crop_counts(
                SimpleNamespace(
                    n_global_crops=0,
                    n_local_crops=1,
                )
            )

        with self.assertRaisesRegex(AssertionError, "n_local_crops"):
            pretrain._validate_crop_counts(
                SimpleNamespace(
                    n_global_crops=1,
                    n_local_crops=0,
                )
            )

    def test_compute_weighted_loss_total_respects_config_weights(self) -> None:
        config = SimpleNamespace(
            dino_loss_weight=1.5,
            koleo_loss_weight=0.25,
            ibot_loss_weight=2.0,
        )
        reduced_losses = {
            "dino_global_cls_loss": 2.0,
            "dino_local_cls_loss": 4.0,
            "koleo_loss": 8.0,
            "ibot_patch_loss": 16.0,
        }

        actual = pretrain._compute_weighted_loss_total(reduced_losses, config)

        expected = 1.5 * (2.0 + 4.0) + 0.25 * 8.0 + 2.0 * 16.0
        self.assertEqual(actual, expected)

    def test_compute_weighted_loss_total_handles_missing_terms(self) -> None:
        config = SimpleNamespace(
            dino_loss_weight=1.0,
            koleo_loss_weight=0.1,
            ibot_loss_weight=1.0,
        )

        actual = pretrain._compute_weighted_loss_total({"koleo_loss": 5.0}, config)

        self.assertEqual(actual, 0.5)

    def test_mask_generator_accepts_random_alias(self) -> None:
        mask = mask_generator(high=3, grid_size=(2, 2, 2), mask_type="random")

        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.shape, (2, 2, 2))
        self.assertEqual(int(mask.sum().item()), 3)

    def test_optimizer_update_timing_is_independent_of_ddp_sync(self) -> None:
        self.assertFalse(
            pretrain._should_update_optimizer_step(
                accum_iter=2,
                micro_steps_in_step=0,
            )
        )
        self.assertTrue(
            pretrain._should_update_optimizer_step(
                accum_iter=2,
                micro_steps_in_step=1,
            )
        )

    def test_wrap_model_for_training_returns_plain_model_without_distributed(self) -> None:
        model = object()

        wrapped = pretrain._wrap_model_for_training(
            model=model,
            distributed=False,
            local_rank=0,
            find_unused_parameters=False,
        )

        self.assertIs(wrapped, model)

    def test_wrap_model_for_training_uses_ddp_when_distributed(self) -> None:
        model = object()
        sentinel = object()

        with patch.object(pretrain, "DDP", return_value=sentinel) as ddp:
            wrapped = pretrain._wrap_model_for_training(
                model=model,
                distributed=True,
                local_rank=3,
                find_unused_parameters=True,
            )

        self.assertIs(wrapped, sentinel)
        ddp.assert_called_once_with(
            model,
            device_ids=[3],
            find_unused_parameters=True,
        )
