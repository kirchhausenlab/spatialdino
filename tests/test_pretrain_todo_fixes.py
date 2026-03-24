from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

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
