from __future__ import annotations

import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from spatialdino.config import parse_config
from spatialdino.data.collate import collate_fn_train
from spatialdino.models.utils import build_ssl_model, load_model
from spatialdino.optim.lr_decay import get_params_groups_with_decay


def _load_pretrain_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train" / "pretrain.py"
    spec = importlib.util.spec_from_file_location("pretrain_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pretrain module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRAIN_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "spatialdino" / "config" / "pretrain.yaml"
)
pretrain = _load_pretrain_module()


def _make_tiny_pretrain_config(output_dir: Path):
    with patch.object(sys, "argv", [sys.argv[0]]):
        config = parse_config(PRETRAIN_CONFIG_PATH)

    config.output_dir = str(output_dir)
    config.batch_size = 2
    config.accum_iter = 2
    config.max_steps = 1
    config.log_interval = 1
    config.save_interval = 1
    config.num_workers = 0
    config.persistent_workers = False
    config.pin_mem = False
    config.distributed = False
    config.find_unused_parameters = False
    config.device = "cuda"
    config.device_type = "cuda"
    config.dtype = "fp32"
    config.use_amp = False
    config.lr = 1.0e-3
    config.min_lr = 1.0e-6
    config.warmup_steps = 0
    config.freeze_last_layer_steps = 0
    config.warmup_teacher_temp_steps = 2
    config.warmup_teacher_temp = 1.0
    config.teacher_temp = 1.0
    config.student_temp = 1.0
    config.momentum_teacher = 0.9
    config.final_momentum_teacher = 1.0
    config.mask_type = "random"
    config.min_mask_ratio = 0.25
    config.max_mask_ratio = 0.25
    config.mask_sample_probability = 1.0
    config.global_crop_size = [8, 8, 8]
    config.local_crop_size = [8, 8, 8]
    config.patch_size = [4, 4, 4]
    config.stride = [4, 4, 4]
    config.embed_dim = 16
    config.depth = 1
    config.num_heads = 4
    config.mlp_ratio = 2.0
    config.drop_path_rate = 0.0
    config.drop_path_uniform = False
    config.layerscale = None
    config.interpolate_offset = 0.1
    config.interpolate_antialias = True
    config.interpolate_align_corners = True
    config.pos_embed_type = "none"
    config.n_global_crops = 2
    config.n_local_crops = 1
    config.n_prototypes = 8
    config.dino_loss_weight = 1.0
    config.ibot_loss_weight = 1.0
    config.koleo_loss_weight = 0.1
    config.centering = "sinkhorn_knopp"
    return config


def _make_smoke_batch(config, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    samples = []
    for _ in range(config.batch_size):
        samples.append(
            {
                "global_crops": torch.randn(
                    config.n_global_crops,
                    config.in_chans,
                    *config.global_crop_size,
                ),
                "local_crops": torch.randn(
                    config.n_local_crops,
                    config.in_chans,
                    *config.local_crop_size,
                ),
            }
        )

    grid_size = tuple(
        global_size // patch_size
        for global_size, patch_size in zip(config.global_crop_size, config.patch_size)
    )
    patch_size = tuple(config.patch_size)

    return collate_fn_train(
        samples_list=samples,
        grid_size=grid_size,
        patch_size=patch_size,
        min_mask_ratio=config.min_mask_ratio,
        max_mask_ratio=config.max_mask_ratio,
        mask_sample_probability=config.mask_sample_probability,
        mask_type=config.mask_type,
    )


class PretrainSmokeTests(unittest.TestCase):
    def test_ssl_layerwise_lr_decay_uses_student_encoder_depth(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config = _make_tiny_pretrain_config(Path(tmpdir))
            config.depth = 3
            config.layerwise_decay = 0.5
            config.patch_embed_lr_mult = 0.2

            model = build_ssl_model(config)
            param_groups = get_params_groups_with_decay(
                model=model,
                lr_decay_rate=config.layerwise_decay,
                patch_embed_lr_mult=config.patch_embed_lr_mult,
            )

        groups_by_name = {group["name"]: group for group in param_groups}
        block0 = groups_by_name["student.encoder.blocks.0.attn.qkv.weight"]
        block2 = groups_by_name["student.encoder.blocks.2.attn.qkv.weight"]
        patch_embed = groups_by_name["student.encoder.patch_embed.proj.weight"]
        dino_head_name = next(
            name for name in groups_by_name if name.startswith("student.dino_head")
        )
        ibot_head_name = next(
            name for name in groups_by_name if name.startswith("student.ibot_head")
        )

        self.assertLess(block0["lr_multiplier"], block2["lr_multiplier"])
        self.assertEqual(block0["lr_multiplier"], 0.125)
        self.assertEqual(block2["lr_multiplier"], 0.5)
        self.assertEqual(patch_embed["lr_multiplier"], 0.0125)
        self.assertEqual(groups_by_name[dino_head_name]["lr_multiplier"], 1.0)
        self.assertEqual(groups_by_name[ibot_head_name]["lr_multiplier"], 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "Pretrain smoke test requires CUDA.")
    def test_train_returns_empty_metrics_when_no_optimizer_steps_run(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = _make_tiny_pretrain_config(output_dir)
            config.max_steps = 0

            torch.cuda.set_device(0)
            model = build_ssl_model(config).cuda()
            optimizer = torch.optim.AdamW(
                get_params_groups_with_decay(
                    model=model,
                    lr_decay_rate=config.layerwise_decay,
                    patch_embed_lr_mult=config.patch_embed_lr_mult,
                ),
                lr=config.lr,
                betas=tuple(config.betas),
            )

            metrics = pretrain.train(
                config=config,
                step=0,
                model=model,
                train_model=model,
                optimizer=optimizer,
                train_dataloader=[],
                rank=0,
                world_size=1,
                loss_scaler=None,
                run=None,
            )

        self.assertEqual(metrics, {})

    @unittest.skipUnless(torch.cuda.is_available(), "Pretrain smoke test requires CUDA.")
    def test_non_distributed_pretrain_smoke_covers_resume_and_single_process_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = _make_tiny_pretrain_config(output_dir)

            torch.cuda.set_device(0)
            model = build_ssl_model(config).cuda()
            train_model = pretrain._wrap_model_for_training(
                model=model,
                distributed=False,
                local_rank=0,
                find_unused_parameters=config.find_unused_parameters,
            )

            self.assertIs(train_model, model)

            optimizer = torch.optim.AdamW(
                get_params_groups_with_decay(
                    model=model,
                    lr_decay_rate=config.layerwise_decay,
                    patch_embed_lr_mult=config.patch_embed_lr_mult,
                ),
                lr=config.lr,
                betas=tuple(config.betas),
            )

            train_batches = [
                _make_smoke_batch(config, seed=0),
                _make_smoke_batch(config, seed=1),
            ]

            metrics = pretrain.train(
                config=config,
                step=0,
                model=model,
                train_model=train_model,
                optimizer=optimizer,
                train_dataloader=train_batches,
                rank=0,
                world_size=1,
                loss_scaler=None,
                run=None,
            )

            self.assertIn("loss", metrics)
            self.assertTrue(math.isfinite(metrics["loss"]))

            checkpoint_path = output_dir / "checkpoints" / "step=1" / "ckpt.pth"
            self.assertTrue(checkpoint_path.exists())

            resumed_model = build_ssl_model(config)
            resumed_optimizer = torch.optim.AdamW(
                get_params_groups_with_decay(
                    model=resumed_model,
                    lr_decay_rate=config.layerwise_decay,
                    patch_embed_lr_mult=config.patch_embed_lr_mult,
                ),
                lr=config.lr,
                betas=tuple(config.betas),
            )

            resumed_step = load_model(
                checkpoint_path=str(checkpoint_path),
                model=resumed_model,
                optimizer=resumed_optimizer,
            )

            self.assertEqual(resumed_step, 1)

            original_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            resumed_state = resumed_model.state_dict()
            self.assertSetEqual(set(original_state.keys()), set(resumed_state.keys()))
            for key, value in original_state.items():
                with self.subTest(key=key):
                    self.assertTrue(torch.equal(value, resumed_state[key]))

    @unittest.skipUnless(torch.cuda.is_available(), "Pretrain smoke test requires CUDA.")
    def test_koleo_handles_singleton_per_view_batches_without_squeezing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = _make_tiny_pretrain_config(output_dir)
            config.batch_size = 1

            torch.cuda.set_device(0)
            model = build_ssl_model(config).cuda()

            optimizer = torch.optim.AdamW(
                get_params_groups_with_decay(
                    model=model,
                    lr_decay_rate=config.layerwise_decay,
                    patch_embed_lr_mult=config.patch_embed_lr_mult,
                ),
                lr=config.lr,
                betas=tuple(config.betas),
            )

            train_batches = [
                _make_smoke_batch(config, seed=0),
                _make_smoke_batch(config, seed=1),
            ]

            metrics = pretrain.train(
                config=config,
                step=0,
                model=model,
                train_model=model,
                optimizer=optimizer,
                train_dataloader=train_batches,
                rank=0,
                world_size=1,
                loss_scaler=None,
                run=None,
            )

        self.assertIn("loss", metrics)
        self.assertTrue(math.isfinite(metrics["loss"]))
