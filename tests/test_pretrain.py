from __future__ import annotations

import importlib.util
import math
import os
import random
import socket
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as tdist
import torch.multiprocessing as mp

from spatialdino.config import parse_config
from spatialdino.data.collate import collate_fn_train, mask_generator
from spatialdino.models.utils import build_ssl_model, load_model
from spatialdino.optim.lr_decay import get_params_groups_with_decay


def _load_pretrain_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "train" / "pretrain.py"
    )
    spec = importlib.util.spec_from_file_location("pretrain_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pretrain module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRETRAIN_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "spatialdino"
    / "config"
    / "pretrain.yaml"
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
    config.dtype = "bf16"
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
    config.pos_embed_type = "rope"
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

    @unittest.skipUnless(
        torch.cuda.is_available(), "Pretrain smoke test requires CUDA."
    )
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
                train_model=model,
                optimizer=optimizer,
                train_dataloader=[],
                rank=0,
                world_size=1,
                loss_scaler=None,
                run=None,
            )

        self.assertEqual(metrics, {})

    @unittest.skipUnless(
        torch.cuda.is_available(), "Pretrain smoke test requires CUDA."
    )
    def test_non_distributed_pretrain_smoke_covers_resume_and_single_process_path(
        self,
    ) -> None:
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

            original_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            resumed_state = resumed_model.state_dict()
            self.assertSetEqual(set(original_state.keys()), set(resumed_state.keys()))
            for key, value in original_state.items():
                with self.subTest(key=key):
                    self.assertTrue(torch.equal(value, resumed_state[key]))

    @unittest.skipUnless(
        torch.cuda.is_available(), "Pretrain smoke test requires CUDA."
    )
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


class PretrainUnitTests(unittest.TestCase):
    """Focused unit tests for individual pretrain helper functions."""

    def test_validate_crop_counts_requires_positive_global_and_local_crops(
        self,
    ) -> None:
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

    def test_wrap_model_for_training_returns_plain_model_without_distributed(
        self,
    ) -> None:
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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _distributed_rope_register_worker(
    rank: int,
    world_size: int,
    output_dir: str,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    tdist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    try:
        config = _make_tiny_pretrain_config(Path(output_dir))
        config.pos_embed_type = "rope"
        config.rope_theta = 100.0
        config.rope_normalize_coords = True
        config.num_register_tokens = 4
        config.distributed = True

        model = build_ssl_model(config).cuda(rank)
        train_model = pretrain._wrap_model_for_training(
            model=model,
            distributed=True,
            local_rank=rank,
            find_unused_parameters=config.find_unused_parameters,
        )

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
            train_model=train_model,
            optimizer=optimizer,
            train_dataloader=train_batches,
            rank=rank,
            world_size=world_size,
            loss_scaler=None,
            run=None,
        )

        assert metrics, f"[rank {rank}] Expected non-empty metrics"
        assert math.isfinite(
            metrics["loss"]
        ), f"[rank {rank}] Loss not finite: {metrics['loss']}"
    finally:
        tdist.destroy_process_group()


class DistributedPretrainSmokeTests(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "Requires 2 CUDA GPUs.",
    )
    def test_distributed_pretrain_with_rope_and_register_tokens(self) -> None:
        """2-GPU DDP smoke test with RoPE position embedding and register tokens."""
        port = _find_free_port()
        with TemporaryDirectory() as tmpdir:
            mp.spawn(
                _distributed_rope_register_worker,
                args=(2, tmpdir, port),
                nprocs=2,
                join=True,
            )
