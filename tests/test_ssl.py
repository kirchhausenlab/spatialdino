from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from spatialdino.loss.dino_clstoken_loss import DINOLoss, split_sample_major_batch
from spatialdino.loss.ibot_patch_loss import iBOTPatchLoss
from spatialdino.models.layers.encoder import Encoder
from spatialdino.models.ssl import SSL
from spatialdino.models.ssl.utils import save_model
from spatialdino.models.utils import init_backbone, load_model


def make_ssl_config():
    return OmegaConf.create(
        {
            "global_crop_size": [8, 8, 8],
            "patch_size": [4, 4, 4],
            "in_chans": 1,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "qkv_bias": True,
            "proj_bias": True,
            "ffn_bias": True,
            "ffn_layer": "mlp",
            "drop_path_rate": 0.0,
            "drop_path_uniform": False,
            "layerscale": None,
            "interpolate_offset": 0.1,
            "dino_loss_weight": 1.0,
            "ibot_loss_weight": 1.0,
            "n_prototypes": 8,
        }
    )


def make_backbone_config():
    return OmegaConf.create(
        {
            "global_crop_size": [8, 8, 8],
            "patch_size": [4, 4, 4],
            "stride": [4, 4, 4],
            "in_chans": 1,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "qkv_bias": True,
            "proj_bias": True,
            "ffn_bias": True,
            "ffn_layer": "mlp",
            "drop_path_rate": 0.0,
            "drop_path_uniform": False,
            "layerscale": None,
            "interpolate_offset": 0.1,
            "interpolate_antialias": True,
            "interpolate_align_corners": True,
            "pos_embed_type": "none",
            "backbone_path": None,
        }
    )


class SSLTests(unittest.TestCase):
    def test_init_backbone_defaults_test_time_registers_to_zero(self) -> None:
        model = init_backbone(make_backbone_config())

        self.assertEqual(model.num_tt_register_tokens, 0)

    def test_encoder_forward_rejects_test_time_registers(self) -> None:
        model = Encoder(
            img_size=(8, 8, 8),
            patch_size=(4, 4, 4),
            stride=(4, 4, 4),
            in_chans=1,
            embed_dim=16,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            ffn_layer="mlp",
            drop_path_rate=0.0,
            drop_path_uniform=False,
            init_values=None,
            num_tt_register_tokens=1,
            interpolate_offset=0.1,
            interpolate_antialias=True,
            interpolate_align_corners=True,
            pos_embed_type="none",
        )

        with self.assertRaisesRegex(ValueError, "Test-time register tokens"):
            model(torch.randn(1, 1, 8, 8, 8))

    def test_split_sample_major_batch_groups_views_across_samples(self) -> None:
        sample_major_tokens = torch.tensor([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ])

        view_major_tokens = split_sample_major_batch(sample_major_tokens, n_views=2)

        self.assertEqual(len(view_major_tokens), 2)
        self.assertTrue(torch.equal(view_major_tokens[0], torch.tensor([[0.0], [2.0], [4.0]])))
        self.assertTrue(torch.equal(view_major_tokens[1], torch.tensor([[1.0], [3.0], [5.0]])))

    def test_split_sample_major_batch_supports_cross_view_global_pairing(self) -> None:
        loss_fn = DINOLoss(out_dim=2, student_temp=1.0)
        student_outputs = torch.tensor([
            [8.0, 0.0],
            [0.0, 8.0],
            [8.0, 0.0],
            [0.0, 8.0],
        ])
        teacher_outputs = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        student_views = split_sample_major_batch(student_outputs, n_views=2)
        teacher_views = split_sample_major_batch(teacher_outputs, n_views=2)

        self_pair_loss = sum(
            loss_fn([student_views[i]], (teacher_views[i],))
            for i in range(2)
        )
        cross_view_loss = sum(
            loss_fn([student_views[i]], teacher_views[:i] + teacher_views[i + 1 :])
            for i in range(2)
        )

        self.assertLess(self_pair_loss.item(), cross_view_loss.item())

    def test_dino_sinkhorn_uses_global_batch_size_when_distributed(self) -> None:
        loss_fn = DINOLoss(out_dim=2, student_temp=1.0)
        teacher_output = torch.tensor([[0.2, 1.1], [1.7, -0.3]], dtype=torch.float32)

        def _double_in_place(tensor: torch.Tensor, async_op: bool = False):
            tensor.mul_(2)
            return None

        with patch(
            "spatialdino.loss.dino_clstoken_loss.dist.is_available",
            return_value=True,
        ), patch(
            "spatialdino.loss.dino_clstoken_loss.dist.is_initialized",
            return_value=True,
        ), patch(
            "spatialdino.loss.dino_clstoken_loss.dist.all_reduce",
            side_effect=_double_in_place,
        ):
            actual = loss_fn.sinkhorn_knopp_teacher(
                teacher_output,
                teacher_temp=1.0,
                n_iterations=2,
            )

        expected = teacher_output.float()
        Q = torch.exp(expected).t()
        K = Q.shape[0]
        B = Q.new_tensor(float(Q.shape[1] * 2))
        Q /= torch.sum(Q) * 2

        for _ in range(2):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True) * 2
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        expected = (Q * B).t()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_ibot_sinkhorn_skips_collectives_without_initialized_process_group(self) -> None:
        loss_fn = iBOTPatchLoss(patch_out_dim=2, student_temp=1.0)
        teacher_output = torch.tensor([[0.2, 1.1], [1.7, -0.3]], dtype=torch.float32)
        n_masked_patches = torch.tensor([teacher_output.shape[0]], dtype=torch.long)

        with patch(
            "spatialdino.loss.ibot_patch_loss.dist.is_available",
            return_value=True,
        ), patch(
            "spatialdino.loss.ibot_patch_loss.dist.is_initialized",
            return_value=False,
        ), patch(
            "spatialdino.loss.ibot_patch_loss.dist.all_reduce",
            side_effect=AssertionError("all_reduce should not be called"),
        ):
            output = loss_fn.sinkhorn_knopp_teacher(
                teacher_output,
                teacher_temp=1.0,
                n_masked_patches_tensor=n_masked_patches,
                n_iterations=1,
            )

        self.assertEqual(output.shape, teacher_output.shape)

    def test_teacher_starts_from_student_weights_and_is_frozen(self) -> None:
        model = SSL(make_ssl_config())

        student_state = model.student.state_dict()
        teacher_state = model.teacher.state_dict()

        self.assertListEqual(list(student_state.keys()), list(teacher_state.keys()))

        for key in student_state:
            with self.subTest(key=key):
                self.assertTrue(torch.equal(student_state[key], teacher_state[key]))

        self.assertTrue(all(not param.requires_grad for param in model.teacher.parameters()))

    def test_pretrain_checkpoint_round_trip_uses_completed_optimizer_steps(self) -> None:
        model = SSL(make_ssl_config())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_model(
                output_dir=output_dir,
                step=5,
                model=model,
                optimizer=optimizer,
            )

            resumed_model = SSL(make_ssl_config())
            resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
            resumed_step = load_model(
                checkpoint_path=str(output_dir / "step=5" / "ckpt.pth"),
                model=resumed_model,
                optimizer=resumed_optimizer,
            )

        self.assertEqual(resumed_step, 5)

    @unittest.skipUnless(torch.cuda.is_available(), "GradScaler state requires CUDA.")
    def test_checkpoint_with_scaler_can_resume_without_amp(self) -> None:
        model = SSL(make_ssl_config()).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(device="cuda")

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_model(
                output_dir=output_dir,
                step=5,
                model=model,
                optimizer=optimizer,
                loss_scaler=scaler,
            )

            resumed_model = SSL(make_ssl_config())
            resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
            resumed_step = load_model(
                checkpoint_path=str(output_dir / "step=5" / "ckpt.pth"),
                model=resumed_model,
                optimizer=resumed_optimizer,
                loss_scaler=None,
            )

        self.assertEqual(resumed_step, 5)

    @unittest.skipUnless(torch.cuda.is_available(), "GradScaler state requires CUDA.")
    def test_checkpoint_without_scaler_can_resume_with_amp(self) -> None:
        model = SSL(make_ssl_config())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_model(
                output_dir=output_dir,
                step=5,
                model=model,
                optimizer=optimizer,
                loss_scaler=None,
            )

            resumed_model = SSL(make_ssl_config())
            resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
            resumed_scaler = torch.amp.GradScaler(device="cuda")
            resumed_step = load_model(
                checkpoint_path=str(output_dir / "step=5" / "ckpt.pth"),
                model=resumed_model,
                optimizer=resumed_optimizer,
                loss_scaler=resumed_scaler,
            )

        self.assertEqual(resumed_step, 5)

    def test_legacy_checkpoints_still_resume_with_next_step_semantics(self) -> None:
        model = SSL(make_ssl_config())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy_ckpt.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": 5,
                },
                checkpoint_path,
            )

            resumed_model = SSL(make_ssl_config())
            resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
            resumed_step = load_model(
                checkpoint_path=str(checkpoint_path),
                model=resumed_model,
                optimizer=resumed_optimizer,
            )

        self.assertEqual(resumed_step, 6)
