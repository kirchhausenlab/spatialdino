"""Tests for learnable register tokens (DINOv2-reg).

Verifies that register tokens work correctly across all pathways:
- Training forward (single tensor and list/xFormers nested)
- Inference (_predict)
- Output shapes (registers stripped from patch tokens)
- Gradient flow through register tokens
- Compatibility with RoPE, sincos, and no positional embedding
- SSL model integration (student + teacher)
"""

from __future__ import annotations

import unittest
from functools import partial
from typing import List

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from spatialdino.models.layers.attention import Attention, MemEffAttention
from spatialdino.models.layers.block import Block, NestedTensorBlock
from spatialdino.models.layers.encoder import Encoder
from spatialdino.models.ssl import SSL
from spatialdino.models.utils import init_backbone

IMG_SIZE = (16, 16, 16)
PATCH_SIZE = (4, 4, 4)
EMBED_DIM = 32
NUM_HEADS = 4
DEPTH = 2
NUM_REGISTER_TOKENS = 4

# Grid size: 16/4 = 4 patches per dim → 64 total patches
GRID = tuple(s // p for s, p in zip(IMG_SIZE, PATCH_SIZE))
NUM_PATCHES = GRID[0] * GRID[1] * GRID[2]


def _make_encoder(
    num_register_tokens: int = NUM_REGISTER_TOKENS,
    num_tt_register_tokens: int = 0,
    pos_embed_type: str = "none",
    use_mem_eff: bool = False,
) -> Encoder:
    if use_mem_eff:
        block_fn = partial(NestedTensorBlock, attn_class=MemEffAttention)
    else:
        block_fn = partial(Block, attn_class=Attention)
    return Encoder(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=1,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=2.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        ffn_layer="mlp",
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        num_register_tokens=num_register_tokens,
        num_tt_register_tokens=num_tt_register_tokens,
        interpolate_offset=0.1,
        interpolate_antialias=True,
        interpolate_align_corners=True,
        pos_embed_type=pos_embed_type,
        block_fn=block_fn,
    )


def _make_ssl_config(num_register_tokens: int = NUM_REGISTER_TOKENS):
    return OmegaConf.create(
        {
            "global_crop_size": list(IMG_SIZE),
            "patch_size": list(PATCH_SIZE),
            "in_chans": 1,
            "embed_dim": EMBED_DIM,
            "depth": DEPTH,
            "num_heads": NUM_HEADS,
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
            "num_register_tokens": num_register_tokens,
        }
    )


class TestRegisterTokensInit(unittest.TestCase):
    """Test register token initialization."""

    def test_register_tokens_parameter_exists(self):
        enc = _make_encoder(num_register_tokens=4)
        self.assertIsNotNone(enc.register_tokens)
        self.assertEqual(enc.register_tokens.shape, (1, 4, EMBED_DIM))
        self.assertTrue(enc.register_tokens.requires_grad)

    def test_register_tokens_zero_means_none(self):
        enc = _make_encoder(num_register_tokens=0)
        self.assertIsNone(enc.register_tokens)
        self.assertEqual(enc.num_register_tokens, 0)

    def test_register_tokens_initialized_nonzero(self):
        enc = _make_encoder(num_register_tokens=4)
        # After init, register_tokens should have been initialized with normal_ std=1e-6
        # They should be very small but not all zero
        self.assertGreater(enc.register_tokens.abs().max().item(), 0.0)


class TestRegisterTokensForwardSingleTensor(unittest.TestCase):
    """Test register tokens in the single-tensor forward path."""

    def test_output_shapes_with_registers(self):
        enc = _make_encoder(num_register_tokens=4)
        x = torch.randn(2, 1, *IMG_SIZE)
        out = enc(x)

        self.assertEqual(out["x_norm_clstoken"].shape, (2, EMBED_DIM))
        self.assertEqual(out["x_norm_patchtokens"].shape, (2, NUM_PATCHES, EMBED_DIM))
        self.assertEqual(out["x_norm_regtokens"].shape, (2, 4, EMBED_DIM))
        self.assertEqual(out["x_prenorm_clstoken"].shape, (2, EMBED_DIM))
        self.assertEqual(out["x_prenorm_patchtokens"].shape, (2, NUM_PATCHES, EMBED_DIM))
        self.assertEqual(out["x_prenorm_regtokens"].shape, (2, 4, EMBED_DIM))

    def test_output_shapes_without_registers(self):
        enc = _make_encoder(num_register_tokens=0)
        x = torch.randn(2, 1, *IMG_SIZE)
        out = enc(x)

        self.assertEqual(out["x_norm_clstoken"].shape, (2, EMBED_DIM))
        self.assertEqual(out["x_norm_patchtokens"].shape, (2, NUM_PATCHES, EMBED_DIM))
        # regtokens should be empty (0 tokens)
        self.assertEqual(out["x_norm_regtokens"].shape, (2, 0, EMBED_DIM))

    def test_patch_tokens_not_polluted_by_registers(self):
        """Verify that the number of patch tokens is independent of register count."""
        enc_0 = _make_encoder(num_register_tokens=0)
        enc_4 = _make_encoder(num_register_tokens=4)
        x = torch.randn(1, 1, *IMG_SIZE)

        out_0 = enc_0(x)
        out_4 = enc_4(x)

        self.assertEqual(
            out_0["x_norm_patchtokens"].shape[1],
            out_4["x_norm_patchtokens"].shape[1],
        )


class TestRegisterTokensForwardList(unittest.TestCase):
    """Test register tokens in the list forward path (nested tensor / xFormers)."""

    @unittest.skipUnless(torch.cuda.is_available(), "xFormers nested path requires CUDA")
    def test_output_shapes_list_forward(self):
        # List forward requires NestedTensorBlock + MemEffAttention (xFormers) on CUDA
        enc = _make_encoder(num_register_tokens=4, use_mem_eff=True).cuda()
        # Simulate 2 crops of different sizes
        x_global = torch.randn(2, 1, *IMG_SIZE, device="cuda")
        x_local = torch.randn(2, 1, 8, 8, 8, device="cuda")
        masks_global = torch.zeros(2, NUM_PATCHES, dtype=torch.bool, device="cuda")
        masks_local = torch.zeros(2, 8, dtype=torch.bool, device="cuda")  # (8/4)^3 = 8

        out_list = enc([x_global, x_local], [masks_global, masks_local])

        self.assertEqual(len(out_list), 2)

        # Global crop output
        self.assertEqual(out_list[0]["x_norm_clstoken"].shape, (2, EMBED_DIM))
        self.assertEqual(out_list[0]["x_norm_patchtokens"].shape, (2, NUM_PATCHES, EMBED_DIM))
        self.assertEqual(out_list[0]["x_norm_regtokens"].shape, (2, 4, EMBED_DIM))

        # Local crop output (8/4=2 patches per dim → 8 patches)
        self.assertEqual(out_list[1]["x_norm_clstoken"].shape, (2, EMBED_DIM))
        self.assertEqual(out_list[1]["x_norm_patchtokens"].shape, (2, 8, EMBED_DIM))
        self.assertEqual(out_list[1]["x_norm_regtokens"].shape, (2, 4, EMBED_DIM))


class TestRegisterTokensPredict(unittest.TestCase):
    """Test register tokens in the _predict inference path."""

    def test_predict_strips_registers(self):
        enc = _make_encoder(num_register_tokens=4)
        enc.eval()
        x = torch.randn(1, 1, *IMG_SIZE)
        with torch.no_grad():
            out = enc._predict(x, vit_feat="patch")
        self.assertEqual(out.shape, (1, EMBED_DIM, *GRID))

    def test_predict_with_both_register_types(self):
        """Register tokens + test-time register tokens together."""
        enc = _make_encoder(num_register_tokens=4, num_tt_register_tokens=2)
        enc.eval()
        x = torch.randn(1, 1, *IMG_SIZE)
        with torch.no_grad():
            out = enc._predict(x, vit_feat="patch")
        # Output should still be just patch tokens in spatial layout
        self.assertEqual(out.shape, (1, EMBED_DIM, *GRID))

    def test_predict_patch_attn_with_registers(self):
        enc = _make_encoder(num_register_tokens=4)
        enc.eval()
        x = torch.randn(1, 1, *IMG_SIZE)
        with torch.no_grad():
            out = enc._predict(x, vit_feat="patch_attn")
        # patch_attn adds num_heads channels
        self.assertEqual(out.shape, (1, EMBED_DIM + NUM_HEADS, *GRID))


class TestRegisterTokensRoPE(unittest.TestCase):
    """Test register tokens with RoPE positional encoding."""

    def test_forward_with_rope_and_registers(self):
        enc = _make_encoder(num_register_tokens=4, pos_embed_type="rope")
        x = torch.randn(1, 1, *IMG_SIZE)
        out = enc(x)

        self.assertEqual(out["x_norm_clstoken"].shape, (1, EMBED_DIM))
        self.assertEqual(out["x_norm_patchtokens"].shape, (1, NUM_PATCHES, EMBED_DIM))
        self.assertEqual(out["x_norm_regtokens"].shape, (1, 4, EMBED_DIM))

    def test_predict_with_rope_and_registers(self):
        enc = _make_encoder(num_register_tokens=4, pos_embed_type="rope")
        enc.eval()
        x = torch.randn(1, 1, *IMG_SIZE)
        with torch.no_grad():
            out = enc._predict(x, vit_feat="patch")
        self.assertEqual(out.shape, (1, EMBED_DIM, *GRID))

    def test_predict_with_rope_and_both_register_types(self):
        enc = _make_encoder(
            num_register_tokens=4,
            num_tt_register_tokens=2,
            pos_embed_type="rope",
        )
        enc.eval()
        x = torch.randn(1, 1, *IMG_SIZE)
        with torch.no_grad():
            out = enc._predict(x, vit_feat="patch")
        self.assertEqual(out.shape, (1, EMBED_DIM, *GRID))


class TestRegisterTokensSincos(unittest.TestCase):
    """Test register tokens with sincos positional encoding."""

    def test_forward_with_sincos_and_registers(self):
        enc = _make_encoder(num_register_tokens=4, pos_embed_type="sincos")
        x = torch.randn(1, 1, *IMG_SIZE)
        out = enc(x)

        self.assertEqual(out["x_norm_clstoken"].shape, (1, EMBED_DIM))
        self.assertEqual(out["x_norm_patchtokens"].shape, (1, NUM_PATCHES, EMBED_DIM))
        self.assertEqual(out["x_norm_regtokens"].shape, (1, 4, EMBED_DIM))


class TestRegisterTokensGradients(unittest.TestCase):
    """Test that gradients flow through register tokens."""

    def test_register_tokens_receive_gradients(self):
        enc = _make_encoder(num_register_tokens=4)
        enc.train()
        x = torch.randn(1, 1, *IMG_SIZE)
        out = enc(x)
        loss = out["x_norm_clstoken"].sum() + out["x_norm_patchtokens"].sum()
        loss.backward()

        self.assertIsNotNone(enc.register_tokens.grad)
        self.assertGreater(enc.register_tokens.grad.abs().sum().item(), 0.0)

    def test_register_tokens_receive_gradients_with_rope(self):
        enc = _make_encoder(num_register_tokens=4, pos_embed_type="rope")
        enc.train()
        x = torch.randn(1, 1, *IMG_SIZE)
        out = enc(x)
        loss = out["x_norm_clstoken"].sum() + out["x_norm_patchtokens"].sum()
        loss.backward()

        self.assertIsNotNone(enc.register_tokens.grad)
        self.assertGreater(enc.register_tokens.grad.abs().sum().item(), 0.0)


class TestRegisterTokensSSL(unittest.TestCase):
    """Test register tokens in the full SSL (student-teacher) model."""

    def test_ssl_model_with_register_tokens_has_matching_keys(self):
        model = SSL(_make_ssl_config(num_register_tokens=4))

        student_state = model.student.state_dict()
        teacher_state = model.teacher.state_dict()

        self.assertIn("encoder.register_tokens", student_state)
        self.assertIn("encoder.register_tokens", teacher_state)

        # Teacher should mirror student at init
        self.assertTrue(
            torch.equal(
                student_state["encoder.register_tokens"],
                teacher_state["encoder.register_tokens"],
            )
        )

    def test_ssl_model_without_register_tokens_has_no_register_key(self):
        model = SSL(_make_ssl_config(num_register_tokens=0))
        student_state = model.student.state_dict()
        self.assertNotIn("encoder.register_tokens", student_state)

    def test_ssl_teacher_register_tokens_are_frozen(self):
        model = SSL(_make_ssl_config(num_register_tokens=4))
        for name, param in model.teacher.named_parameters():
            if "register_tokens" in name:
                self.assertFalse(param.requires_grad)


class TestRegisterTokensInitBackbone(unittest.TestCase):
    """Test register tokens via init_backbone."""

    def test_init_backbone_with_register_tokens(self):
        config = OmegaConf.create(
            {
                "global_crop_size": list(IMG_SIZE),
                "patch_size": list(PATCH_SIZE),
                "stride": list(PATCH_SIZE),
                "in_chans": 1,
                "embed_dim": EMBED_DIM,
                "depth": DEPTH,
                "num_heads": NUM_HEADS,
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
                "num_register_tokens": 4,
            }
        )
        model = init_backbone(config)
        self.assertEqual(model.num_register_tokens, 4)
        self.assertIsNotNone(model.register_tokens)
        self.assertEqual(model.register_tokens.shape, (1, 4, EMBED_DIM))

    def test_init_backbone_defaults_register_tokens_to_zero(self):
        config = OmegaConf.create(
            {
                "global_crop_size": list(IMG_SIZE),
                "patch_size": list(PATCH_SIZE),
                "stride": list(PATCH_SIZE),
                "in_chans": 1,
                "embed_dim": EMBED_DIM,
                "depth": DEPTH,
                "num_heads": NUM_HEADS,
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
                # num_register_tokens intentionally omitted
            }
        )
        model = init_backbone(config)
        self.assertEqual(model.num_register_tokens, 0)
        self.assertIsNone(model.register_tokens)


class TestPrepareTokensWithRegisters(unittest.TestCase):
    """Test that prepare_tokens_with_masks inserts registers correctly."""

    def test_token_sequence_layout(self):
        enc = _make_encoder(num_register_tokens=4, pos_embed_type="none")
        x = torch.randn(2, 1, *IMG_SIZE)
        tokens = enc.prepare_tokens_with_masks(x)

        # Expected: [CLS, reg*4, patches*64] = 1 + 4 + 64 = 69
        expected_seq_len = 1 + 4 + NUM_PATCHES
        self.assertEqual(tokens.shape, (2, expected_seq_len, EMBED_DIM))

    def test_token_sequence_without_registers(self):
        enc = _make_encoder(num_register_tokens=0, pos_embed_type="none")
        x = torch.randn(2, 1, *IMG_SIZE)
        tokens = enc.prepare_tokens_with_masks(x)

        expected_seq_len = 1 + NUM_PATCHES
        self.assertEqual(tokens.shape, (2, expected_seq_len, EMBED_DIM))

    def test_mask_application_with_registers(self):
        """Masking applies to patches, not to CLS or registers."""
        enc = _make_encoder(num_register_tokens=4, pos_embed_type="none")
        x = torch.randn(1, 1, *IMG_SIZE)
        mask = torch.ones(1, NUM_PATCHES, dtype=torch.bool)  # mask all patches

        tokens = enc.prepare_tokens_with_masks(x, masks=mask)

        # CLS is at 0, registers at 1:5, patches at 5:
        # All patches should be mask_token (since mask=True for all)
        patch_tokens = tokens[:, 5:]
        mask_value = enc.mask_token.detach()
        self.assertTrue(
            torch.allclose(patch_tokens, mask_value.expand_as(patch_tokens), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
