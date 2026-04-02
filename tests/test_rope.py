"""Tests for 3D Rotary Position Embedding (RoPE) across all attention pathways.

Converted from pytest to unittest style to match the project test conventions.
"""

from __future__ import annotations

import math
import os
import unittest
from functools import partial
from unittest.mock import patch as mock_patch

import torch
import torch.nn as nn

from spatialdino.models.layers.rope import (
    RoPE3D,
    build_3d_rope_cache,
    apply_rotary_emb,
    concat_rope_for_nested,
)
from spatialdino.models.layers.attention import Attention, MemEffAttention
from spatialdino.models.layers.block import Block, NestedTensorBlock
from spatialdino.models.layers.encoder import Encoder
from spatialdino.models.layers.patch_embed import PatchEmbed

DIM = 192
NUM_HEADS = 6
HEAD_DIM = DIM // NUM_HEADS  # 32
GRID = (4, 4, 4)  # small grid for fast tests
NUM_PATCHES = GRID[0] * GRID[1] * GRID[2]  # 64

XFORMERS_AVAILABLE = os.environ.get("XFORMERS_DISABLED") is None
try:
    import xformers  # noqa: F401
except ImportError:
    XFORMERS_AVAILABLE = False

requires_xformers = unittest.skipUnless(XFORMERS_AVAILABLE, "xFormers not available")
requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")


def _make_rope_cache():
    return build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=1)


def _make_rope_cache_no_prefix():
    return build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=0)


class TestBuild3DRopeCache(unittest.TestCase):
    def setUp(self):
        self.rope_cache = _make_rope_cache()
        self.rope_cache_no_prefix = _make_rope_cache_no_prefix()

    def test_cache_returns_shape_matching_prefix_plus_patches(self):
        cos, sin = self.rope_cache
        N = 1 + NUM_PATCHES  # CLS + patches
        self.assertEqual(cos.shape, (N, HEAD_DIM // 2))
        self.assertEqual(sin.shape, (N, HEAD_DIM // 2))

    def test_prefix_token_returns_identity_rotation(self):
        """CLS (prefix) token should have cos=1, sin=0 (identity rotation)."""
        cos, sin = self.rope_cache
        torch.testing.assert_close(cos[0], torch.ones(HEAD_DIM // 2))
        torch.testing.assert_close(sin[0], torch.zeros(HEAD_DIM // 2))

    def test_cache_without_prefix_returns_patch_count_rows(self):
        cos, sin = self.rope_cache_no_prefix
        self.assertEqual(cos.shape, (NUM_PATCHES, HEAD_DIM // 2))

    def test_different_positions_return_different_embeddings(self):
        """Patches at different spatial positions should have different embeddings."""
        cos, sin = self.rope_cache
        self.assertFalse(torch.allclose(cos[1], cos[2]))

    def test_multiple_prefix_tokens_all_return_identity_rotation(self):
        """Test with CLS + register tokens."""
        num_prefix = 3
        cos, sin = build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=num_prefix)
        self.assertEqual(cos.shape[0], num_prefix + NUM_PATCHES)
        for i in range(num_prefix):
            torch.testing.assert_close(cos[i], torch.ones(HEAD_DIM // 2))
            torch.testing.assert_close(sin[i], torch.zeros(HEAD_DIM // 2))

    def test_asymmetric_grid_returns_correct_shape(self):
        """Different grid dimensions should work."""
        grid = (2, 4, 8)
        cos, sin = build_3d_rope_cache(grid, HEAD_DIM)
        N = 1 + 2 * 4 * 8
        self.assertEqual(cos.shape, (N, HEAD_DIM // 2))

    def test_cache_returns_requested_dtype(self):
        cos, sin = build_3d_rope_cache(GRID, HEAD_DIM, dtype=torch.float64)
        self.assertEqual(cos.dtype, torch.float64)


class TestNormalizedCoords(unittest.TestCase):
    """Tests for normalize_coords=True (DiNOv3-style)."""

    def test_normalized_cache_returns_correct_shape(self):
        cos, sin = build_3d_rope_cache(
            GRID, HEAD_DIM, num_prefix_tokens=1, theta=100.0, normalize_coords=True
        )
        N = 1 + NUM_PATCHES
        self.assertEqual(cos.shape, (N, HEAD_DIM // 2))
        self.assertEqual(sin.shape, (N, HEAD_DIM // 2))

    def test_normalized_prefix_returns_identity_rotation(self):
        cos, sin = build_3d_rope_cache(
            GRID, HEAD_DIM, num_prefix_tokens=2, theta=100.0, normalize_coords=True
        )
        for i in range(2):
            torch.testing.assert_close(cos[i], torch.ones(HEAD_DIM // 2))
            torch.testing.assert_close(sin[i], torch.zeros(HEAD_DIM // 2))

    def test_normalized_edge_patches_align_across_grid_sizes(self):
        """With normalization, the last patch in any grid maps to coords near +1,
        so its embedding should be similar regardless of grid size.
        """
        cos_4, _ = build_3d_rope_cache(
            (4, 4, 4), HEAD_DIM, num_prefix_tokens=0, theta=100.0, normalize_coords=True
        )
        cos_8, _ = build_3d_rope_cache(
            (8, 8, 8), HEAD_DIM, num_prefix_tokens=0, theta=100.0, normalize_coords=True
        )
        # Last patch: coords (0.875,0.875,0.875) for 4×4×4 vs (0.9375,…) for 8×8×8
        edge_diff_norm = (cos_4[-1] - cos_8[-1]).abs().max().item()

        # Without normalization, last patch is at (3,3,3) vs (7,7,7) — much bigger diff
        cos_4_raw, _ = build_3d_rope_cache((4, 4, 4), HEAD_DIM, num_prefix_tokens=0)
        cos_8_raw, _ = build_3d_rope_cache((8, 8, 8), HEAD_DIM, num_prefix_tokens=0)
        edge_diff_raw = (cos_4_raw[-1] - cos_8_raw[-1]).abs().max().item()

        # Normalized edge patches should be much more aligned than raw ones
        self.assertLess(edge_diff_norm, edge_diff_raw)

    def test_raw_positions_have_grid_dependent_embeddings(self):
        """Without normalization, the last patch in a small vs large grid differs wildly."""
        cos_small, _ = build_3d_rope_cache((2, 2, 2), HEAD_DIM, num_prefix_tokens=0)
        cos_large, _ = build_3d_rope_cache((16, 16, 16), HEAD_DIM, num_prefix_tokens=0)
        # Last patch: position (1,1,1) vs (15,15,15) — very different angles
        self.assertFalse(torch.allclose(cos_small[-1], cos_large[-1], atol=0.1))


class TestCoordAugmentation(unittest.TestCase):
    """Tests for training-time coordinate augmentations."""

    def test_coord_shift_produces_different_outputs(self):
        """With shift augmentation, two calls should (almost certainly) differ."""
        torch.manual_seed(0)
        cos1, sin1 = build_3d_rope_cache(
            GRID,
            HEAD_DIM,
            num_prefix_tokens=0,
            theta=100.0,
            normalize_coords=True,
            coord_shift=0.5,
        )
        torch.manual_seed(1)
        cos2, sin2 = build_3d_rope_cache(
            GRID,
            HEAD_DIM,
            num_prefix_tokens=0,
            theta=100.0,
            normalize_coords=True,
            coord_shift=0.5,
        )
        self.assertFalse(torch.allclose(cos1, cos2))

    def test_coord_jitter_produces_different_outputs(self):
        torch.manual_seed(0)
        cos1, _ = build_3d_rope_cache(
            GRID,
            HEAD_DIM,
            num_prefix_tokens=0,
            theta=100.0,
            normalize_coords=True,
            coord_jitter=2.0,
        )
        torch.manual_seed(1)
        cos2, _ = build_3d_rope_cache(
            GRID,
            HEAD_DIM,
            num_prefix_tokens=0,
            theta=100.0,
            normalize_coords=True,
            coord_jitter=2.0,
        )
        self.assertFalse(torch.allclose(cos1, cos2))

    def test_coord_rescale_produces_different_outputs(self):
        torch.manual_seed(0)
        cos1, _ = build_3d_rope_cache(
            GRID,
            HEAD_DIM,
            num_prefix_tokens=0,
            theta=100.0,
            normalize_coords=True,
            coord_rescale=2.0,
        )
        torch.manual_seed(1)
        cos2, _ = build_3d_rope_cache(
            GRID,
            HEAD_DIM,
            num_prefix_tokens=0,
            theta=100.0,
            normalize_coords=True,
            coord_rescale=2.0,
        )
        self.assertFalse(torch.allclose(cos1, cos2))

    def test_augmentation_preserves_prefix_identity(self):
        """Prefix tokens must remain identity even with augmentation."""
        cos, sin = build_3d_rope_cache(
            GRID,
            HEAD_DIM,
            num_prefix_tokens=2,
            theta=100.0,
            normalize_coords=True,
            coord_shift=0.5,
            coord_jitter=2.0,
            coord_rescale=2.0,
        )
        for i in range(2):
            torch.testing.assert_close(cos[i], torch.ones(HEAD_DIM // 2))
            torch.testing.assert_close(sin[i], torch.zeros(HEAD_DIM // 2))

    def test_augmentation_works_with_raw_positions(self):
        """Augmentation should also work when normalize_coords=False."""
        torch.manual_seed(0)
        cos1, _ = build_3d_rope_cache(
            GRID,
            HEAD_DIM,
            num_prefix_tokens=0,
            coord_shift=1.0,
        )
        torch.manual_seed(1)
        cos2, _ = build_3d_rope_cache(
            GRID,
            HEAD_DIM,
            num_prefix_tokens=0,
            coord_shift=1.0,
        )
        self.assertFalse(torch.allclose(cos1, cos2))


class TestApplyRotaryEmb(unittest.TestCase):
    def test_identity_rotation_returns_input_unchanged(self):
        """cos=1, sin=0 should return input unchanged."""
        x = torch.randn(2, 10, 6, HEAD_DIM)
        cos = torch.ones(10, HEAD_DIM // 2)
        sin = torch.zeros(10, HEAD_DIM // 2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        out = apply_rotary_emb(x, cos, sin)
        torch.testing.assert_close(out, x)

    def test_rotary_emb_returns_same_shape_as_input(self):
        x = torch.randn(2, 10, 6, HEAD_DIM)
        cos = torch.randn(10, HEAD_DIM // 2).unsqueeze(0).unsqueeze(2)
        sin = torch.randn(10, HEAD_DIM // 2).unsqueeze(0).unsqueeze(2)
        out = apply_rotary_emb(x, cos, sin)
        self.assertEqual(out.shape, x.shape)

    def test_non_identity_rotation_returns_different_output(self):
        """Non-identity rotation should change the tensor."""
        x = torch.randn(2, 10, 6, HEAD_DIM)
        cos = torch.randn(10, HEAD_DIM // 2).unsqueeze(0).unsqueeze(2)
        sin = torch.randn(10, HEAD_DIM // 2).unsqueeze(0).unsqueeze(2)
        out = apply_rotary_emb(x, cos, sin)
        self.assertFalse(torch.allclose(out, x))

    def test_inverse_rotation_recovers_original_input(self):
        """Applying rotation then its inverse should recover the original."""
        x = torch.randn(2, 10, 6, HEAD_DIM)
        angles = torch.randn(10, HEAD_DIM // 2).unsqueeze(0).unsqueeze(2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        rotated = apply_rotary_emb(x, cos, sin)
        recovered = apply_rotary_emb(rotated, cos, -sin)
        torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)


class TestConcatRopeForNested(unittest.TestCase):
    def test_concat_returns_total_length_across_all_crops(self):
        """Two crops, each with batch=2, should produce properly concatenated rope."""
        grid_a = (4, 4, 4)
        grid_b = (2, 2, 2)
        rope_a = build_3d_rope_cache(grid_a, HEAD_DIM, num_prefix_tokens=1)
        rope_b = build_3d_rope_cache(grid_b, HEAD_DIM, num_prefix_tokens=1)

        N_a = 1 + 4 * 4 * 4  # 65
        N_b = 1 + 2 * 2 * 2  # 9
        B_a, B_b = 2, 3

        cos_cat, sin_cat = concat_rope_for_nested(
            [rope_a, rope_b],
            batch_sizes=[B_a, B_b],
            seq_lens=[N_a, N_b],
        )
        expected_len = B_a * N_a + B_b * N_b
        self.assertEqual(cos_cat.shape, (expected_len, HEAD_DIM // 2))
        self.assertEqual(sin_cat.shape, (expected_len, HEAD_DIM // 2))


class TestAttentionWithRope(unittest.TestCase):
    def setUp(self):
        self.rope_cache = _make_rope_cache()

    def _make_attn(self):
        return Attention(dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, proj_bias=True)

    def test_forward_without_rope_returns_same_shape(self):
        """Baseline: forward without RoPE should work as before."""
        attn = self._make_attn()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_with_rope_returns_same_shape(self):
        attn = self._make_attn()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM)
        out = attn(x, rope=self.rope_cache)
        self.assertEqual(out.shape, x.shape)

    def test_rope_returns_different_output_than_no_rope(self):
        """Output should differ when RoPE is applied vs not."""
        attn = self._make_attn()
        attn.eval()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM)
        out_no_rope = attn(x)
        out_with_rope = attn(x, rope=self.rope_cache)
        self.assertFalse(torch.allclose(out_no_rope, out_with_rope, atol=1e-5))

    def test_same_input_and_rope_returns_identical_output(self):
        """Same input + same RoPE -> same output."""
        attn = self._make_attn()
        attn.eval()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM)
        out1 = attn(x, rope=self.rope_cache)
        out2 = attn(x, rope=self.rope_cache)
        torch.testing.assert_close(out1, out2)

    def test_attn_mode_returns_attention_map_shape(self):
        """vit_feat='attn' should return attention maps, not patch features."""
        attn = self._make_attn()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM)
        out = attn(x, vit_feat="attn", rope=self.rope_cache)
        self.assertEqual(out.shape, (2, 1 + NUM_PATCHES, NUM_HEADS))

    def test_patch_attn_mode_returns_concatenated_shape(self):
        """vit_feat='patch_attn' should concatenate features and attention."""
        attn = self._make_attn()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM)
        out = attn(x, vit_feat="patch_attn", rope=self.rope_cache)
        self.assertEqual(out.shape, (2, 1 + NUM_PATCHES, DIM + NUM_HEADS))


@requires_xformers
@requires_cuda
class TestMemEffAttentionWithRope(unittest.TestCase):
    """MemEffAttention uses xFormers memory_efficient_attention which requires CUDA."""

    def _make_attn(self, device="cuda"):
        return MemEffAttention(
            dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, proj_bias=True
        ).to(device)

    def test_forward_with_rope_returns_same_shape(self):
        device = torch.device("cuda")
        attn = self._make_attn(device)
        attn.eval()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM, device=device)
        rope = build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=1, device=device)
        out = attn(x, rope=rope)
        self.assertEqual(out.shape, x.shape)

    def test_rope_returns_different_output_than_no_rope(self):
        device = torch.device("cuda")
        attn = self._make_attn(device)
        attn.eval()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM, device=device)
        rope = build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=1, device=device)
        out_no_rope = attn(x)
        out_with_rope = attn(x, rope=rope)
        self.assertFalse(torch.allclose(out_no_rope, out_with_rope, atol=1e-5))

    def test_forward_without_rope_returns_same_shape(self):
        device = torch.device("cuda")
        attn = self._make_attn(device)
        x = torch.randn(2, 1 + NUM_PATCHES, DIM, device=device)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_same_input_and_rope_returns_identical_output(self):
        device = torch.device("cuda")
        attn = self._make_attn(device)
        attn.eval()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM, device=device)
        rope = build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=1, device=device)
        out1 = attn(x, rope=rope)
        out2 = attn(x, rope=rope)
        torch.testing.assert_close(out1, out2)


class TestMemEffAttentionFallbackWithRope(unittest.TestCase):
    """Test MemEffAttention's fallback to Attention.forward when xFormers is unavailable."""

    def setUp(self):
        self.rope_cache = _make_rope_cache()

    def _make_attn(self):
        return MemEffAttention(
            dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, proj_bias=True
        )

    def test_fallback_with_rope_returns_same_shape(self):
        """MemEffAttention should fall back to Attention.forward and still apply RoPE."""
        import spatialdino.models.layers.attention as attn_mod

        with mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False):
            attn = self._make_attn()
            attn.eval()
            x = torch.randn(2, 1 + NUM_PATCHES, DIM)
            out = attn(x, rope=self.rope_cache)
            self.assertEqual(out.shape, x.shape)

    def test_attn_bias_without_xformers_raises_assertion_error(self):
        """attn_bias requires xFormers; should raise when unavailable."""
        import spatialdino.models.layers.attention as attn_mod

        with mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False):
            attn = self._make_attn()
            x = torch.randn(2, 1 + NUM_PATCHES, DIM)
            with self.assertRaisesRegex(AssertionError, "xFormers is required"):
                attn(x, attn_bias="dummy", rope=self.rope_cache)

    def test_fallback_attn_mode_returns_attention_map_shape(self):
        import spatialdino.models.layers.attention as attn_mod

        with mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False):
            attn = self._make_attn()
            x = torch.randn(2, 1 + NUM_PATCHES, DIM)
            out = attn(x, vit_feat="attn", rope=self.rope_cache)
            self.assertEqual(out.shape, (2, 1 + NUM_PATCHES, NUM_HEADS))

    def test_fallback_patch_attn_mode_returns_concatenated_shape(self):
        import spatialdino.models.layers.attention as attn_mod

        with mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False):
            attn = self._make_attn()
            x = torch.randn(2, 1 + NUM_PATCHES, DIM)
            out = attn(x, vit_feat="patch_attn", rope=self.rope_cache)
            self.assertEqual(out.shape, (2, 1 + NUM_PATCHES, DIM + NUM_HEADS))


@requires_cuda
class TestAttentionWithRopeGPU(unittest.TestCase):
    """Test vanilla Attention with RoPE on CUDA (no xFormers dependency)."""

    def _make_attn(self, device="cuda"):
        return Attention(
            dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, proj_bias=True
        ).to(device)

    def test_gpu_forward_with_rope_returns_same_shape(self):
        device = torch.device("cuda")
        attn = self._make_attn(device)
        attn.eval()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM, device=device)
        rope = build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=1, device=device)
        out = attn(x, rope=rope)
        self.assertEqual(out.shape, x.shape)

    def test_gpu_rope_returns_different_output_than_no_rope(self):
        device = torch.device("cuda")
        attn = self._make_attn(device)
        attn.eval()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM, device=device)
        rope = build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=1, device=device)
        out_no_rope = attn(x)
        out_with_rope = attn(x, rope=rope)
        self.assertFalse(torch.allclose(out_no_rope, out_with_rope, atol=1e-5))

    def test_cpu_rope_with_gpu_input_raises_runtime_error(self):
        """RoPE cache on CPU with input on GPU should raise a RuntimeError."""
        device = torch.device("cuda")
        attn = self._make_attn(device)
        x = torch.randn(2, 1 + NUM_PATCHES, DIM, device=device)
        rope_cpu = build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=1)
        with self.assertRaises(RuntimeError):
            attn(x, rope=rope_cpu)


@requires_cuda
class TestMemEffAttentionFallbackWithRopeGPU(unittest.TestCase):
    """Test MemEffAttention fallback on CUDA (xFormers monkeypatched away)."""

    def _make_attn(self, device="cuda"):
        return MemEffAttention(
            dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, proj_bias=True
        ).to(device)

    def test_gpu_fallback_with_rope_returns_same_shape(self):
        import spatialdino.models.layers.attention as attn_mod

        with mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False):
            device = torch.device("cuda")
            attn = self._make_attn(device)
            attn.eval()
            x = torch.randn(2, 1 + NUM_PATCHES, DIM, device=device)
            rope = build_3d_rope_cache(
                GRID, HEAD_DIM, num_prefix_tokens=1, device=device
            )
            out = attn(x, rope=rope)
            self.assertEqual(out.shape, x.shape)


class TestCrossPathwayAgreement(unittest.TestCase):
    """Verify that different attention pathways produce numerically identical results."""

    def setUp(self):
        self.rope_cache = _make_rope_cache()

    def test_vanilla_and_memeff_fallback_return_identical_output(self):
        """Attention.forward and MemEffAttention fallback must produce the same output."""
        import spatialdino.models.layers.attention as attn_mod

        with mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False):
            attn_vanilla = Attention(
                dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, proj_bias=True
            )
            attn_vanilla.eval()

            memeff = MemEffAttention(
                dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, proj_bias=True
            )
            memeff.load_state_dict(attn_vanilla.state_dict())
            memeff.eval()

            x = torch.randn(2, 1 + NUM_PATCHES, DIM)
            out_vanilla = attn_vanilla(x, rope=self.rope_cache)
            out_memeff = memeff(x, rope=self.rope_cache)
            torch.testing.assert_close(out_vanilla, out_memeff)

    def test_vanilla_and_memeff_fallback_match_without_rope(self):
        """Baseline: agreement without RoPE."""
        import spatialdino.models.layers.attention as attn_mod

        with mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False):
            attn_vanilla = Attention(
                dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, proj_bias=True
            )
            attn_vanilla.eval()

            memeff = MemEffAttention(
                dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, proj_bias=True
            )
            memeff.load_state_dict(attn_vanilla.state_dict())
            memeff.eval()

            x = torch.randn(2, 1 + NUM_PATCHES, DIM)
            out_vanilla = attn_vanilla(x)
            out_memeff = memeff(x)
            torch.testing.assert_close(out_vanilla, out_memeff)

    @requires_xformers
    @requires_cuda
    def test_xformers_and_fallback_return_numerically_close_output(self):
        """xFormers and fallback paths should produce numerically close results."""
        import spatialdino.models.layers.attention as attn_mod

        device = torch.device("cuda")
        attn = MemEffAttention(
            dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, proj_bias=True
        ).to(device)
        attn.eval()

        x = torch.randn(2, 1 + NUM_PATCHES, DIM, device=device)
        rope = build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=1, device=device)

        # xFormers path
        out_xformers = attn(x, rope=rope)

        # Fallback path
        with mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False):
            out_fallback = attn(x, rope=rope)

        torch.testing.assert_close(out_xformers, out_fallback, atol=1e-2, rtol=1e-2)


class TestBlockWithRope(unittest.TestCase):
    def setUp(self):
        self.rope_cache = _make_rope_cache()

    def _make_block(self, drop_path=0.0):
        return Block(
            dim=DIM,
            num_heads=NUM_HEADS,
            qkv_bias=True,
            proj_bias=True,
            attn_class=Attention,
            drop_path=drop_path,
        )

    def test_block_with_rope_returns_same_shape(self):
        block = self._make_block()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM)
        out = block(x, rope=self.rope_cache)
        self.assertEqual(out.shape, x.shape)

    def test_block_rope_returns_different_output_than_no_rope(self):
        block = self._make_block()
        block.eval()
        x = torch.randn(2, 1 + NUM_PATCHES, DIM)
        out_no_rope = block(x)
        out_with_rope = block(x, rope=self.rope_cache)
        self.assertFalse(torch.allclose(out_no_rope, out_with_rope, atol=1e-5))

    def test_block_with_high_drop_path_returns_same_shape(self):
        """Block with drop_path > 0.1 uses stochastic depth -- rope must still thread through."""
        block = self._make_block(drop_path=0.2)
        block.train()
        x = torch.randn(4, 1 + NUM_PATCHES, DIM)
        out = block(x, rope=self.rope_cache)
        self.assertEqual(out.shape, x.shape)

    def test_block_with_low_drop_path_returns_same_shape(self):
        """Block with 0 < drop_path <= 0.1 uses the drop_path1 branch."""
        block = self._make_block(drop_path=0.05)
        block.train()
        x = torch.randn(4, 1 + NUM_PATCHES, DIM)
        out = block(x, rope=self.rope_cache)
        self.assertEqual(out.shape, x.shape)


# ===========================================================================
# 7. NestedTensorBlock with RoPE
# ===========================================================================


@requires_xformers
@requires_cuda
class TestNestedTensorBlockWithRope(unittest.TestCase):
    """NestedTensorBlock uses MemEffAttention which requires CUDA."""

    def _make_block(self, drop_path=0.0, device="cuda"):
        return NestedTensorBlock(
            dim=DIM,
            num_heads=NUM_HEADS,
            qkv_bias=True,
            proj_bias=True,
            attn_class=MemEffAttention,
            drop_path=drop_path,
        ).to(device)

    def test_single_tensor_returns_same_shape(self):
        """NestedTensorBlock with a single tensor should delegate to Block.forward."""
        device = torch.device("cuda")
        block = self._make_block(device=device)
        x = torch.randn(2, 1 + NUM_PATCHES, DIM, device=device)
        rope = build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=1, device=device)
        out = block(x, rope=rope)
        self.assertEqual(out.shape, x.shape)

    def test_nested_list_returns_matching_per_crop_shapes(self):
        """Test the nested tensor (list of crops) path with per-crop RoPE."""
        device = torch.device("cuda")
        block = self._make_block(device=device)
        block.eval()

        grid_a = (4, 4, 4)
        grid_b = (2, 2, 2)
        N_a = 1 + 4 * 4 * 4
        N_b = 1 + 2 * 2 * 2

        x_a = torch.randn(2, N_a, DIM, device=device)
        x_b = torch.randn(2, N_b, DIM, device=device)

        rope_a = build_3d_rope_cache(
            grid_a, HEAD_DIM, num_prefix_tokens=1, device=device
        )
        rope_b = build_3d_rope_cache(
            grid_b, HEAD_DIM, num_prefix_tokens=1, device=device
        )

        out = block([x_a, x_b], rope=[rope_a, rope_b])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, x_a.shape)
        self.assertEqual(out[1].shape, x_b.shape)

    def test_nested_list_rope_returns_different_output_than_no_rope(self):
        """Verify RoPE affects output in the nested tensor path."""
        device = torch.device("cuda")
        block = self._make_block(device=device)
        block.eval()

        grid_a = (4, 4, 4)
        N_a = 1 + 4 * 4 * 4

        x_a = torch.randn(2, N_a, DIM, device=device)
        x_b = torch.randn(2, N_a, DIM, device=device)

        rope_a = build_3d_rope_cache(
            grid_a, HEAD_DIM, num_prefix_tokens=1, device=device
        )

        out_no_rope = block([x_a, x_b])
        out_with_rope = block([x_a, x_b], rope=[rope_a, rope_a])

        self.assertFalse(torch.allclose(out_no_rope[0], out_with_rope[0], atol=1e-5))

    def test_nested_with_drop_path_returns_matching_shapes(self):
        """Test nested tensor path with stochastic depth > 0 during training."""
        device = torch.device("cuda")
        block = self._make_block(drop_path=0.2, device=device)
        block.train()

        grid_a = (4, 4, 4)
        N_a = 1 + 4 * 4 * 4

        x_a = torch.randn(4, N_a, DIM, device=device)
        x_b = torch.randn(4, N_a, DIM, device=device)

        rope_a = build_3d_rope_cache(
            grid_a, HEAD_DIM, num_prefix_tokens=1, device=device
        )

        out = block([x_a, x_b], rope=[rope_a, rope_a])
        self.assertIsInstance(out, list)
        self.assertEqual(out[0].shape, x_a.shape)
        self.assertEqual(out[1].shape, x_b.shape)


# ===========================================================================
# 7a. NestedTensorBlock single-tensor path (CPU fallback)
# ===========================================================================


class TestNestedTensorBlockSingleTensorFallback(unittest.TestCase):
    """NestedTensorBlock with a single tensor delegates to Block.forward,
    which works without xFormers (via MemEffAttention fallback or vanilla Attention)."""

    def setUp(self):
        self.rope_cache = _make_rope_cache()

    def _make_block(self, drop_path=0.0):
        return NestedTensorBlock(
            dim=DIM,
            num_heads=NUM_HEADS,
            qkv_bias=True,
            proj_bias=True,
            attn_class=MemEffAttention,
            drop_path=drop_path,
        )

    def test_cpu_fallback_returns_same_shape(self):
        import spatialdino.models.layers.attention as attn_mod
        import spatialdino.models.layers.block as block_mod

        with (
            mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False),
            mock_patch.object(block_mod, "XFORMERS_AVAILABLE", False),
        ):
            block = self._make_block()
            block.eval()
            x = torch.randn(2, 1 + NUM_PATCHES, DIM)
            out = block(x, rope=self.rope_cache)
            self.assertEqual(out.shape, x.shape)

    def test_cpu_fallback_rope_returns_different_output_than_no_rope(self):
        import spatialdino.models.layers.attention as attn_mod
        import spatialdino.models.layers.block as block_mod

        with (
            mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False),
            mock_patch.object(block_mod, "XFORMERS_AVAILABLE", False),
        ):
            block = self._make_block()
            block.eval()
            x = torch.randn(2, 1 + NUM_PATCHES, DIM)
            out_no_rope = block(x)
            out_with_rope = block(x, rope=self.rope_cache)
            self.assertFalse(torch.allclose(out_no_rope, out_with_rope, atol=1e-5))

    def test_nested_list_without_xformers_raises_assertion_error(self):
        """Nested list path should raise when xFormers is unavailable."""
        import spatialdino.models.layers.attention as attn_mod
        import spatialdino.models.layers.block as block_mod

        with (
            mock_patch.object(attn_mod, "XFORMERS_AVAILABLE", False),
            mock_patch.object(block_mod, "XFORMERS_AVAILABLE", False),
        ):
            block = self._make_block()
            x_a = torch.randn(2, 1 + NUM_PATCHES, DIM)
            with self.assertRaisesRegex(AssertionError, "xFormers is required"):
                block([x_a, x_a], rope=[self.rope_cache, self.rope_cache])


# ===========================================================================
# 8. Full Encoder with pos_embed_type="rope"
# ===========================================================================


class TestEncoderWithRope(unittest.TestCase):
    IMG_SIZE = (32, 32, 32)
    PATCH_SIZE = (8, 8, 8)
    ENCODER_DIM = 192
    ENCODER_HEADS = 6
    DEPTH = 2

    def _make_encoder(self, pos_embed_type="rope", depth=None, use_memeff=False):
        if use_memeff:
            block_fn = partial(NestedTensorBlock, attn_class=MemEffAttention)
        else:
            block_fn = partial(Block, attn_class=Attention)
        return Encoder(
            img_size=self.IMG_SIZE,
            patch_size=self.PATCH_SIZE,
            in_chans=1,
            embed_dim=self.ENCODER_DIM,
            depth=depth or self.DEPTH,
            num_heads=self.ENCODER_HEADS,
            mlp_ratio=2.0,
            qkv_bias=True,
            pos_embed_type=pos_embed_type,
            drop_path_rate=0.0,
            block_fn=block_fn,
        )

    def test_encoder_init_sets_rope_enabled_and_no_pos_embed(self):
        enc = self._make_encoder()
        self.assertTrue(enc.use_rope)
        self.assertFalse(enc.use_pos_embed)
        self.assertIsNone(enc.pos_embed)

    def test_encoder_forward_returns_correct_patch_token_shape(self):
        enc = self._make_encoder()
        enc.eval()
        x = torch.randn(1, 1, *self.IMG_SIZE)
        out = enc(x)
        self.assertIn("x_norm_clstoken", out)
        grid = tuple(s // p for s, p in zip(self.IMG_SIZE, self.PATCH_SIZE))
        num_patches = grid[0] * grid[1] * grid[2]
        self.assertEqual(
            out["x_norm_patchtokens"].shape, (1, num_patches, self.ENCODER_DIM)
        )

    def test_encoder_rope_returns_different_output_than_none(self):
        """Encoder with rope should differ from encoder with no pos embed (same weights)."""
        enc_rope = self._make_encoder(pos_embed_type="rope")
        enc_none = self._make_encoder(pos_embed_type="none")

        enc_none.load_state_dict(enc_rope.state_dict(), strict=False)
        enc_rope.eval()
        enc_none.eval()

        x = torch.randn(1, 1, *self.IMG_SIZE)
        out_rope = enc_rope(x)
        out_none = enc_none(x)

        self.assertFalse(
            torch.allclose(
                out_rope["x_norm_patchtokens"],
                out_none["x_norm_patchtokens"],
                atol=1e-5,
            )
        )

    @requires_xformers
    @requires_cuda
    def test_encoder_list_forward_returns_per_crop_shapes(self):
        """Test the list (training multi-crop) path with RoPE."""
        device = torch.device("cuda")
        enc = self._make_encoder(use_memeff=True).to(device)
        enc.eval()

        x_global = torch.randn(2, 1, 32, 32, 32, device=device)
        x_local = torch.randn(2, 1, 16, 16, 16, device=device)

        grid_g = tuple(s // p for s, p in zip((32, 32, 32), self.PATCH_SIZE))
        grid_l = tuple(s // p for s, p in zip((16, 16, 16), self.PATCH_SIZE))
        n_g = grid_g[0] * grid_g[1] * grid_g[2]
        n_l = grid_l[0] * grid_l[1] * grid_l[2]

        masks_g = torch.zeros(2, n_g, dtype=torch.bool, device=device)
        masks_l = torch.zeros(2, n_l, dtype=torch.bool, device=device)

        out = enc([x_global, x_local], masks=[masks_g, masks_l])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["x_norm_patchtokens"].shape, (2, n_g, self.ENCODER_DIM))
        self.assertEqual(out[1]["x_norm_patchtokens"].shape, (2, n_l, self.ENCODER_DIM))

    def test_predict_returns_spatial_feature_map(self):
        """Test the _predict inference path (single tensor, no masks)."""
        enc = self._make_encoder()
        enc.eval()
        x = torch.randn(1, 1, *self.IMG_SIZE)
        grid = tuple(s // p for s, p in zip(self.IMG_SIZE, self.PATCH_SIZE))
        with torch.no_grad():
            out = enc._predict(x, vit_feat="patch")
        self.assertEqual(out.shape, (1, self.ENCODER_DIM, *grid))

    def test_predict_with_registers_returns_spatial_feature_map(self):
        """Test _predict with register tokens (they should get identity RoPE)."""
        enc = Encoder(
            img_size=self.IMG_SIZE,
            patch_size=self.PATCH_SIZE,
            in_chans=1,
            embed_dim=self.ENCODER_DIM,
            depth=self.DEPTH,
            num_heads=self.ENCODER_HEADS,
            mlp_ratio=2.0,
            qkv_bias=True,
            pos_embed_type="rope",
            drop_path_rate=0.0,
            num_tt_register_tokens=4,
            block_fn=partial(Block, attn_class=Attention),
        )
        enc.eval()
        x = torch.randn(1, 1, *self.IMG_SIZE)
        grid = tuple(s // p for s, p in zip(self.IMG_SIZE, self.PATCH_SIZE))
        with torch.no_grad():
            out = enc._predict(x, vit_feat="patch")
        self.assertEqual(out.shape, (1, self.ENCODER_DIM, *grid))

    def test_backward_pass_produces_nonzero_gradients(self):
        """Ensure gradients flow through the RoPE path."""
        enc = self._make_encoder()
        enc.train()
        x = torch.randn(1, 1, *self.IMG_SIZE, requires_grad=True)
        out = enc(x)
        loss = out["x_norm_clstoken"].sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum(), 0)


# ===========================================================================
# 8a. Encoder RoPE caching via RoPE3D module
# ===========================================================================


class TestRoPE3DModule(unittest.TestCase):
    """Tests for the RoPE3D nn.Module train/eval behavior."""

    def _make_module(self, **kwargs):
        return RoPE3D(head_dim=HEAD_DIM, theta=100.0, normalize_coords=True, **kwargs)

    def test_eval_mode_caches_results(self):
        m = self._make_module()
        m.eval()
        rope_a = m(GRID, num_prefix_tokens=1)
        rope_b = m(GRID, num_prefix_tokens=1)
        self.assertIs(rope_a, rope_b)

    def test_eval_mode_without_augmentation_is_deterministic(self):
        m = self._make_module(coord_shift=0.5)
        m.eval()
        rope_a = m(GRID, num_prefix_tokens=0)
        rope_b = m(GRID, num_prefix_tokens=0)
        # In eval mode augmentation is disabled — should be cached and identical
        self.assertIs(rope_a, rope_b)

    def test_train_mode_with_augmentation_produces_fresh_results(self):
        m = self._make_module(coord_shift=0.5, coord_jitter=2.0)
        m.train()
        torch.manual_seed(0)
        cos1, _ = m(GRID, num_prefix_tokens=0)
        torch.manual_seed(1)
        cos2, _ = m(GRID, num_prefix_tokens=0)
        self.assertFalse(torch.allclose(cos1, cos2))

    def test_train_mode_without_augmentation_caches(self):
        m = self._make_module()  # no augmentation configured
        m.train()
        rope_a = m(GRID, num_prefix_tokens=1)
        rope_b = m(GRID, num_prefix_tokens=1)
        self.assertIs(rope_a, rope_b)

    def test_clear_cache_empties_module_cache(self):
        m = self._make_module()
        m.eval()
        m(GRID, num_prefix_tokens=1)
        self.assertGreater(len(m._cache), 0)
        m.clear_cache()
        self.assertEqual(len(m._cache), 0)

    def test_state_dict_is_empty(self):
        """RoPE3D has no learnable params — state_dict should be empty."""
        m = self._make_module()
        self.assertEqual(len(m.state_dict()), 0)

    def test_augmentation_disabled_in_eval_preserves_prefix_identity(self):
        m = self._make_module(coord_shift=0.5, coord_jitter=2.0, coord_rescale=2.0)
        m.eval()
        cos, sin = m(GRID, num_prefix_tokens=2)
        for i in range(2):
            torch.testing.assert_close(cos[i], torch.ones(HEAD_DIM // 2))
            torch.testing.assert_close(sin[i], torch.zeros(HEAD_DIM // 2))


class TestEncoderRopeCaching(unittest.TestCase):
    """Verify that the Encoder's RoPE3D module caching behaves correctly."""

    IMG_SIZE = (32, 32, 32)
    PATCH_SIZE = (8, 8, 8)
    ENCODER_DIM = 192
    ENCODER_HEADS = 6
    DEPTH = 2

    def _make_encoder(self, **kwargs):
        return Encoder(
            img_size=self.IMG_SIZE,
            patch_size=self.PATCH_SIZE,
            in_chans=1,
            embed_dim=self.ENCODER_DIM,
            depth=self.DEPTH,
            num_heads=self.ENCODER_HEADS,
            mlp_ratio=2.0,
            qkv_bias=True,
            pos_embed_type="rope",
            drop_path_rate=0.0,
            block_fn=partial(Block, attn_class=Attention),
            **kwargs,
        )

    def test_forward_populates_one_cache_entry(self):
        """After one forward pass the rope cache dict should be non-empty."""
        enc = self._make_encoder()
        enc.eval()
        self.assertEqual(len(enc.rope._cache), 0)
        x = torch.randn(1, 1, *self.IMG_SIZE)
        enc(x)
        self.assertEqual(len(enc.rope._cache), 1)

    def test_repeated_forward_returns_unchanged_cache_size(self):
        """Calling forward twice with the same spatial size should not grow the cache."""
        enc = self._make_encoder()
        enc.eval()
        x = torch.randn(1, 1, *self.IMG_SIZE)
        enc(x)
        self.assertEqual(len(enc.rope._cache), 1)
        enc(x)
        self.assertEqual(len(enc.rope._cache), 1)

    def test_cache_hit_returns_same_object(self):
        """The exact same (cos, sin) tuple should be returned on cache hit."""
        enc = self._make_encoder()
        enc.eval()
        rope_a = enc._build_rope_cache(
            *self.IMG_SIZE, device=torch.device("cpu"), dtype=torch.float32
        )
        rope_b = enc._build_rope_cache(
            *self.IMG_SIZE, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertIs(rope_a, rope_b)

    def test_different_grid_sizes_create_separate_cache_entries(self):
        """Different spatial sizes should produce distinct cache entries."""
        enc = self._make_encoder()
        enc.eval()
        enc._build_rope_cache(
            32, 32, 32, device=torch.device("cpu"), dtype=torch.float32
        )
        enc._build_rope_cache(
            16, 16, 16, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(len(enc.rope._cache), 2)

    def test_different_prefix_counts_create_separate_cache_entries(self):
        """Same grid but different num_prefix_tokens -> separate entries."""
        enc = self._make_encoder()
        rope_1 = enc._build_rope_cache(
            32,
            32,
            32,
            device=torch.device("cpu"),
            dtype=torch.float32,
            num_prefix_tokens=1,
        )
        rope_5 = enc._build_rope_cache(
            32,
            32,
            32,
            device=torch.device("cpu"),
            dtype=torch.float32,
            num_prefix_tokens=5,
        )
        self.assertIsNot(rope_1, rope_5)
        self.assertEqual(len(enc.rope._cache), 2)

    def test_different_dtypes_create_separate_cache_entries(self):
        """Same grid but float32 vs float64 -> separate entries."""
        enc = self._make_encoder()
        r32 = enc._build_rope_cache(
            32, 32, 32, device=torch.device("cpu"), dtype=torch.float32
        )
        r64 = enc._build_rope_cache(
            32, 32, 32, device=torch.device("cpu"), dtype=torch.float64
        )
        self.assertIsNot(r32, r64)
        self.assertEqual(r32[0].dtype, torch.float32)
        self.assertEqual(r64[0].dtype, torch.float64)

    def test_clear_empties_cache(self):
        enc = self._make_encoder()
        enc.eval()
        x = torch.randn(1, 1, *self.IMG_SIZE)
        enc(x)
        self.assertGreater(len(enc.rope._cache), 0)
        enc.clear_rope_cache()
        self.assertEqual(len(enc.rope._cache), 0)

    def test_cached_values_match_fresh_computation(self):
        """Cache hit must return the same numerical values as a fresh computation."""
        enc = self._make_encoder()
        rope_first = enc._build_rope_cache(
            32, 32, 32, device=torch.device("cpu"), dtype=torch.float32
        )
        rope_second = enc._build_rope_cache(
            32, 32, 32, device=torch.device("cpu"), dtype=torch.float32
        )
        head_dim = self.ENCODER_DIM // self.ENCODER_HEADS
        grid = tuple(s // p for s, p in zip(self.IMG_SIZE, self.PATCH_SIZE))
        cos_fresh, sin_fresh = build_3d_rope_cache(grid, head_dim, num_prefix_tokens=1)
        torch.testing.assert_close(rope_first[0], cos_fresh)
        torch.testing.assert_close(rope_first[1], sin_fresh)
        torch.testing.assert_close(rope_second[0], cos_fresh)
        torch.testing.assert_close(rope_second[1], sin_fresh)

    @requires_xformers
    @requires_cuda
    def test_multi_crop_forward_creates_two_cache_entries(self):
        """forward_features_list with 2 crop sizes should create 2 cache entries."""
        device = torch.device("cuda")
        enc = Encoder(
            img_size=self.IMG_SIZE,
            patch_size=self.PATCH_SIZE,
            in_chans=1,
            embed_dim=self.ENCODER_DIM,
            depth=self.DEPTH,
            num_heads=self.ENCODER_HEADS,
            mlp_ratio=2.0,
            qkv_bias=True,
            pos_embed_type="rope",
            drop_path_rate=0.0,
            block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        ).to(device)
        enc.eval()

        x_g = torch.randn(2, 1, 32, 32, 32, device=device)
        x_l = torch.randn(2, 1, 16, 16, 16, device=device)

        grid_g = tuple(s // p for s, p in zip((32, 32, 32), self.PATCH_SIZE))
        grid_l = tuple(s // p for s, p in zip((16, 16, 16), self.PATCH_SIZE))
        n_g = grid_g[0] * grid_g[1] * grid_g[2]
        n_l = grid_l[0] * grid_l[1] * grid_l[2]

        masks_g = torch.zeros(2, n_g, dtype=torch.bool, device=device)
        masks_l = torch.zeros(2, n_l, dtype=torch.bool, device=device)

        enc([x_g, x_l], masks=[masks_g, masks_l])
        self.assertEqual(len(enc.rope._cache), 2)

        enc([x_g, x_l], masks=[masks_g, masks_l])
        self.assertEqual(len(enc.rope._cache), 2)

    def test_independent_instances_return_identical_caches(self):
        """Two independent Encoder instances (simulating two DDP ranks) must
        produce bit-identical rope caches for the same input dimensions."""
        enc_a = self._make_encoder()
        enc_b = self._make_encoder()

        rope_a = enc_a._build_rope_cache(
            32, 32, 32, device=torch.device("cpu"), dtype=torch.float32
        )
        rope_b = enc_b._build_rope_cache(
            32, 32, 32, device=torch.device("cpu"), dtype=torch.float32
        )

        torch.testing.assert_close(rope_a[0], rope_b[0])
        torch.testing.assert_close(rope_a[1], rope_b[1])

    def test_state_dict_excludes_rope_cache(self):
        """The rope cache is derived data and must NOT appear in state_dict."""
        enc = self._make_encoder()
        enc.eval()
        x = torch.randn(1, 1, *self.IMG_SIZE)
        enc(x)
        sd = enc.state_dict()
        for key in sd:
            self.assertNotIn("_cache", key)

    def test_cache_stays_empty_when_rope_disabled(self):
        enc = Encoder(
            img_size=self.IMG_SIZE,
            patch_size=self.PATCH_SIZE,
            in_chans=1,
            embed_dim=self.ENCODER_DIM,
            depth=self.DEPTH,
            num_heads=self.ENCODER_HEADS,
            mlp_ratio=2.0,
            qkv_bias=True,
            pos_embed_type="none",
            drop_path_rate=0.0,
            block_fn=partial(Block, attn_class=Attention),
        )
        enc.eval()
        x = torch.randn(1, 1, *self.IMG_SIZE)
        enc(x)
        self.assertFalse(hasattr(enc, "rope"))

    def test_train_mode_with_augmentation_skips_cache(self):
        """During training with augmentation, cache should not be populated."""
        enc = self._make_encoder(rope_coord_shift=0.5)
        enc.train()
        enc._build_rope_cache(
            32, 32, 32, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(len(enc.rope._cache), 0)

    def test_train_mode_with_augmentation_produces_stochastic_results(self):
        """Two forward calls in training mode with augmentation should differ."""
        enc = self._make_encoder(
            rope_coord_shift=0.5,
            rope_coord_jitter=2.0,
        )
        enc.train()
        torch.manual_seed(0)
        rope_a = enc._build_rope_cache(
            32, 32, 32, device=torch.device("cpu"), dtype=torch.float32
        )
        torch.manual_seed(1)
        rope_b = enc._build_rope_cache(
            32, 32, 32, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertFalse(torch.allclose(rope_a[0], rope_b[0]))


# ===========================================================================
# 9. Equivariance / position sensitivity
# ===========================================================================


class TestPositionSensitivity(unittest.TestCase):
    """Verify that RoPE makes the model position-aware: swapping patch positions
    in the sequence should change the output (unlike a position-free model)."""

    def test_swapping_patch_positions_changes_cls_output(self):
        attn = Attention(dim=DIM, num_heads=NUM_HEADS, qkv_bias=True)
        attn.eval()

        N = 1 + NUM_PATCHES
        x = torch.randn(1, N, DIM)
        rope = build_3d_rope_cache(GRID, HEAD_DIM, num_prefix_tokens=1)

        out_original = attn(x, rope=rope)

        # Swap two patch tokens (positions 1 and 10)
        x_swapped = x.clone()
        x_swapped[:, 1], x_swapped[:, 10] = x[:, 10].clone(), x[:, 1].clone()
        out_swapped = attn(x_swapped, rope=rope)

        # Because RoPE encodes position, the CLS output should change
        self.assertFalse(
            torch.allclose(out_original[:, 0], out_swapped[:, 0], atol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
