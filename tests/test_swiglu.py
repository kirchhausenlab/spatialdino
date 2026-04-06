"""Tests for SwiGLU FFN layer and checkpoint portability."""

from __future__ import annotations

import unittest

import torch

from spatialdino.models.layers.swiglu_ffn import SwiGLUFFN, XFORMERS_AVAILABLE


class TestSwiGLUFFN(unittest.TestCase):
    def setUp(self):
        self.in_features = 64
        self.hidden_features = int(self.in_features * 4)
        self.batch = 2
        self.seq_len = 8

    def test_output_shape(self):
        ffn = SwiGLUFFN(
            in_features=self.in_features, hidden_features=self.hidden_features
        )
        x = torch.randn(self.batch, self.seq_len, self.in_features)
        y = ffn(x)
        self.assertEqual(y.shape, x.shape)

    def test_parameter_names_match_xformers_convention(self):
        """w12, w3 must be present (same names as xformers.ops.SwiGLU)."""
        ffn = SwiGLUFFN(
            in_features=self.in_features, hidden_features=self.hidden_features
        )
        param_names = {n.split(".")[0] for n, _ in ffn.named_parameters()}
        self.assertEqual(param_names, {"w12", "w3"})

    def test_hidden_dim_scaling(self):
        """Hidden dim should be (int(hidden * 2/3) + 7) // 8 * 8."""
        ffn = SwiGLUFFN(
            in_features=self.in_features, hidden_features=self.hidden_features
        )
        expected = (int(self.hidden_features * 2 / 3) + 7) // 8 * 8
        # w12 outputs 2 * hidden_features
        self.assertEqual(ffn.w12.out_features, 2 * expected)
        self.assertEqual(ffn.w3.in_features, expected)

    def test_custom_out_features(self):
        out_features = 32
        ffn = SwiGLUFFN(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=out_features,
        )
        x = torch.randn(self.batch, self.seq_len, self.in_features)
        y = ffn(x)
        self.assertEqual(y.shape, (self.batch, self.seq_len, out_features))

    @unittest.skipUnless(XFORMERS_AVAILABLE, "xFormers not installed")
    def test_state_dict_compatible_with_fused(self):
        """SwiGLUFFN and SwiGLUFFNFused must have identical state_dict keys and shapes."""
        from spatialdino.models.layers.swiglu_ffn import SwiGLUFFNFused

        plain = SwiGLUFFN(
            in_features=self.in_features, hidden_features=self.hidden_features
        )
        fused = SwiGLUFFNFused(
            in_features=self.in_features, hidden_features=self.hidden_features
        )

        plain_sd = plain.state_dict()
        fused_sd = fused.state_dict()
        self.assertEqual(set(plain_sd.keys()), set(fused_sd.keys()))

        for key in plain_sd:
            self.assertEqual(
                plain_sd[key].shape,
                fused_sd[key].shape,
                f"Shape mismatch for {key}",
            )

    @unittest.skipUnless(XFORMERS_AVAILABLE, "xFormers not installed")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_fused_and_plain_numerically_equivalent(self):
        """Same weights must produce the same output regardless of xFormers."""
        from spatialdino.models.layers.swiglu_ffn import SwiGLUFFNFused

        device = torch.device("cuda")
        plain = SwiGLUFFN(
            in_features=self.in_features, hidden_features=self.hidden_features
        ).to(device)
        fused = SwiGLUFFNFused(
            in_features=self.in_features, hidden_features=self.hidden_features
        ).to(device)
        fused.load_state_dict(plain.state_dict())

        plain.eval()
        fused.eval()
        x = torch.randn(self.batch, self.seq_len, self.in_features, device=device)
        with torch.no_grad():
            out_plain = plain(x)
            out_fused = fused(x)
        torch.testing.assert_close(out_plain, out_fused)#, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_encoder_with_swiglu(self):
        """Encoder can be built with ffn_layer='swiglu' without xFormers."""
        from spatialdino.models.layers.encoder import Encoder

        device = torch.device("cuda")
        enc = Encoder(
            img_size=(8, 8, 8),
            patch_size=(4, 4, 4),
            in_chans=1,
            embed_dim=64,
            depth=1,
            num_heads=4,
            mlp_ratio=4.0,
            ffn_layer="swiglu",
        ).to(device)
        x = torch.randn(1, 1, 8, 8, 8, device=device)
        out = enc(x)
        self.assertIn("x_norm_clstoken", out)

    def test_load_state_dict_portability(self):
        """A state_dict saved from one SwiGLUFFN can be loaded into another."""
        ffn1 = SwiGLUFFN(
            in_features=self.in_features, hidden_features=self.hidden_features
        )
        sd = ffn1.state_dict()

        ffn2 = SwiGLUFFN(
            in_features=self.in_features, hidden_features=self.hidden_features
        )
        ffn2.load_state_dict(sd)

        x = torch.randn(self.batch, self.seq_len, self.in_features)
        torch.testing.assert_close(ffn1(x), ffn2(x))


if __name__ == "__main__":
    unittest.main()
