"""Tests for StreamingEncoder with off-GPU K/V storage.

Verifies that streaming attention with CPU-resident or disk-backed K/V
produces numerically correct results (matching a full-sequence reference)
and, when CUDA is available, agrees with the GPU-resident K/V (SDPA) path.
"""
from __future__ import annotations

import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from spatialdino.inference.streaming.storage import TokenStore
from spatialdino.inference.streaming.streaming_encoder import StreamingEncoder


# ---------------------------------------------------------------------------
# Minimal transformer block / encoder stubs
# ---------------------------------------------------------------------------

class _FakeAttn(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = float((embed_dim // num_heads) ** -0.5)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_drop = nn.Identity()


class _FakeBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = _FakeAttn(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ls1 = nn.Identity()
        self.ls2 = nn.Identity()

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Full non-streaming forward pass, used as ground-truth reference."""
        B, N, D = x.shape
        H = self.attn.num_heads
        head_dim = D // H

        x_norm = self.norm1(x)
        qkv = self.attn.qkv(x_norm).view(B, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, scale=self.attn.scale)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.attn.proj(out)
        out = self.attn.proj_drop(out)
        out = self.ls1(out)
        x = x + out
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _FakeEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block: _FakeBlock) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = (2, 2, 2)
        self.stride = (2, 2, 2)
        self.num_register_tokens = 0
        self.num_tt_register_tokens = 0
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = None
        self.tt_register_tokens = None
        self.blocks = nn.ModuleList([block])

    @property
    def use_pos_embed(self) -> bool:
        return False

    def _build_rope_cache(self, z, y, x, device, dtype, num_prefix_tokens):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    device_type: str = "cpu",
    storage_kind: str = "cpu",
    kv_storage_kind: str = "cpu",
    q_block_tokens: int = 8,
    save_path: str | None = None,
) -> OmegaConf:
    cfg: dict = {
        "use_amp": False,
        "device_type": device_type,
        "dtype": "fp32",
        "streaming_storage": storage_kind,
        "streaming_kv_storage": kv_storage_kind,
        "streaming_q_block_tokens": q_block_tokens,
        "streaming_use_triton": False,
        "streaming_pin_memory": False,
    }
    if save_path is not None:
        cfg["save_path"] = save_path
    return OmegaConf.create(cfg)


def _run_block_with_storage(
    block: _FakeBlock,
    tokens: torch.Tensor,
    embed_dim: int,
    num_heads: int,
    *,
    q_block_tokens: int,
    kv_storage_kind: str,
    device: torch.device,
    save_path: str | None = None,
) -> torch.Tensor:
    """Run a single streaming block and return the output token tensor on CPU."""
    total_tokens = tokens.shape[0]
    grid_size = (2, 2, 2)
    tok_storage = "gpu" if device.type == "cuda" else "cpu"

    cfg = _make_config(
        device_type=device.type,
        storage_kind=tok_storage,
        kv_storage_kind=kv_storage_kind,
        q_block_tokens=q_block_tokens,
        save_path=save_path,
    )
    encoder = _FakeEncoder(embed_dim, num_heads, block)
    se = StreamingEncoder(encoder, device, cfg)

    block.to(device)

    x_in = TokenStore.create(
        (total_tokens, embed_dim), torch.float32, tok_storage, 1, grid_size, device
    )
    x_out = TokenStore.create(
        (total_tokens, embed_dim), torch.float32, tok_storage, 1, grid_size, device
    )
    x_in.tensor.copy_(tokens.to(x_in.tensor.device))

    with torch.no_grad():
        se._run_block(block, x_in, x_out, total_tokens, rope=None)

    block.to("cpu")
    return x_out.tensor.clone().cpu()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStreamingCpuKV(unittest.TestCase):
    """Streaming attention with off-GPU K/V gives correct results."""

    def _reference(
        self, block: _FakeBlock, tokens: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return block.forward_reference(tokens.unsqueeze(0)).squeeze(0)

    # ------------------------------------------------------------------
    # CPU K/V storage
    # ------------------------------------------------------------------

    def test_cpu_kv_many_blocks_matches_reference(self):
        """Online softmax across many small KV blocks accumulates correctly."""
        torch.manual_seed(0)
        embed_dim, num_heads, total = 32, 4, 48
        block = _FakeBlock(embed_dim, num_heads).eval()
        tokens = torch.randn(total, embed_dim)

        ref = self._reference(block, tokens)
        got = _run_block_with_storage(
            block, tokens, embed_dim, num_heads,
            q_block_tokens=8, kv_storage_kind="cpu",
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_cpu_kv_q_block_equals_total_tokens(self):
        """Single Q block (no Q chunking) still works with CPU KV."""
        torch.manual_seed(1)
        embed_dim, num_heads, total = 32, 4, 16
        block = _FakeBlock(embed_dim, num_heads).eval()
        tokens = torch.randn(total, embed_dim)

        ref = self._reference(block, tokens)
        got = _run_block_with_storage(
            block, tokens, embed_dim, num_heads,
            q_block_tokens=total, kv_storage_kind="cpu",
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_cpu_kv_kv_block_equals_total_tokens(self):
        """Entire K/V in one chunk (no KV chunking) with CPU storage."""
        torch.manual_seed(2)
        embed_dim, num_heads, total = 32, 4, 32
        block = _FakeBlock(embed_dim, num_heads).eval()
        tokens = torch.randn(total, embed_dim)

        ref = self._reference(block, tokens)
        # kv_block_tokens defaults to q_block_tokens; pass total to get one chunk
        got = _run_block_with_storage(
            block, tokens, embed_dim, num_heads,
            q_block_tokens=total, kv_storage_kind="cpu",
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_cpu_kv_uneven_block_boundaries(self):
        """Total tokens not divisible by block size — boundary handling is correct."""
        torch.manual_seed(3)
        embed_dim, num_heads, total = 32, 4, 30  # 30 % 8 != 0
        block = _FakeBlock(embed_dim, num_heads).eval()
        tokens = torch.randn(total, embed_dim)

        ref = self._reference(block, tokens)
        got = _run_block_with_storage(
            block, tokens, embed_dim, num_heads,
            q_block_tokens=8, kv_storage_kind="cpu",
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    # ------------------------------------------------------------------
    # Disk K/V storage
    # ------------------------------------------------------------------

    def test_disk_kv_matches_reference(self):
        """Memory-mapped (disk-backed) K/V storage gives correct results."""
        torch.manual_seed(4)
        embed_dim, num_heads, total = 32, 4, 24
        block = _FakeBlock(embed_dim, num_heads).eval()
        tokens = torch.randn(total, embed_dim)

        ref = self._reference(block, tokens)
        with tempfile.TemporaryDirectory() as tmp:
            got = _run_block_with_storage(
                block, tokens, embed_dim, num_heads,
                q_block_tokens=8, kv_storage_kind="disk",
                device=torch.device("cpu"), save_path=tmp,
            )

        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_disk_kv_uneven_boundaries(self):
        """Disk KV with non-divisible token count handles partial final block."""
        torch.manual_seed(5)
        embed_dim, num_heads, total = 32, 4, 29
        block = _FakeBlock(embed_dim, num_heads).eval()
        tokens = torch.randn(total, embed_dim)

        ref = self._reference(block, tokens)
        with tempfile.TemporaryDirectory() as tmp:
            got = _run_block_with_storage(
                block, tokens, embed_dim, num_heads,
                q_block_tokens=8, kv_storage_kind="disk",
                device=torch.device("cpu"), save_path=tmp,
            )

        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    # ------------------------------------------------------------------
    # GPU vs CPU agreement (requires CUDA)
    # ------------------------------------------------------------------

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_kv_matches_gpu_kv_sdpa(self):
        """CPU K/V (manual online softmax) agrees with GPU K/V (SDPA)."""
        torch.manual_seed(6)
        embed_dim, num_heads, total = 32, 4, 48
        block = _FakeBlock(embed_dim, num_heads).eval()
        tokens = torch.randn(total, embed_dim)
        device = torch.device("cuda")

        cpu_out = _run_block_with_storage(
            block, tokens, embed_dim, num_heads,
            q_block_tokens=8, kv_storage_kind="cpu", device=device,
        )
        gpu_out = _run_block_with_storage(
            block, tokens, embed_dim, num_heads,
            q_block_tokens=8, kv_storage_kind="gpu", device=device,
        )

        torch.testing.assert_close(cpu_out, gpu_out, rtol=1e-3, atol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_disk_kv_matches_gpu_kv_sdpa(self):
        """Disk K/V agrees with GPU K/V (SDPA) when CUDA is available."""
        torch.manual_seed(7)
        embed_dim, num_heads, total = 32, 4, 32
        block = _FakeBlock(embed_dim, num_heads).eval()
        tokens = torch.randn(total, embed_dim)
        device = torch.device("cuda")

        with tempfile.TemporaryDirectory() as tmp:
            disk_out = _run_block_with_storage(
                block, tokens, embed_dim, num_heads,
                q_block_tokens=8, kv_storage_kind="disk",
                device=device, save_path=tmp,
            )
        gpu_out = _run_block_with_storage(
            block, tokens, embed_dim, num_heads,
            q_block_tokens=8, kv_storage_kind="gpu", device=device,
        )

        torch.testing.assert_close(disk_out, gpu_out, rtol=1e-3, atol=1e-4)
