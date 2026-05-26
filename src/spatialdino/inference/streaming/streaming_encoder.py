from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from spatialdino.data import DTYPE_MAPPING
from spatialdino.inference.streaming.storage import HeadMajorStore, StorageKind, TokenStore
from spatialdino.inference.streaming.triton_kernels import (
    TRITON_AVAILABLE,
    finalize_attn_output,
    online_attn_update,
)
from spatialdino.models.layers.rope import RoPECache, apply_rotary_emb

logger = logging.getLogger("streaming_inference")


def _iter_blocks(total: int, block: int) -> Iterable[Tuple[int, int]]:
    if block <= 0:
        raise ValueError(f"block must be > 0, got {block}")
    for start in range(0, total, block):
        end = min(start + block, total)
        yield start, end


class StreamingEncoder:
    def __init__(
        self,
        encoder: torch.nn.Module,
        device: torch.device,
        config: DictConfig,
    ) -> None:
        self.encoder = encoder
        self.device = device
        self.config = config
        self.use_amp = bool(config.use_amp)
        self.device_type = str(config.device_type)
        self.compute_dtype = DTYPE_MAPPING[str(config.dtype)]

        self.storage_kind: StorageKind = str(
            getattr(config, "streaming_storage", "cpu")
        )  # type: ignore
        self.storage_dtype = self.compute_dtype
        self.output_dtype = self.compute_dtype
        self.q_block_tokens = int(getattr(config, "streaming_q_block_tokens", 1024))
        self.kv_block_tokens = int(
            getattr(config, "streaming_kv_block_tokens", self.q_block_tokens)
        )
        self.pin_memory = bool(getattr(config, "streaming_pin_memory", True))
        self.use_triton = bool(getattr(config, "streaming_use_triton", False))
        if self.use_triton and not TRITON_AVAILABLE:
            raise RuntimeError(
                "streaming_use_triton requested but Triton is not available."
            )
        if self.use_triton and self.device.type != "cuda":
            raise RuntimeError("streaming_use_triton requires a CUDA device.")
        self.triton_optimized = bool(
            getattr(config, "streaming_triton_optimized", True)
        )
        self.triton_async_kv_copy = bool(
            getattr(config, "streaming_triton_async_kv_copy", True)
        )
        self.triton_fused_finalize = bool(
            getattr(config, "streaming_triton_fused_finalize", True)
        )
        self.log_q_blocks = bool(getattr(config, "streaming_log_q_blocks", False))
        self.triton_block_m = int(getattr(config, "streaming_triton_block_m", 128))
        self.triton_block_n = int(getattr(config, "streaming_triton_block_n", 128))
        self.triton_block_d = int(getattr(config, "streaming_triton_block_d", 64))
        self.triton_num_warps = int(getattr(config, "streaming_triton_num_warps", 4))
        self.triton_num_stages = int(getattr(config, "streaming_triton_num_stages", 3))
        self._triton_copy_stream: Optional[torch.cuda.Stream] = None
        if self.use_triton and self.triton_async_kv_copy and self.device.type == "cuda":
            self._triton_copy_stream = torch.cuda.Stream(device=self.device)
        self.kv_storage_kind: StorageKind = str(
            getattr(config, "streaming_kv_storage", self.storage_kind)
        )  # type: ignore
        self.kv_storage_dtype = self.storage_dtype

        logger.info(
            "StreamingEncoder: storage=%s, kv_storage=%s, use_triton=%s, "
            "triton_optimized=%s, q_block_tokens=%d, kv_block_tokens=%d, "
            "triton_blocks=(M=%d, N=%d, D=%d)",
            self.storage_kind,
            self.kv_storage_kind,
            self.use_triton,
            self.triton_optimized,
            self.q_block_tokens,
            self.kv_block_tokens,
            self.triton_block_m,
            self.triton_block_n,
            self.triton_block_d,
        )

        self.tmp_dir: Optional[Path] = None
        if self.storage_kind == "disk" or self.kv_storage_kind == "disk":
            save_path = getattr(config, "save_path", None)
            if save_path is None:
                raise ValueError("save_path is required for disk storage.")
            self.tmp_dir = Path(str(save_path)).joinpath("tmp")
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
            if self.storage_kind == "disk":
                if self.storage_dtype == torch.bfloat16:
                    logger.warning(
                        "Disk storage does not support bf16; using fp16 for storage."
                    )
                    self.storage_dtype = torch.float16
                    if self.output_dtype == torch.bfloat16:
                        self.output_dtype = torch.float16
            if (
                self.kv_storage_kind == "disk"
                and self.kv_storage_dtype == torch.bfloat16
            ):
                logger.warning(
                    "Disk storage does not support bf16; using fp16 for KV storage."
                )
                self.kv_storage_dtype = torch.float16

    def predict(
        self,
        volume: torch.Tensor,
        vol_metadata: dict,
        vit_feat: str = "patch_attn",
        norm_feat: str = "prenorm",
        return_special_tokens: bool = False,
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]":
        if vit_feat != "patch_attn":
            raise ValueError("Streaming route only supports vit_feat='patch_attn'.")
        if volume.ndim == 5:
            if volume.shape[0] != 1:
                raise ValueError("Streaming route expects batch size = 1.")
            volume = volume.squeeze(0)
        if volume.ndim != 4:
            raise ValueError(f"Expected volume shape [C, Z, Y, X], got {volume.shape}")

        _, z, y, x = volume.shape
        patch_size = self.encoder.patch_size
        stride = self.encoder.stride
        if tuple(stride) != tuple(patch_size):
            raise ValueError(
                "Streaming inference currently requires stride == patch_size."
            )
        if z % patch_size[0] != 0 or y % patch_size[1] != 0 or x % patch_size[2] != 0:
            raise ValueError("Volume shape must be divisible by patch_size.")

        z0 = 1 + (z - patch_size[0]) // stride[0]
        y0 = 1 + (y - patch_size[1]) // stride[1]
        x0 = 1 + (x - patch_size[2]) // stride[2]
        grid_size = (z0, y0, x0)

        num_special = (
            1
            + int(self.encoder.num_register_tokens)
            + int(self.encoder.num_tt_register_tokens)
        )
        num_patches = z0 * y0 * x0
        total_tokens = num_special + num_patches
        embed_dim = int(self.encoder.embed_dim)
        num_heads = int(self.encoder.num_heads)

        x_store_a = self._create_store(
            name="tokens_a",
            shape=(total_tokens, embed_dim),
            num_special=num_special,
            grid_size=grid_size,
            dtype=self.storage_dtype,
        )
        x_store_b = self._create_store(
            name="tokens_b",
            shape=(total_tokens, embed_dim),
            num_special=num_special,
            grid_size=grid_size,
            dtype=self.storage_dtype,
        )

        logger.info(
            "Streaming inference: volume %s, %d tokens (%d patches + %d special), %d blocks",
            list(volume.shape[1:]),
            total_tokens,
            num_patches,
            num_special,
            len(self.encoder.blocks),
        )

        self._init_special_tokens(x_store_a, embed_dim)
        self._stream_patch_embed(volume, x_store_a, vol_metadata)
        if self.encoder.use_pos_embed:
            self._apply_pos_embed(x_store_a, z, y, x, num_special, embed_dim)

        # Build RoPE cache (CLS + register tokens get identity rotation)
        rope = self.encoder._build_rope_cache(
            z,
            y,
            x,
            device=self.device,
            dtype=self.compute_dtype,
            num_prefix_tokens=num_special,
        )

        x_in = x_store_a
        x_out = x_store_b
        blocks = list(self.encoder.blocks)
        last_index = len(blocks) - 1
        num_blocks = len(blocks)

        for idx, block in enumerate(blocks):
            logger.info("Block %d/%d", idx + 1, num_blocks)
            if idx < last_index:
                self._run_block(block, x_in, x_out, total_tokens, rope=rope)
                x_in, x_out = x_out, x_in
            else:
                out_store = self._run_last_block(
                    block,
                    x_in,
                    x_out,
                    total_tokens,
                    embed_dim,
                    num_heads,
                    norm_feat=norm_feat,
                    rope=rope,
                )

        logger.info("Streaming inference complete.")
        patch_tokens = out_store.tensor[num_special:].view(
            z0, y0, x0, embed_dim + num_heads
        )
        patch_tokens = patch_tokens.permute(3, 0, 1, 2).unsqueeze(0)
        if out_store.storage_kind == "gpu":
            patch_tokens = patch_tokens.cpu()
        patch_tokens._streaming_store = out_store  # type: ignore[attr-defined]

        if not return_special_tokens:
            return patch_tokens

        # Extract CLS and register feature vectors (embed_dim dims only, not attn cols).
        # out_store rows: [CLS, reg_0..reg_n, tt_reg_0..tt_reg_m, patch_0..patch_N]
        special_feats = out_store.tensor[:num_special, :embed_dim].cpu().float()
        cls_token = special_feats[0]  # [embed_dim]
        num_regs = (
            int(self.encoder.num_register_tokens)
            + int(self.encoder.num_tt_register_tokens)
        )
        register_tokens: Optional[torch.Tensor] = (
            special_feats[1 : 1 + num_regs] if num_regs > 0 else None
        )
        return patch_tokens, cls_token, register_tokens

    def _apply_pos_embed(
        self,
        store: TokenStore,
        vol_z: int,
        vol_y: int,
        vol_x: int,
        num_special: int,
        embed_dim: int,
    ) -> None:
        """Add interpolated positional embeddings to CLS and patch tokens in the store.

        Register tokens are intentionally skipped — they receive no positional
        encoding in the standard DINOv2 forward pass either.
        """
        dummy = torch.zeros(
            1, 1, embed_dim, device=self.device, dtype=self.compute_dtype
        )
        with torch.inference_mode():
            with torch.amp.autocast(
                enabled=self.use_amp,
                dtype=self.compute_dtype,
                device_type=self.device_type,
            ):
                pos_embed = self.encoder.interpolate_pos_encoding(
                    self.encoder.pos_embed, dummy, vol_z, vol_y, vol_x
                )  # [1, 1+num_patches, embed_dim]
            pos_embed = pos_embed[0].to(device=self.device, dtype=self.compute_dtype)

            # CLS token (index 0)
            cls_tok = store.read(0, 1, self.device, self.compute_dtype)
            store.write(0, 1, (cls_tok + pos_embed[0:1]).to(store.tensor.dtype))

            # Patch tokens (indices num_special onwards)
            total_patches = pos_embed.shape[0] - 1
            for start, end in _iter_blocks(total_patches, self.q_block_tokens):
                tokens = store.read(
                    num_special + start, num_special + end, self.device, self.compute_dtype
                )
                tokens = tokens + pos_embed[1 + start : 1 + end]
                store.write(num_special + start, num_special + end, tokens.to(store.tensor.dtype))

    def _create_store(
        self,
        name: str,
        shape: Tuple[int, int],
        num_special: int,
        grid_size: Tuple[int, int, int],
        dtype: torch.dtype,
        storage_kind: Optional[StorageKind] = None,
    ) -> TokenStore:
        storage_kind = storage_kind or self.storage_kind
        if storage_kind == "disk":
            assert self.tmp_dir is not None
            path = self.tmp_dir.joinpath(f"{name}_{os.getpid()}.mmap")
        else:
            path = None
        return TokenStore.create(
            shape=shape,
            dtype=dtype,
            storage_kind=storage_kind,
            num_special=num_special,
            grid_size=grid_size,
            device=self.device,
            pin_memory=self.pin_memory,
            path=path,
        )

    def _create_head_major_store(
        self,
        name: str,
        shape: Tuple[int, int, int],
        dtype: torch.dtype,
        storage_kind: Optional[StorageKind] = None,
    ) -> HeadMajorStore:
        storage_kind = storage_kind or self.kv_storage_kind
        if storage_kind == "disk":
            assert self.tmp_dir is not None
            path = self.tmp_dir.joinpath(f"{name}_{os.getpid()}.mmap")
        else:
            path = None
        return HeadMajorStore.create(
            shape=shape,
            dtype=dtype,
            storage_kind=storage_kind,
            device=self.device,
            pin_memory=self.pin_memory,
            path=path,
        )

    def _init_special_tokens(self, store: TokenStore, embed_dim: int) -> None:
        cls = self.encoder.cls_token[0, 0].detach()
        store.write(0, 1, cls.view(1, embed_dim))
        offset = 1
        # Write learnable register tokens
        if self.encoder.register_tokens is not None:
            regs = self.encoder.register_tokens[0].detach()
            store.write(offset, offset + regs.shape[0], regs)
            offset += regs.shape[0]
        # Write test-time register tokens
        if self.encoder.tt_register_tokens is not None:
            tt_regs = self.encoder.tt_register_tokens[0].detach()
            store.write(offset, offset + tt_regs.shape[0], tt_regs)

    def _stream_patch_embed(
        self, volume: torch.Tensor, store: TokenStore, vol_metadata: dict
    ) -> None:
        patch_size = self.encoder.patch_size
        chunk_size = getattr(self.config, "streaming_patch_chunk_size", None)
        if chunk_size is None:
            chunk_size = tuple(int(v) for v in vol_metadata["chunk_size"])
        else:
            chunk_size = tuple(int(v) for v in chunk_size)

        if any(c % p != 0 for c, p in zip(chunk_size, patch_size)):
            raise ValueError(
                "streaming_patch_chunk_size must be divisible by patch_size."
            )

        _, z, y, x = volume.shape
        pz, py, px = patch_size

        with torch.inference_mode():
            for z0 in range(0, z, chunk_size[0]):
                z1 = min(z0 + chunk_size[0], z)
                for y0 in range(0, y, chunk_size[1]):
                    y1 = min(y0 + chunk_size[1], y)
                    for x0 in range(0, x, chunk_size[2]):
                        x1 = min(x0 + chunk_size[2], x)
                        chunk = volume[:, z0:z1, y0:y1, x0:x1]
                        if any(
                            dim % p != 0 for dim, p in zip(chunk.shape[1:], patch_size)
                        ):
                            raise ValueError(
                                "Chunk dims must be divisible by patch_size."
                            )
                        chunk = chunk.unsqueeze(0).to(
                            self.device, non_blocking=self.pin_memory
                        )
                        with torch.amp.autocast(
                            enabled=self.use_amp,
                            dtype=self.compute_dtype,
                            device_type=self.device_type,
                        ):
                            tokens = self.encoder.patch_embed(chunk)
                        z_p = (z1 - z0) // pz
                        y_p = (y1 - y0) // py
                        x_p = (x1 - x0) // px
                        tokens = tokens.view(1, z_p, y_p, x_p, -1).squeeze(0)
                        tokens = tokens.to(
                            device=store.tensor.device, dtype=store.tensor.dtype
                        )
                        store.patch_view[
                            z0 // pz : z1 // pz,
                            y0 // py : y1 // py,
                            x0 // px : x1 // px,
                        ] = tokens

    def _run_block(
        self,
        block: torch.nn.Module,
        x_in: TokenStore,
        x_out: TokenStore,
        total_tokens: int,
        rope: Optional[RoPECache] = None,
    ) -> None:
        attn = block.attn
        num_heads = attn.num_heads
        head_dim = attn.qkv.in_features // num_heads
        embed_dim = num_heads * head_dim
        scale = attn.scale

        if self.use_triton and self.triton_optimized:
            self._run_block_triton_optimized(
                block=block,
                x_in=x_in,
                x_out=x_out,
                total_tokens=total_tokens,
                embed_dim=embed_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                rope=rope,
            )
            return

        q_store, k_store, v_store = self._precompute_qkv(
            block=block,
            x_in=x_in,
            total_tokens=total_tokens,
            embed_dim=embed_dim,
            num_heads=num_heads,
            rope=rope,
        )

        use_sdpa = k_store.storage_kind == "gpu" and not self.use_triton
        k_all: Optional[torch.Tensor] = None
        v_all: Optional[torch.Tensor] = None
        if use_sdpa:
            k_all = (
                k_store.tensor.view(total_tokens, num_heads, head_dim)
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
                .to(self.compute_dtype)
            )
            v_all = (
                v_store.tensor.view(total_tokens, num_heads, head_dim)
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
                .to(self.compute_dtype)
            )

        num_q_blocks = (total_tokens + self.q_block_tokens - 1) // self.q_block_tokens
        with torch.inference_mode():
            for q_idx, (q_start, q_end) in enumerate(
                _iter_blocks(total_tokens, self.q_block_tokens)
            ):
                if self.log_q_blocks and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "  Q block %d/%d [%d:%d]",
                        q_idx + 1,
                        num_q_blocks,
                        q_start,
                        q_end,
                    )
                x_q = x_in.read(q_start, q_end, self.device, self.compute_dtype)
                q_len = q_end - q_start

                # Read precomputed Q
                q_data = q_store.read(q_start, q_end, self.device, self.compute_dtype)
                q = (
                    q_data.view(q_len, num_heads, head_dim)
                    .permute(1, 0, 2)
                    .unsqueeze(0)
                )  # [1, H, q_len, D]

                if use_sdpa:
                    assert k_all is not None and v_all is not None
                    with torch.amp.autocast(
                        enabled=self.use_amp,
                        dtype=self.compute_dtype,
                        device_type=self.device_type,
                    ):
                        out = F.scaled_dot_product_attention(
                            q.contiguous(),
                            k_all,
                            v_all,
                            scale=scale,
                        )
                    out = out.transpose(1, 2).reshape(1, q_len, embed_dim)
                elif self.use_triton:
                    q_3d = q.squeeze(0).contiguous()  # [H, q_len, D]
                    m = torch.full(
                        (num_heads, q_len),
                        float("-inf"),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    row_sum = torch.zeros(
                        (num_heads, q_len),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    out_acc = torch.zeros(
                        (num_heads, q_len, head_dim),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    for kv_start, kv_end in _iter_blocks(
                        total_tokens, self.kv_block_tokens
                    ):
                        k_block = k_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        v_block = v_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        k_3d = (
                            k_block.view(-1, num_heads, head_dim)
                            .permute(1, 0, 2)
                            .contiguous()
                        )
                        v_3d = (
                            v_block.view(-1, num_heads, head_dim)
                            .permute(1, 0, 2)
                            .contiguous()
                        )
                        online_attn_update(
                            q_3d,
                            k_3d,
                            v_3d,
                            m,
                            row_sum,
                            out_acc,
                            scale,
                            block_m=self.triton_block_m,
                            block_n=self.triton_block_n,
                            block_d=self.triton_block_d,
                            num_warps=self.triton_num_warps,
                            num_stages=self.triton_num_stages,
                        )
                    out = out_acc / row_sum.unsqueeze(-1)
                    out = out.to(dtype=self.compute_dtype)
                    out = out.transpose(0, 1).reshape(1, q_len, embed_dim)
                else:
                    # Manual online softmax — matmuls in compute_dtype for
                    # tensor-core utilisation, softmax accumulators in float32.
                    m = torch.full(
                        (1, num_heads, q_len),
                        float("-inf"),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    row_sum = torch.zeros(
                        (1, num_heads, q_len),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    out_acc = torch.zeros(
                        (1, num_heads, q_len, head_dim),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    for kv_start, kv_end in _iter_blocks(
                        total_tokens, self.kv_block_tokens
                    ):
                        k_block = k_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        v_block = v_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        kv_len = kv_end - kv_start
                        k = k_block.view(1, kv_len, num_heads, head_dim).permute(
                            0, 2, 1, 3
                        )
                        v = v_block.view(1, kv_len, num_heads, head_dim).permute(
                            0, 2, 1, 3
                        )
                        scores = (torch.matmul(q, k.transpose(-2, -1)) * scale).float()
                        block_max = scores.max(dim=-1).values
                        m_new = torch.maximum(m, block_max)
                        exp_m = torch.exp(m - m_new)
                        exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
                        row_sum = exp_m * row_sum + exp_scores.sum(dim=-1)
                        out_acc = (
                            exp_m.unsqueeze(-1) * out_acc
                            + torch.matmul(exp_scores.to(self.compute_dtype), v).float()
                        )
                        m = m_new
                    out = out_acc / row_sum.unsqueeze(-1)
                    out = out.to(dtype=self.compute_dtype)
                    out = out.transpose(1, 2).reshape(1, q_len, embed_dim)

                with torch.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.compute_dtype,
                    device_type=self.device_type,
                ):
                    x_attn = attn.proj(out)
                    x_attn = attn.proj_drop(x_attn)
                    x_attn = block.ls1(x_attn)
                    x_new = x_q + x_attn
                    x_new = x_new + block.ls2(block.mlp(block.norm2(x_new)))

                x_out.write(q_start, q_end, x_new.squeeze(0))

        del q_store, k_store, v_store

    def _run_last_block(
        self,
        block: torch.nn.Module,
        x_in: TokenStore,
        x_out: TokenStore,
        total_tokens: int,
        embed_dim: int,
        num_heads: int,
        norm_feat: str,
        rope: Optional[RoPECache] = None,
    ) -> TokenStore:
        attn = block.attn
        head_dim = attn.qkv.in_features // num_heads
        scale = attn.scale
        embed_dim = num_heads * head_dim

        if self.use_triton and self.triton_optimized:
            return self._run_last_block_triton_optimized(
                block=block,
                x_in=x_in,
                x_out=x_out,
                total_tokens=total_tokens,
                embed_dim=embed_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                norm_feat=norm_feat,
                rope=rope,
            )

        q_store, k_store, v_store = self._precompute_qkv(
            block=block,
            x_in=x_in,
            total_tokens=total_tokens,
            embed_dim=embed_dim,
            num_heads=num_heads,
            rope=rope,
        )

        # SDPA fast path: K/V on GPU and Triton not forced → use SDPA
        use_sdpa = k_store.storage_kind == "gpu" and not self.use_triton
        k_all: Optional[torch.Tensor] = None
        v_all: Optional[torch.Tensor] = None
        if use_sdpa:
            k_all = (
                k_store.tensor.view(total_tokens, num_heads, head_dim)
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
                .to(self.compute_dtype)
            )
            v_all = (
                v_store.tensor.view(total_tokens, num_heads, head_dim)
                .permute(1, 0, 2)
                .unsqueeze(0)
                .contiguous()
                .to(self.compute_dtype)
            )

        cls_context: Optional[torch.Tensor] = None
        num_q_blocks = (total_tokens + self.q_block_tokens - 1) // self.q_block_tokens
        with torch.inference_mode():
            for q_idx, (q_start, q_end) in enumerate(
                _iter_blocks(total_tokens, self.q_block_tokens)
            ):
                if self.log_q_blocks and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "  Q block %d/%d [%d:%d]",
                        q_idx + 1,
                        num_q_blocks,
                        q_start,
                        q_end,
                    )
                x_q = x_in.read(q_start, q_end, self.device, self.compute_dtype)
                q_len = q_end - q_start

                # Read precomputed Q
                q_data = q_store.read(q_start, q_end, self.device, self.compute_dtype)
                q = (
                    q_data.view(q_len, num_heads, head_dim)
                    .permute(1, 0, 2)
                    .unsqueeze(0)
                )  # [1, H, q_len, D]

                if use_sdpa:
                    assert k_all is not None and v_all is not None
                    with torch.amp.autocast(
                        enabled=self.use_amp,
                        dtype=self.compute_dtype,
                        device_type=self.device_type,
                    ):
                        out = F.scaled_dot_product_attention(
                            q.contiguous(),
                            k_all,
                            v_all,
                            scale=scale,
                        )
                    if q_start <= 0 < q_end:
                        cls_index = 0 - q_start
                        cls_context = out[:, :, cls_index : cls_index + 1, :].clone()
                    out = out.transpose(1, 2).reshape(1, q_len, embed_dim)
                elif self.use_triton:
                    q_3d = q.squeeze(0).contiguous()  # [H, q_len, D]
                    m = torch.full(
                        (num_heads, q_len),
                        float("-inf"),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    row_sum = torch.zeros(
                        (num_heads, q_len),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    out_acc = torch.zeros(
                        (num_heads, q_len, head_dim),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    for kv_start, kv_end in _iter_blocks(
                        total_tokens, self.kv_block_tokens
                    ):
                        k_block = k_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        v_block = v_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        k_3d = (
                            k_block.view(-1, num_heads, head_dim)
                            .permute(1, 0, 2)
                            .contiguous()
                        )
                        v_3d = (
                            v_block.view(-1, num_heads, head_dim)
                            .permute(1, 0, 2)
                            .contiguous()
                        )
                        online_attn_update(
                            q_3d,
                            k_3d,
                            v_3d,
                            m,
                            row_sum,
                            out_acc,
                            scale,
                            block_m=self.triton_block_m,
                            block_n=self.triton_block_n,
                            block_d=self.triton_block_d,
                            num_warps=self.triton_num_warps,
                            num_stages=self.triton_num_stages,
                        )
                    out = out_acc / row_sum.unsqueeze(-1)
                    if q_start <= 0 < q_end:
                        cls_index = 0 - q_start
                        cls_context = (
                            out[:, cls_index : cls_index + 1, :].clone().unsqueeze(0)
                        )
                    out = out.to(dtype=self.compute_dtype)
                    out = out.transpose(0, 1).reshape(1, q_len, embed_dim)
                else:
                    # Manual online softmax — matmuls in compute_dtype for
                    # tensor-core utilisation, softmax accumulators in float32.
                    m = torch.full(
                        (1, num_heads, q_len),
                        float("-inf"),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    row_sum = torch.zeros(
                        (1, num_heads, q_len),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    out_acc = torch.zeros(
                        (1, num_heads, q_len, head_dim),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    for kv_start, kv_end in _iter_blocks(
                        total_tokens, self.kv_block_tokens
                    ):
                        k_block = k_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        v_block = v_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        kv_len = kv_end - kv_start
                        k = k_block.view(1, kv_len, num_heads, head_dim).permute(
                            0, 2, 1, 3
                        )
                        v = v_block.view(1, kv_len, num_heads, head_dim).permute(
                            0, 2, 1, 3
                        )
                        scores = (torch.matmul(q, k.transpose(-2, -1)) * scale).float()
                        block_max = scores.max(dim=-1).values
                        m_new = torch.maximum(m, block_max)
                        exp_m = torch.exp(m - m_new)
                        exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
                        row_sum = exp_m * row_sum + exp_scores.sum(dim=-1)
                        out_acc = (
                            exp_m.unsqueeze(-1) * out_acc
                            + torch.matmul(exp_scores.to(self.compute_dtype), v).float()
                        )
                        m = m_new
                    out = out_acc / row_sum.unsqueeze(-1)
                    if q_start <= 0 < q_end:
                        cls_index = 0 - q_start
                        cls_context = out[:, :, cls_index : cls_index + 1, :].clone()
                    out = out.to(dtype=self.compute_dtype)
                    out = out.transpose(1, 2).reshape(1, q_len, embed_dim)

                with torch.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.compute_dtype,
                    device_type=self.device_type,
                ):
                    x_attn = attn.proj(out)
                    x_attn = attn.proj_drop(x_attn)
                    x_attn = block.ls1(x_attn)
                    x_new = x_q + x_attn
                    x_new = x_new + block.ls2(block.mlp(block.norm2(x_new)))

                x_out.write(q_start, q_end, x_new.squeeze(0))

        if cls_context is None:
            raise RuntimeError("Failed to capture CLS context for patch_attn.")

        if norm_feat == "norm":
            self._apply_norm(self.encoder.norm, x_out, total_tokens)

        attn_store = self._create_store(
            name="attn_store",
            shape=(total_tokens, num_heads),
            num_special=x_out.num_special,
            grid_size=x_out.grid_size,
            dtype=self.output_dtype,
        )

        # Compute cls_attn: CLS attention output × V^T
        with torch.inference_mode():
            if use_sdpa:
                assert v_all is not None
                # Single matmul with all V already on GPU
                cls_attn = torch.matmul(
                    cls_context.float(), v_all.float().transpose(-2, -1)
                )
                cls_attn = cls_attn.squeeze(2).transpose(1, 2).contiguous()
                attn_store.write(0, total_tokens, cls_attn.squeeze(0))
            else:
                for kv_start, kv_end in _iter_blocks(
                    total_tokens, self.kv_block_tokens
                ):
                    v_block = v_store.read(
                        kv_start, kv_end, self.device, self.compute_dtype
                    )
                    kv_len = kv_end - kv_start
                    v = v_block.view(1, kv_len, num_heads, head_dim).permute(0, 2, 1, 3)
                    cls_attn = torch.matmul(
                        cls_context.float(), v.float().transpose(-2, -1)
                    )
                    cls_attn = cls_attn.squeeze(2).transpose(1, 2).contiguous()
                    attn_store.write(kv_start, kv_end, cls_attn.squeeze(0))

        del q_store, k_store, v_store

        final_store = self._create_store(
            name="final_store",
            shape=(total_tokens, embed_dim + num_heads),
            num_special=x_out.num_special,
            grid_size=x_out.grid_size,
            dtype=self.output_dtype,
        )

        combine_device = (
            self.device if self.storage_kind == "gpu" else torch.device("cpu")
        )
        for start, end in _iter_blocks(total_tokens, self.q_block_tokens):
            feats = x_out.read(start, end, combine_device, self.output_dtype)
            attn = attn_store.read(start, end, combine_device, self.output_dtype)
            combined = torch.cat([feats, attn], dim=-1)
            final_store.write(start, end, combined)

        final_store.flush()
        attn_store.flush()
        x_out.flush()
        return final_store

    def _run_block_triton_optimized(
        self,
        block: torch.nn.Module,
        x_in: TokenStore,
        x_out: TokenStore,
        total_tokens: int,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        rope: Optional[RoPECache] = None,
    ) -> None:
        attn = block.attn
        q_store, k_store, v_store = self._precompute_qkv_head_major(
            block=block,
            x_in=x_in,
            total_tokens=total_tokens,
            embed_dim=embed_dim,
            num_heads=num_heads,
            rope=rope,
        )

        num_q_blocks = (total_tokens + self.q_block_tokens - 1) // self.q_block_tokens
        with torch.inference_mode():
            for q_idx, (q_start, q_end) in enumerate(
                _iter_blocks(total_tokens, self.q_block_tokens)
            ):
                if self.log_q_blocks and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "  Q block %d/%d [%d:%d]",
                        q_idx + 1,
                        num_q_blocks,
                        q_start,
                        q_end,
                    )
                x_q = x_in.read(q_start, q_end, self.device, self.compute_dtype)
                q_len = q_end - q_start
                q_3d = q_store.read_tokens(
                    q_start, q_end, self.device, self.compute_dtype
                )
                out_head, _, _ = self._run_triton_attention_head_major(
                    q_3d=q_3d,
                    k_store=k_store,
                    v_store=v_store,
                    total_tokens=total_tokens,
                    scale=scale,
                )
                out = out_head.transpose(0, 1).reshape(1, q_len, embed_dim)

                with torch.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.compute_dtype,
                    device_type=self.device_type,
                ):
                    x_attn = attn.proj(out)
                    x_attn = attn.proj_drop(x_attn)
                    x_attn = block.ls1(x_attn)
                    x_new = x_q + x_attn
                    x_new = x_new + block.ls2(block.mlp(block.norm2(x_new)))

                x_out.write(q_start, q_end, x_new.squeeze(0))

        del q_store, k_store, v_store

    def _run_last_block_triton_optimized(
        self,
        block: torch.nn.Module,
        x_in: TokenStore,
        x_out: TokenStore,
        total_tokens: int,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        norm_feat: str,
        rope: Optional[RoPECache] = None,
    ) -> TokenStore:
        attn = block.attn
        q_store, k_store, v_store = self._precompute_qkv_head_major(
            block=block,
            x_in=x_in,
            total_tokens=total_tokens,
            embed_dim=embed_dim,
            num_heads=num_heads,
            rope=rope,
        )

        cls_context: Optional[torch.Tensor] = None
        num_q_blocks = (total_tokens + self.q_block_tokens - 1) // self.q_block_tokens
        with torch.inference_mode():
            for q_idx, (q_start, q_end) in enumerate(
                _iter_blocks(total_tokens, self.q_block_tokens)
            ):
                if self.log_q_blocks and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "  Q block %d/%d [%d:%d]",
                        q_idx + 1,
                        num_q_blocks,
                        q_start,
                        q_end,
                    )
                x_q = x_in.read(q_start, q_end, self.device, self.compute_dtype)
                q_len = q_end - q_start
                q_3d = q_store.read_tokens(
                    q_start, q_end, self.device, self.compute_dtype
                )
                out_head, out_acc, row_sum = self._run_triton_attention_head_major(
                    q_3d=q_3d,
                    k_store=k_store,
                    v_store=v_store,
                    total_tokens=total_tokens,
                    scale=scale,
                )
                if q_start <= 0 < q_end:
                    cls_index = 0 - q_start
                    cls_context = (
                        out_acc[:, cls_index : cls_index + 1, :]
                        / row_sum[:, cls_index : cls_index + 1].unsqueeze(-1)
                    ).clone().unsqueeze(0)
                out = out_head.transpose(0, 1).reshape(1, q_len, embed_dim)

                with torch.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.compute_dtype,
                    device_type=self.device_type,
                ):
                    x_attn = attn.proj(out)
                    x_attn = attn.proj_drop(x_attn)
                    x_attn = block.ls1(x_attn)
                    x_new = x_q + x_attn
                    x_new = x_new + block.ls2(block.mlp(block.norm2(x_new)))

                x_out.write(q_start, q_end, x_new.squeeze(0))

        if cls_context is None:
            raise RuntimeError("Failed to capture CLS context for patch_attn.")

        if norm_feat == "norm":
            self._apply_norm(self.encoder.norm, x_out, total_tokens)

        attn_store = self._create_store(
            name="attn_store",
            shape=(total_tokens, num_heads),
            num_special=x_out.num_special,
            grid_size=x_out.grid_size,
            dtype=self.output_dtype,
        )

        with torch.inference_mode():
            if v_store.storage_kind == "gpu":
                v_all = v_store.tensor.to(dtype=self.compute_dtype)
                cls_attn = torch.matmul(
                    cls_context.float(), v_all.float().transpose(-2, -1)
                )
                cls_attn = cls_attn.squeeze(2).transpose(1, 2).contiguous()
                attn_store.write(0, total_tokens, cls_attn.squeeze(0))
            else:
                for kv_start, kv_end in _iter_blocks(
                    total_tokens, self.kv_block_tokens
                ):
                    v_3d = v_store.read_tokens(
                        kv_start, kv_end, self.device, self.compute_dtype
                    )
                    cls_attn = torch.matmul(
                        cls_context.float(), v_3d.float().transpose(-2, -1)
                    )
                    cls_attn = cls_attn.squeeze(2).transpose(1, 2).contiguous()
                    attn_store.write(kv_start, kv_end, cls_attn.squeeze(0))

        del q_store, k_store, v_store

        final_store = self._create_store(
            name="final_store",
            shape=(total_tokens, embed_dim + num_heads),
            num_special=x_out.num_special,
            grid_size=x_out.grid_size,
            dtype=self.output_dtype,
        )

        combine_device = (
            self.device if self.storage_kind == "gpu" else torch.device("cpu")
        )
        for start, end in _iter_blocks(total_tokens, self.q_block_tokens):
            feats = x_out.read(start, end, combine_device, self.output_dtype)
            attn = attn_store.read(start, end, combine_device, self.output_dtype)
            combined = torch.cat([feats, attn], dim=-1)
            final_store.write(start, end, combined)

        final_store.flush()
        attn_store.flush()
        x_out.flush()
        return final_store

    def _iter_head_major_kv_blocks(
        self,
        k_store: HeadMajorStore,
        v_store: HeadMajorStore,
        total_tokens: int,
    ) -> Iterable[Tuple[int, int, torch.Tensor, torch.Tensor]]:
        if k_store.storage_kind == "gpu":
            yield (
                0,
                total_tokens,
                k_store.tensor.to(dtype=self.compute_dtype),
                v_store.tensor.to(dtype=self.compute_dtype),
            )
            return

        blocks = list(_iter_blocks(total_tokens, self.kv_block_tokens))
        if (
            not self.triton_async_kv_copy
            or self.device.type != "cuda"
            or len(blocks) <= 1
            or self._triton_copy_stream is None
        ):
            for start, end in blocks:
                yield (
                    start,
                    end,
                    k_store.read_tokens(start, end, self.device, self.compute_dtype),
                    v_store.read_tokens(start, end, self.device, self.compute_dtype),
                )
            return

        copy_stream = self._triton_copy_stream
        compute_stream = torch.cuda.current_stream(self.device)
        num_heads, _, head_dim = k_store.tensor.shape
        max_block = max(end - start for start, end in blocks)
        buffers = [
            (
                torch.empty(
                    (num_heads, max_block, head_dim),
                    device=self.device,
                    dtype=self.compute_dtype,
                ),
                torch.empty(
                    (num_heads, max_block, head_dim),
                    device=self.device,
                    dtype=self.compute_dtype,
                ),
            ),
            (
                torch.empty(
                    (num_heads, max_block, head_dim),
                    device=self.device,
                    dtype=self.compute_dtype,
                ),
                torch.empty(
                    (num_heads, max_block, head_dim),
                    device=self.device,
                    dtype=self.compute_dtype,
                ),
            ),
        ]

        def schedule(
            index: int,
            slot: int,
        ) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.cuda.Event, int]:
            start, end = blocks[index]
            length = end - start
            k_block = buffers[slot][0][:, :length, :]
            v_block = buffers[slot][1][:, :length, :]
            k_source = k_store.tensor[:, start:end, :]
            v_source = v_store.tensor[:, start:end, :]
            with torch.cuda.stream(copy_stream):
                k_block.copy_(k_source, non_blocking=k_store.pin_memory)
                v_block.copy_(v_source, non_blocking=v_store.pin_memory)
                event = torch.cuda.Event()
                event.record(copy_stream)
            return start, end, k_block, v_block, event, slot

        pending = schedule(0, 0)
        for index in range(len(blocks)):
            start, end, k_block, v_block, event, slot = pending
            next_pending = (
                schedule(index + 1, 1 - slot) if index + 1 < len(blocks) else None
            )
            compute_stream.wait_event(event)
            yield start, end, k_block, v_block
            if next_pending is not None:
                pending = next_pending

    def _finalize_triton_output(
        self,
        out_acc: torch.Tensor,
        row_sum: torch.Tensor,
    ) -> torch.Tensor:
        if self.triton_fused_finalize:
            return finalize_attn_output(
                out_acc,
                row_sum,
                dtype=self.compute_dtype,
                block_m=self.triton_block_m,
                block_d=self.triton_block_d,
                num_warps=self.triton_num_warps,
            )
        out = out_acc / row_sum.unsqueeze(-1)
        return out.to(dtype=self.compute_dtype)

    def _run_triton_attention_head_major(
        self,
        q_3d: torch.Tensor,
        k_store: HeadMajorStore,
        v_store: HeadMajorStore,
        total_tokens: int,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_heads, q_len, head_dim = q_3d.shape
        m = torch.full(
            (num_heads, q_len),
            float("-inf"),
            device=self.device,
            dtype=torch.float32,
        )
        row_sum = torch.zeros(
            (num_heads, q_len),
            device=self.device,
            dtype=torch.float32,
        )
        out_acc = torch.zeros(
            (num_heads, q_len, head_dim),
            device=self.device,
            dtype=torch.float32,
        )

        for _, _, k_3d, v_3d in self._iter_head_major_kv_blocks(
            k_store, v_store, total_tokens
        ):
            online_attn_update(
                q_3d,
                k_3d,
                v_3d,
                m,
                row_sum,
                out_acc,
                scale,
                block_m=self.triton_block_m,
                block_n=self.triton_block_n,
                block_d=self.triton_block_d,
                num_warps=self.triton_num_warps,
                num_stages=self.triton_num_stages,
            )

        out = self._finalize_triton_output(out_acc, row_sum)
        return out, out_acc, row_sum

    def _precompute_qkv(
        self,
        block: torch.nn.Module,
        x_in: TokenStore,
        total_tokens: int,
        embed_dim: int,
        num_heads: int,
        rope: Optional[RoPECache] = None,
    ) -> Tuple[TokenStore, TokenStore, TokenStore]:
        head_dim = embed_dim // num_heads
        attn = block.attn

        block_tag = f"{id(block)}"
        q_store = self._create_store(
            name=f"qkv_q_{block_tag}",
            shape=(total_tokens, embed_dim),
            num_special=x_in.num_special,
            grid_size=x_in.grid_size,
            dtype=self.kv_storage_dtype,
            storage_kind=self.kv_storage_kind,
        )
        k_store = self._create_store(
            name=f"qkv_k_{block_tag}",
            shape=(total_tokens, embed_dim),
            num_special=x_in.num_special,
            grid_size=x_in.grid_size,
            dtype=self.kv_storage_dtype,
            storage_kind=self.kv_storage_kind,
        )
        v_store = self._create_store(
            name=f"qkv_v_{block_tag}",
            shape=(total_tokens, embed_dim),
            num_special=x_in.num_special,
            grid_size=x_in.grid_size,
            dtype=self.kv_storage_dtype,
            storage_kind=self.kv_storage_kind,
        )

        with torch.inference_mode():
            for start, end in _iter_blocks(total_tokens, self.kv_block_tokens):
                x_block = x_in.read(start, end, self.device, self.compute_dtype)
                with torch.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.compute_dtype,
                    device_type=self.device_type,
                ):
                    x_norm = block.norm1(x_block)
                    qkv = attn.qkv(x_norm)
                qkv = qkv.view(1, -1, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q_head = qkv[0]  # [1, num_heads, seq_len, head_dim]
                k_head = qkv[1]
                if rope is not None:
                    cos, sin = rope
                    cos_slice = cos[start:end].unsqueeze(0).unsqueeze(0)
                    sin_slice = sin[start:end].unsqueeze(0).unsqueeze(0)
                    q_head = apply_rotary_emb(q_head, cos_slice, sin_slice)
                    k_head = apply_rotary_emb(k_head, cos_slice, sin_slice)
                q_flat = q_head.transpose(1, 2).reshape(1, -1, embed_dim).squeeze(0)
                k_flat = k_head.transpose(1, 2).reshape(1, -1, embed_dim).squeeze(0)
                v_flat = qkv[2].transpose(1, 2).reshape(1, -1, embed_dim).squeeze(0)
                q_store.write(start, end, q_flat)
                k_store.write(start, end, k_flat)
                v_store.write(start, end, v_flat)

        q_store.flush()
        k_store.flush()
        v_store.flush()
        return q_store, k_store, v_store

    def _precompute_qkv_head_major(
        self,
        block: torch.nn.Module,
        x_in: TokenStore,
        total_tokens: int,
        embed_dim: int,
        num_heads: int,
        rope: Optional[RoPECache] = None,
    ) -> Tuple[HeadMajorStore, HeadMajorStore, HeadMajorStore]:
        head_dim = embed_dim // num_heads
        if head_dim > self.triton_block_d:
            raise ValueError(
                f"head_dim={head_dim} exceeds streaming_triton_block_d="
                f"{self.triton_block_d}."
            )

        attn = block.attn
        block_tag = f"{id(block)}"
        shape = (num_heads, total_tokens, head_dim)
        q_store = self._create_head_major_store(
            name=f"qkv_q_hm_{block_tag}",
            shape=shape,
            dtype=self.kv_storage_dtype,
            storage_kind=self.kv_storage_kind,
        )
        k_store = self._create_head_major_store(
            name=f"qkv_k_hm_{block_tag}",
            shape=shape,
            dtype=self.kv_storage_dtype,
            storage_kind=self.kv_storage_kind,
        )
        v_store = self._create_head_major_store(
            name=f"qkv_v_hm_{block_tag}",
            shape=shape,
            dtype=self.kv_storage_dtype,
            storage_kind=self.kv_storage_kind,
        )

        with torch.inference_mode():
            for start, end in _iter_blocks(total_tokens, self.kv_block_tokens):
                x_block = x_in.read(start, end, self.device, self.compute_dtype)
                with torch.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.compute_dtype,
                    device_type=self.device_type,
                ):
                    x_norm = block.norm1(x_block)
                    qkv = attn.qkv(x_norm)
                qkv = qkv.view(1, -1, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q_head = qkv[0].squeeze(0)  # [num_heads, seq_len, head_dim]
                k_head = qkv[1].squeeze(0)
                if rope is not None:
                    cos, sin = rope
                    cos_slice = cos[start:end].unsqueeze(0).unsqueeze(0)
                    sin_slice = sin[start:end].unsqueeze(0).unsqueeze(0)
                    q_rot = apply_rotary_emb(qkv[0], cos_slice, sin_slice)
                    k_rot = apply_rotary_emb(qkv[1], cos_slice, sin_slice)
                    q_head = q_rot.squeeze(0)
                    k_head = k_rot.squeeze(0)
                v_head = qkv[2].squeeze(0)
                q_store.write_tokens(start, end, q_head)
                k_store.write_tokens(start, end, k_head)
                v_store.write_tokens(start, end, v_head)

        q_store.flush()
        k_store.flush()
        v_store.flush()
        return q_store, k_store, v_store

    def _apply_norm(
        self, norm: torch.nn.Module, store: TokenStore, total_tokens: int
    ) -> None:
        with torch.inference_mode():
            for start, end in _iter_blocks(total_tokens, self.q_block_tokens):
                block = store.read(start, end, self.device, self.compute_dtype)
                with torch.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.compute_dtype,
                    device_type=self.device_type,
                ):
                    block = norm(block)
                store.write(start, end, block)
