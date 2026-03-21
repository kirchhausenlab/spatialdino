from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from omegaconf import DictConfig

from spatialdino.data import DTYPE_MAPPING
from spatialdino.inference.streaming.storage import StorageKind, TokenStore
from spatialdino.inference.streaming.triton_kernels import (
    TRITON_AVAILABLE,
    online_attn_update,
)

logger = logging.getLogger("streaming_inference")


def _iter_blocks(total: int, block: int) -> Iterable[tuple[int, int]]:
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
        self.kv_block_tokens = self.q_block_tokens
        self.pin_memory = bool(getattr(config, "streaming_pin_memory", True))
        self.precompute_kv = True
        self.use_triton = bool(getattr(config, "streaming_use_triton", False))
        if self.use_triton and not TRITON_AVAILABLE:
            raise RuntimeError(
                "streaming_use_triton requested but Triton is not available."
            )
        self.triton_block_m = int(getattr(config, "streaming_triton_block_m", 128))
        self.triton_block_n = int(getattr(config, "streaming_triton_block_n", 128))
        self.triton_block_d = int(getattr(config, "streaming_triton_block_d", 64))
        self.triton_num_warps = int(getattr(config, "streaming_triton_num_warps", 4))
        self.triton_num_stages = int(getattr(config, "streaming_triton_num_stages", 3))
        self.kv_storage_kind: StorageKind = str(
            getattr(config, "streaming_kv_storage", self.storage_kind)
        )  # type: ignore
        self.kv_storage_dtype = self.storage_dtype

        self.tmp_dir: Path | None = None
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
    ) -> torch.Tensor:
        if vit_feat != "patch_attn":
            raise ValueError("Streaming route only supports vit_feat='patch_attn'.")
        if self.encoder.use_pos_embed:
            raise ValueError(
                "Streaming route currently requires pos_embed_type='none'."
            )
        if volume.ndim == 5:
            if volume.shape[0] != 1:
                raise ValueError("Streaming route expects batch size = 1.")
            volume = volume.squeeze(0)
        if volume.ndim != 4:
            raise ValueError(f"Expected volume shape [C, Z, Y, X], got {volume.shape}")

        _, z, y, x = volume.shape
        patch_size = self.encoder.patch_size
        stride = self.encoder.stride
        if z % patch_size[0] != 0 or y % patch_size[1] != 0 or x % patch_size[2] != 0:
            raise ValueError("Volume shape must be divisible by patch_size.")

        z0 = 1 + (z - patch_size[0]) // stride[0]
        y0 = 1 + (y - patch_size[1]) // stride[1]
        x0 = 1 + (x - patch_size[2]) // stride[2]
        grid_size = (z0, y0, x0)

        num_special = 1 + int(self.encoder.num_tt_register_tokens)
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

        self._init_special_tokens(x_store_a, embed_dim)
        self._stream_patch_embed(volume, x_store_a, vol_metadata)

        x_in = x_store_a
        x_out = x_store_b
        blocks = list(self.encoder.blocks)
        last_index = len(blocks) - 1

        for idx, block in enumerate(blocks):
            if idx < last_index:
                self._run_block(block, x_in, x_out, total_tokens)
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
                )

        patch_tokens = out_store.tensor[num_special:].view(
            z0, y0, x0, embed_dim + num_heads
        )
        patch_tokens = patch_tokens.permute(3, 0, 1, 2).unsqueeze(0)
        if out_store.storage_kind == "gpu":
            patch_tokens = patch_tokens.cpu()
        patch_tokens._streaming_store = out_store  # type: ignore[attr-defined]
        return patch_tokens

    def _create_store(
        self,
        name: str,
        shape: tuple[int, int],
        num_special: int,
        grid_size: tuple[int, int, int],
        dtype: torch.dtype,
        storage_kind: StorageKind | None = None,
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

    def _init_special_tokens(self, store: TokenStore, embed_dim: int) -> None:
        cls = self.encoder.cls_token[0, 0].detach()
        store.write(0, 1, cls.view(1, embed_dim))
        if self.encoder.tt_register_tokens is not None:
            regs = self.encoder.tt_register_tokens[0].detach()
            store.write(1, 1 + regs.shape[0], regs)

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
    ) -> None:
        attn = block.attn
        num_heads = attn.num_heads
        head_dim = attn.qkv.in_features // num_heads
        embed_dim = num_heads * head_dim
        scale = attn.scale

        k_store: TokenStore | None = None
        v_store: TokenStore | None = None
        if self.precompute_kv:
            k_store, v_store = self._precompute_kv(
                block=block,
                x_in=x_in,
                total_tokens=total_tokens,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )

        with torch.inference_mode():
            for q_start, q_end in _iter_blocks(total_tokens, self.q_block_tokens):
                x_q = x_in.read(q_start, q_end, self.device, self.compute_dtype)
                with torch.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.compute_dtype,
                    device_type=self.device_type,
                ):
                    x_q_norm = block.norm1(x_q)
                    qkv_q = attn.qkv(x_q_norm)
                qkv_q = qkv_q.view(1, -1, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q = qkv_q[0]

                q_len = q.shape[2]
                if self.use_triton:
                    m = torch.full(
                        (num_heads, q_len),
                        float("-inf"),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    l = torch.zeros(
                        (num_heads, q_len),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    out = torch.zeros(
                        (num_heads, q_len, head_dim),
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    m = torch.full(
                        (1, num_heads, q_len),
                        float("-inf"),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    l = torch.zeros(
                        (1, num_heads, q_len),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    out = torch.zeros(
                        (1, num_heads, q_len, head_dim),
                        device=self.device,
                        dtype=torch.float32,
                    )

                q_triton = q.squeeze(0).contiguous() if self.use_triton else None

                for kv_start, kv_end in _iter_blocks(
                    total_tokens, self.kv_block_tokens
                ):
                    if self.precompute_kv:
                        assert k_store is not None and v_store is not None
                        k_block = k_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        v_block = v_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        kv_len = kv_end - kv_start
                        k = (
                            k_block
                            .view(1, kv_len, num_heads, head_dim)
                            .permute(0, 2, 1, 3)
                            .contiguous()
                        )
                        v = (
                            v_block
                            .view(1, kv_len, num_heads, head_dim)
                            .permute(0, 2, 1, 3)
                            .contiguous()
                        )
                    else:
                        x_kv = x_in.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        with torch.amp.autocast(
                            enabled=self.use_amp,
                            dtype=self.compute_dtype,
                            device_type=self.device_type,
                        ):
                            x_kv_norm = block.norm1(x_kv)
                            qkv_kv = attn.qkv(x_kv_norm)
                        qkv_kv = qkv_kv.view(1, -1, 3, num_heads, head_dim).permute(
                            2, 0, 3, 1, 4
                        )
                        k = qkv_kv[1]
                        v = qkv_kv[2]

                    if self.use_triton:
                        assert q_triton is not None
                        k_triton = k.squeeze(0).contiguous()
                        v_triton = v.squeeze(0).contiguous()
                        online_attn_update(
                            q_triton,
                            k_triton,
                            v_triton,
                            m,
                            l,
                            out,
                            scale,
                            block_m=self.triton_block_m,
                            block_n=self.triton_block_n,
                            block_d=self.triton_block_d,
                            num_warps=self.triton_num_warps,
                            num_stages=self.triton_num_stages,
                        )
                    else:
                        scores = (
                            torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                        )
                        block_max = scores.max(dim=-1).values
                        m_new = torch.maximum(m, block_max)
                        exp_m = torch.exp(m - m_new)
                        exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
                        l = exp_m * l + exp_scores.sum(dim=-1)
                        out = exp_m.unsqueeze(-1) * out + torch.matmul(
                            exp_scores, v.float()
                        )
                        m = m_new

                if self.use_triton:
                    out = out / l.unsqueeze(-1)
                    out = out.to(dtype=self.compute_dtype)
                    out = out.transpose(0, 1).reshape(1, q_len, num_heads * head_dim)
                else:
                    out = out / l.unsqueeze(-1)
                    out = out.to(dtype=self.compute_dtype)
                    out = out.transpose(1, 2).reshape(1, q_len, num_heads * head_dim)

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

    def _run_last_block(
        self,
        block: torch.nn.Module,
        x_in: TokenStore,
        x_out: TokenStore,
        total_tokens: int,
        embed_dim: int,
        num_heads: int,
        norm_feat: str,
    ) -> TokenStore:
        attn = block.attn
        head_dim = attn.qkv.in_features // num_heads
        scale = attn.scale
        embed_dim = num_heads * head_dim

        cls_context: torch.Tensor | None = None
        k_store: TokenStore | None = None
        v_store: TokenStore | None = None
        if self.precompute_kv:
            k_store, v_store = self._precompute_kv(
                block=block,
                x_in=x_in,
                total_tokens=total_tokens,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )

        with torch.inference_mode():
            for q_start, q_end in _iter_blocks(total_tokens, self.q_block_tokens):
                x_q = x_in.read(q_start, q_end, self.device, self.compute_dtype)
                with torch.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.compute_dtype,
                    device_type=self.device_type,
                ):
                    x_q_norm = block.norm1(x_q)
                    qkv_q = attn.qkv(x_q_norm)
                qkv_q = qkv_q.view(1, -1, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q = qkv_q[0]

                q_len = q.shape[2]
                if self.use_triton:
                    m = torch.full(
                        (num_heads, q_len),
                        float("-inf"),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    l = torch.zeros(
                        (num_heads, q_len),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    out = torch.zeros(
                        (num_heads, q_len, head_dim),
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    m = torch.full(
                        (1, num_heads, q_len),
                        float("-inf"),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    l = torch.zeros(
                        (1, num_heads, q_len),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    out = torch.zeros(
                        (1, num_heads, q_len, head_dim),
                        device=self.device,
                        dtype=torch.float32,
                    )

                q_triton = q.squeeze(0).contiguous() if self.use_triton else None

                for kv_start, kv_end in _iter_blocks(
                    total_tokens, self.kv_block_tokens
                ):
                    if self.precompute_kv:
                        assert k_store is not None and v_store is not None
                        k_block = k_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        v_block = v_store.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        kv_len = kv_end - kv_start
                        k = (
                            k_block
                            .view(1, kv_len, num_heads, head_dim)
                            .permute(0, 2, 1, 3)
                            .contiguous()
                        )
                        v = (
                            v_block
                            .view(1, kv_len, num_heads, head_dim)
                            .permute(0, 2, 1, 3)
                            .contiguous()
                        )
                    else:
                        x_kv = x_in.read(
                            kv_start, kv_end, self.device, self.compute_dtype
                        )
                        with torch.amp.autocast(
                            enabled=self.use_amp,
                            dtype=self.compute_dtype,
                            device_type=self.device_type,
                        ):
                            x_kv_norm = block.norm1(x_kv)
                            qkv_kv = attn.qkv(x_kv_norm)
                        qkv_kv = qkv_kv.view(1, -1, 3, num_heads, head_dim).permute(
                            2, 0, 3, 1, 4
                        )
                        k = qkv_kv[1]
                        v = qkv_kv[2]

                    if self.use_triton:
                        assert q_triton is not None
                        k_triton = k.squeeze(0).contiguous()
                        v_triton = v.squeeze(0).contiguous()
                        online_attn_update(
                            q_triton,
                            k_triton,
                            v_triton,
                            m,
                            l,
                            out,
                            scale,
                            block_m=self.triton_block_m,
                            block_n=self.triton_block_n,
                            block_d=self.triton_block_d,
                            num_warps=self.triton_num_warps,
                            num_stages=self.triton_num_stages,
                        )
                    else:
                        scores = (
                            torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                        )
                        block_max = scores.max(dim=-1).values
                        m_new = torch.maximum(m, block_max)
                        exp_m = torch.exp(m - m_new)
                        exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
                        l = exp_m * l + exp_scores.sum(dim=-1)
                        out = exp_m.unsqueeze(-1) * out + torch.matmul(
                            exp_scores, v.float()
                        )
                        m = m_new

                if self.use_triton:
                    out = out / l.unsqueeze(-1)
                    if q_start <= 0 < q_end:
                        cls_index = 0 - q_start
                        cls_context = (
                            out[:, cls_index : cls_index + 1, :].clone().unsqueeze(0)
                        )

                    out = out.to(dtype=self.compute_dtype)
                    out = out.transpose(0, 1).reshape(1, q_len, num_heads * head_dim)
                else:
                    out = out / l.unsqueeze(-1)
                    if q_start <= 0 < q_end:
                        cls_index = 0 - q_start
                        cls_context = out[:, :, cls_index : cls_index + 1, :].clone()

                    out = out.to(dtype=self.compute_dtype)
                    out = out.transpose(1, 2).reshape(1, q_len, num_heads * head_dim)

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
            for kv_start, kv_end in _iter_blocks(total_tokens, self.kv_block_tokens):
                if self.precompute_kv:
                    assert v_store is not None
                    v_block = v_store.read(
                        kv_start, kv_end, self.device, self.compute_dtype
                    )
                    kv_len = kv_end - kv_start
                    v = (
                        v_block
                        .view(1, kv_len, num_heads, head_dim)
                        .permute(0, 2, 1, 3)
                        .contiguous()
                    )
                else:
                    x_kv = x_in.read(kv_start, kv_end, self.device, self.compute_dtype)
                    with torch.amp.autocast(
                        enabled=self.use_amp,
                        dtype=self.compute_dtype,
                        device_type=self.device_type,
                    ):
                        x_kv_norm = block.norm1(x_kv)
                        qkv_kv = attn.qkv(x_kv_norm)
                    qkv_kv = qkv_kv.view(1, -1, 3, num_heads, head_dim).permute(
                        2, 0, 3, 1, 4
                    )
                    v = qkv_kv[2]
                cls_attn = torch.matmul(
                    cls_context.float(), v.float().transpose(-2, -1)
                )
                cls_attn = cls_attn.squeeze(2).transpose(1, 2).contiguous()
                attn_store.write(kv_start, kv_end, cls_attn.squeeze(0))

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

    def _precompute_kv(
        self,
        block: torch.nn.Module,
        x_in: TokenStore,
        total_tokens: int,
        embed_dim: int,
        num_heads: int,
    ) -> tuple[TokenStore, TokenStore]:
        head_dim = embed_dim // num_heads
        attn = block.attn

        block_tag = f"{id(block)}"
        k_store = self._create_store(
            name=f"kv_k_{block_tag}",
            shape=(total_tokens, embed_dim),
            num_special=x_in.num_special,
            grid_size=x_in.grid_size,
            dtype=self.kv_storage_dtype,
            storage_kind=self.kv_storage_kind,
        )
        v_store = self._create_store(
            name=f"kv_v_{block_tag}",
            shape=(total_tokens, embed_dim),
            num_special=x_in.num_special,
            grid_size=x_in.grid_size,
            dtype=self.kv_storage_dtype,
            storage_kind=self.kv_storage_kind,
        )

        with torch.inference_mode():
            for kv_start, kv_end in _iter_blocks(total_tokens, self.kv_block_tokens):
                x_kv = x_in.read(kv_start, kv_end, self.device, self.compute_dtype)
                with torch.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.compute_dtype,
                    device_type=self.device_type,
                ):
                    x_kv_norm = block.norm1(x_kv)
                    qkv_kv = attn.qkv(x_kv_norm)
                qkv_kv = qkv_kv.view(1, -1, 3, num_heads, head_dim).permute(
                    2, 0, 3, 1, 4
                )
                k = qkv_kv[1].transpose(1, 2).reshape(1, -1, embed_dim).squeeze(0)
                v = qkv_kv[2].transpose(1, 2).reshape(1, -1, embed_dim).squeeze(0)
                k_store.write(kv_start, kv_end, k)
                v_store.write(kv_start, kv_end, v)

        k_store.flush()
        v_store.flush()
        return k_store, v_store

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
