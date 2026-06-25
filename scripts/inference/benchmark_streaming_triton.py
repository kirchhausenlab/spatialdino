from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from spatialdino.inference.streaming.storage import TokenStore
from spatialdino.inference.streaming.streaming_encoder import StreamingEncoder
from spatialdino.inference.streaming.triton_kernels import TRITON_AVAILABLE


class _BenchAttn(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = float((embed_dim // num_heads) ** -0.5)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_drop = nn.Identity()


class _BenchBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = _BenchAttn(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ls1 = nn.Identity()
        self.ls2 = nn.Identity()


class _BenchEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block: _BenchBlock) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = (8, 8, 8)
        self.stride = (8, 8, 8)
        self.num_register_tokens = 0
        self.num_tt_register_tokens = 0
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = None
        self.tt_register_tokens = None
        self.blocks = nn.ModuleList([block])
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def use_pos_embed(self) -> bool:
        return False

    def _build_rope_cache(self, *args: Any, **kwargs: Any) -> None:
        return None


def _make_config(args: argparse.Namespace, *, use_triton: bool, optimized: bool) -> OmegaConf:
    cfg = {
        "use_amp": args.dtype in ("fp16", "bf16"),
        "device_type": "cuda",
        "dtype": args.dtype,
        "streaming_storage": args.token_storage,
        "streaming_kv_storage": args.kv_storage,
        "streaming_q_block_tokens": args.q_block_tokens,
        "streaming_kv_block_tokens": args.kv_block_tokens,
        "streaming_pin_memory": True,
        "streaming_use_triton": use_triton,
        "streaming_triton_optimized": optimized,
        "streaming_triton_async_kv_copy": not args.no_async_kv_copy,
        "streaming_triton_fused_finalize": not args.no_fused_finalize,
        "streaming_log_q_blocks": False,
        "streaming_triton_block_m": args.block_m,
        "streaming_triton_block_n": args.block_n,
        "streaming_triton_block_d": args.block_d,
        "streaming_triton_num_warps": args.num_warps,
        "streaming_triton_num_stages": args.num_stages,
    }
    return OmegaConf.create(cfg)


def _estimate_launches(
    total_tokens: int,
    q_block_tokens: int,
    kv_block_tokens: int,
    *,
    use_triton: bool,
    optimized: bool,
    kv_storage: str,
    fused_finalize: bool,
) -> dict[str, int]:
    q_blocks = math.ceil(total_tokens / q_block_tokens)
    if not use_triton:
        return {"q_blocks": q_blocks, "online": 0, "finalize": 0, "total": 0}
    kv_blocks = 1 if optimized and kv_storage == "gpu" else math.ceil(total_tokens / kv_block_tokens)
    online = q_blocks * kv_blocks
    finalize = q_blocks if optimized and fused_finalize else 0
    return {
        "q_blocks": q_blocks,
        "kv_blocks_per_q": kv_blocks,
        "online": online,
        "finalize": finalize,
        "total": online + finalize,
    }


def _make_stores(
    se: StreamingEncoder,
    tokens: torch.Tensor,
    total_tokens: int,
    embed_dim: int,
    token_storage: str,
    device: torch.device,
) -> tuple[TokenStore, TokenStore]:
    grid_size = (1, 1, total_tokens - 1)
    x_in = TokenStore.create(
        (total_tokens, embed_dim),
        tokens.dtype,
        token_storage,
        1,
        grid_size,
        device,
        pin_memory=True,
    )
    x_out = TokenStore.create(
        (total_tokens, embed_dim),
        tokens.dtype,
        token_storage,
        1,
        grid_size,
        device,
        pin_memory=True,
    )
    x_in.tensor.copy_(tokens.to(x_in.tensor.device))
    return x_in, x_out


def _time_cuda(fn: Any) -> tuple[Any, float, int]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    return result, elapsed, peak


def _run_variant(
    args: argparse.Namespace,
    block: _BenchBlock,
    tokens: torch.Tensor,
    *,
    label: str,
    use_triton: bool,
    optimized: bool,
    save_path: str | None,
) -> dict[str, Any]:
    device = torch.device("cuda")
    cfg = _make_config(args, use_triton=use_triton, optimized=optimized)
    if save_path is not None:
        cfg.save_path = save_path
    encoder = _BenchEncoder(args.embed_dim, args.num_heads, block)
    se = StreamingEncoder(encoder, device, cfg)

    def run_once() -> torch.Tensor:
        x_in, x_out = _make_stores(
            se,
            tokens,
            args.tokens,
            args.embed_dim,
            args.token_storage,
            device,
        )
        se._run_block(block, x_in, x_out, args.tokens, rope=None)
        return x_out.tensor.detach().cpu().float()

    for _ in range(args.warmup):
        run_once()
    output, elapsed, peak = _time_cuda(run_once)
    for _ in range(max(args.repeat - 1, 0)):
        output_i, elapsed_i, peak_i = _time_cuda(run_once)
        if elapsed_i < elapsed:
            output, elapsed, peak = output_i, elapsed_i, peak_i

    checksum = float(output.sum().item())
    return {
        "label": label,
        "seconds": elapsed,
        "peak_cuda_bytes": peak,
        "checksum": checksum,
        "launch_estimate": _estimate_launches(
            args.tokens,
            args.q_block_tokens,
            args.kv_block_tokens,
            use_triton=use_triton,
            optimized=optimized,
            kv_storage=args.kv_storage,
            fused_finalize=not args.no_fused_finalize,
        ),
        "output": output,
    }


def _add_diff(result: dict[str, Any], reference: torch.Tensor) -> None:
    diff = (result["output"] - reference).abs()
    result["max_abs_diff"] = float(diff.max().item())
    result["mean_abs_diff"] = float(diff.mean().item())
    del result["output"]


def _autotune_candidates(args: argparse.Namespace) -> list[tuple[int, int, int, int]]:
    candidates = [
        (64, 128, 4, 3),
        (64, 256, 4, 3),
        (128, 128, 4, 3),
        (128, 256, 8, 2),
        (128, 512, 8, 2),
        (256, 128, 8, 2),
    ]
    current = (args.block_m, args.block_n, args.num_warps, args.num_stages)
    if current not in candidates:
        candidates.insert(0, current)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark streaming Triton attention.")
    parser.add_argument("--tokens", type=int, default=32769)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--token-storage", choices=("gpu", "cpu", "disk"), default="gpu")
    parser.add_argument("--kv-storage", choices=("gpu", "cpu", "disk"), default="gpu")
    parser.add_argument("--q-block-tokens", type=int, default=16384)
    parser.add_argument("--kv-block-tokens", type=int, default=32768)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--block-d", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--no-async-kv-copy", action="store_true")
    parser.add_argument("--no-fused-finalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if not TRITON_AVAILABLE:
        raise SystemExit("Triton is required for this benchmark.")
    if args.embed_dim % args.num_heads != 0:
        raise SystemExit("embed_dim must be divisible by num_heads.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    block = _BenchBlock(args.embed_dim, args.num_heads).eval().to(device=device, dtype=dtype)
    tokens = torch.randn(args.tokens, args.embed_dim, device=device, dtype=dtype)

    with tempfile.TemporaryDirectory() as tmp:
        save_path = tmp if args.token_storage == "disk" or args.kv_storage == "disk" else None
        results = [
            _run_variant(
                args,
                block,
                tokens,
                label="reference_triton",
                use_triton=True,
                optimized=False,
                save_path=save_path,
            )
        ]
        if args.autotune:
            for block_m, block_n, num_warps, num_stages in _autotune_candidates(args):
                tune_args = argparse.Namespace(**vars(args))
                tune_args.block_m = block_m
                tune_args.block_n = block_n
                tune_args.num_warps = num_warps
                tune_args.num_stages = num_stages
                results.append(
                    _run_variant(
                        tune_args,
                        block,
                        tokens,
                        label=(
                            "optimized_triton"
                            f"_m{block_m}_n{block_n}_w{num_warps}_s{num_stages}"
                        ),
                        use_triton=True,
                        optimized=True,
                        save_path=save_path,
                    )
                )
        else:
            results.append(
                _run_variant(
                    args,
                    block,
                    tokens,
                    label="optimized_triton",
                    use_triton=True,
                    optimized=True,
                    save_path=save_path,
                )
            )
        if args.kv_storage == "gpu":
            results.append(
                _run_variant(
                    args,
                    block,
                    tokens,
                    label="sdpa",
                    use_triton=False,
                    optimized=False,
                    save_path=save_path,
                )
            )

    reference = results[0]["output"]
    for result in results:
        _add_diff(result, reference)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
