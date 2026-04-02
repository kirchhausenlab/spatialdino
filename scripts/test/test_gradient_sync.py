"""Multi-node multi-GPU gradient synchronisation test.

Verifies that DDP all-reduce produces identical gradients across all ranks
after a forward+backward pass on a small synthetic model.

Launch:
    torchrun --nnodes=N --nproc_per_node=G --rdzv-id=TEST \
             --rdzv-backend=c10d --rdzv-endpoint=HOST:PORT \
             scripts/test/test_gradient_sync.py

Exit code 0  => all checks passed on every rank.
Exit code 1  => gradient mismatch detected.
"""

import argparse
import datetime
import logging
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][rank %(rank)s] %(message)s",
    datefmt="%H:%M:%S",
)


class _RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


logger = logging.getLogger("grad_sync_test")


# ---------------------------------------------------------------------------
# Tiny model used for the test
# ---------------------------------------------------------------------------
class TinyModel(nn.Module):
    """Small Conv3d + Linear head — mirrors the structure of a 3-D ViT."""

    def __init__(self, in_channels: int = 1, hidden: int = 64, out: int = 16):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        h = self.conv(x)  # [B, hidden, D, H, W]
        h = h.mean(dim=(2, 3, 4))  # global average pool → [B, hidden]
        h = self.norm(h)
        return self.head(h)  # [B, out]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _broadcast_model_params(model: nn.Module, src: int = 0):
    """Ensure all ranks start with identical parameters."""
    for p in model.parameters():
        dist.broadcast(p.data, src=src)


def _gather_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect gradients from all named parameters."""
    return {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def _allgather_tensor(tensor: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    """all_gather a tensor from every rank."""
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def run_test(args: argparse.Namespace) -> bool:
    """Returns True if all gradient checks pass."""

    rank = dist.get_rank()
    local_rank = int(args.local_rank)
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    logger.info(
        f"world_size={world_size}  node={rank // args.nproc_per_node}  "
        f"local_rank={local_rank}  device={device}"
    )

    # --- 1. Build model, sync initial params ----------------------------------
    torch.manual_seed(42)  # same init on every rank
    model = TinyModel(hidden=args.hidden, out=args.out).to(device)
    _broadcast_model_params(model, src=0)

    ddp_model = DDP(model, device_ids=[local_rank])

    # --- 2. Create identical input on every rank (deterministic seed) ----------
    if args.same_data:
        # Same data everywhere — gradients must be bit-identical after all-reduce
        gen = torch.Generator(device=device).manual_seed(123)
    else:
        # Different data per rank — gradients converge but won't be identical
        # before all-reduce; they MUST be identical after DDP all-reduce.
        gen = torch.Generator(device=device).manual_seed(123 + rank)

    x = torch.randn(
        args.batch_size, 1, 8, 8, 8, device=device, generator=gen
    )
    target = torch.randn(
        args.batch_size, args.out, device=device, generator=gen
    )

    # --- 3. Forward / backward ------------------------------------------------
    ddp_model.zero_grad()
    out = ddp_model(x)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()  # DDP hooks trigger all-reduce here

    logger.info(f"loss = {loss.item():.6f}")

    # --- 4. Gather gradients from all ranks and compare -----------------------
    local_grads = _gather_grads(ddp_model.module)

    all_ok = True
    for name, grad in local_grads.items():
        gathered = _allgather_tensor(grad, world_size)

        for other_rank, other_grad in enumerate(gathered):
            if other_rank == rank:
                continue

            if args.same_data:
                # Bit-exact match expected
                match = torch.equal(grad, other_grad)
            else:
                # Allow small floating-point tolerance
                match = torch.allclose(grad, other_grad, atol=args.atol, rtol=args.rtol)

            if not match:
                max_diff = (grad - other_grad).abs().max().item()
                logger.error(
                    f"MISMATCH  param={name}  rank {rank} vs {other_rank}  "
                    f"max_diff={max_diff:.2e}"
                )
                all_ok = False
            else:
                if rank == 0:
                    max_diff = (grad - other_grad).abs().max().item()
                    logger.info(
                        f"OK  param={name}  rank 0 vs {other_rank}  "
                        f"max_diff={max_diff:.2e}"
                    )

    # --- 5. Global agreement on pass/fail -------------------------------------
    passed = torch.tensor([1 if all_ok else 0], device=device)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    global_pass = passed.item() == 1

    if rank == 0:
        if global_pass:
            logger.info("ALL RANKS PASSED — gradients are synchronised.")
        else:
            logger.error("SOME RANKS FAILED — gradient mismatch detected!")

    return global_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-node gradient sync test")
    p.add_argument("--backend", default="nccl", help="torch.distributed backend")
    p.add_argument("--timeout", type=int, default=120, help="init timeout (seconds)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--out", type=int, default=16)
    p.add_argument(
        "--same_data",
        action="store_true",
        help="Use identical input on every rank (expect bit-exact grads)",
    )
    p.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    p.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    return p.parse_args()


def main():
    args = parse_args()

    # torchrun sets LOCAL_RANK in env
    args.local_rank = int(
        __import__("os").environ.get("LOCAL_RANK", 0)
    )
    args.nproc_per_node = int(
        __import__("os").environ.get("LOCAL_WORLD_SIZE", 1)
    )

    dist.init_process_group(
        backend=args.backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=args.timeout),
    )

    logger.addFilter(_RankFilter(dist.get_rank()))

    try:
        passed = run_test(args)
    finally:
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
