"""Test whether DDP gradient synchronization happens when forward is called
through the unwrapped model vs the DDP wrapper.

Launch with:
    torchrun --nproc_per_node=2 scripts/test/test_ddp_forward.py
"""

import os

import torch
import torch.distributed as tdist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.linear(x).sum()

    def custom_forward(self, x):
        """A custom method that does the same thing as forward()."""
        return self.linear(x).sum()


class TinyModelDelegating(nn.Module):
    """Model where forward() delegates to a custom method.

    This tests whether DDP syncs gradients when the actual compute
    happens in a custom method, but is called *through* forward().
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)

    def forward(self, x, mode="default"):
        if mode == "custom":
            return self.custom_forward(x)
        return self.linear(x).sum()

    def custom_forward(self, x):
        return self.linear(x).sum()


def run_test():
    tdist.init_process_group(backend="nccl")
    rank = tdist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()

    # ---------- Test 1: forward through DDP wrapper ----------
    torch.manual_seed(42)
    model1 = TinyModel().to(device)
    ddp1 = DDP(model1, device_ids=[rank])

    # Each rank gets different input so gradients differ before sync
    x1 = torch.randn(3, 4, device=device) * (rank + 1)
    loss1 = ddp1(x1)  # forward through DDP wrapper
    loss1.backward()

    grad_ddp = model1.linear.weight.grad.clone()

    # ---------- Test 2: forward through unwrapped model ----------
    torch.manual_seed(42)
    model2 = TinyModel().to(device)
    ddp2 = DDP(model2, device_ids=[rank])

    x2 = torch.randn(3, 4, device=device) * (rank + 1)
    loss2 = model2(x2)  # forward through UNWRAPPED model (bypassing DDP)
    loss2.backward()

    grad_unwrapped = model2.linear.weight.grad.clone()

    # ---------- Test 3: custom method called through DDP wrapper ----------
    torch.manual_seed(42)
    model3 = TinyModel().to(device)
    ddp3 = DDP(model3, device_ids=[rank])

    x3 = torch.randn(3, 4, device=device) * (rank + 1)
    loss3 = ddp3.module.custom_forward(x3)  # custom method via ddp.module
    loss3.backward()

    grad_custom_via_module = model3.linear.weight.grad.clone()

    # ---------- Test 4: custom method called directly on DDP wrapper ----------
    torch.manual_seed(42)
    model4 = TinyModel().to(device)
    ddp4 = DDP(model4, device_ids=[rank])

    try:
        x4 = torch.randn(3, 4, device=device) * (rank + 1)
        loss4 = ddp4.custom_forward(x4)  # custom method via DDP (attribute forwarding)
        loss4.backward()
        grad_custom_via_ddp = model4.linear.weight.grad.clone()
    except AttributeError:
        print(
            f"Rank {rank}: DDP wrapper did not forward custom method, skipping Test 4"
        )

        grad_custom_via_ddp = torch.zeros_like(model4.linear.weight)

    # ---------- Test 5: forward() delegates to custom method, called through DDP ----------
    torch.manual_seed(42)
    model5 = TinyModelDelegating().to(device)
    ddp5 = DDP(model5, device_ids=[rank])

    x5 = torch.randn(3, 4, device=device) * (rank + 1)
    loss5 = ddp5(x5, mode="custom")  # goes through DDP forward -> forward() -> custom_forward()
    loss5.backward()

    grad_delegating = model5.linear.weight.grad.clone()

    # ---------- Test 6: no DDP at all (baseline — no sync expected) ----------
    torch.manual_seed(42)
    model6 = TinyModel().to(device)
    # No DDP wrapper

    x6 = torch.randn(3, 4, device=device) * (rank + 1)
    loss6 = model6(x6)
    loss6.backward()

    grad_no_ddp = model6.linear.weight.grad.clone()

    # ---------- Gather grads from both ranks to compare ----------
    def gather_grad(grad):
        gathered = [torch.zeros_like(grad) for _ in range(2)]
        tdist.all_gather(gathered, grad)
        return gathered

    grads_ddp = gather_grad(grad_ddp)
    grads_unwrapped = gather_grad(grad_unwrapped)
    grads_custom_via_module = gather_grad(grad_custom_via_module)
    grads_custom_via_ddp = gather_grad(grad_custom_via_ddp)
    grads_delegating = gather_grad(grad_delegating)
    grads_no_ddp = gather_grad(grad_no_ddp)

    if rank == 0:
        ddp_synced = torch.allclose(grads_ddp[0], grads_ddp[1])
        unwrapped_synced = torch.allclose(grads_unwrapped[0], grads_unwrapped[1])
        custom_module_synced = torch.allclose(
            grads_custom_via_module[0], grads_custom_via_module[1]
        )
        custom_ddp_synced = torch.allclose(
            grads_custom_via_ddp[0], grads_custom_via_ddp[1]
        )
        delegating_synced = torch.allclose(grads_delegating[0], grads_delegating[1])
        no_ddp_synced = torch.allclose(grads_no_ddp[0], grads_no_ddp[1])

        print("=" * 60)
        print(
            f"1. Forward through DDP wrapper (ddp(x)):        grads synced = {ddp_synced}"
        )
        print(f"   rank0 grad: {grads_ddp[0]}")
        print(f"   rank1 grad: {grads_ddp[1]}")
        print()
        print(
            f"2. Forward through unwrapped model (model(x)):  grads synced = {unwrapped_synced}"
        )
        print(f"   rank0 grad: {grads_unwrapped[0]}")
        print(f"   rank1 grad: {grads_unwrapped[1]}")
        print()
        print(
            f"3. Custom method via ddp.module (ddp.module.custom_forward(x)): grads synced = {custom_module_synced}"
        )
        print(f"   rank0 grad: {grads_custom_via_module[0]}")
        print(f"   rank1 grad: {grads_custom_via_module[1]}")
        print()
        print(
            f"4. Custom method via DDP attr forwarding (ddp.custom_forward(x)): grads synced = {custom_ddp_synced}"
        )
        print(f"   rank0 grad: {grads_custom_via_ddp[0]}")
        print(f"   rank1 grad: {grads_custom_via_ddp[1]}")
        print()
        print(
            f"5. forward() delegates to custom_forward(), called through DDP (ddp(x, mode='custom')): grads synced = {delegating_synced}"
        )
        print(f"   rank0 grad: {grads_delegating[0]}")
        print(f"   rank1 grad: {grads_delegating[1]}")
        print()
        print(
            f"6. No DDP at all (baseline):                    grads synced = {no_ddp_synced}"
        )
        print(f"   rank0 grad: {grads_no_ddp[0]}")
        print(f"   rank1 grad: {grads_no_ddp[1]}")
        print("=" * 60)

    tdist.destroy_process_group()


if __name__ == "__main__":
    run_test()
