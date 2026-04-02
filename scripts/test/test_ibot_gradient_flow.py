"""Test that the iBOT student path propagates gradients back to the encoder.

The student iBOT path uses a buffer + copy_() pattern to gather masked patch
tokens before passing them through the ibot_head:

    buffer_tensor = output.new_zeros(upperbound, embed_dim)
    buffer_tensor[:n_masked].copy_(index_select(...))
    result = ibot_head(buffer_tensor)[:n_masked]

In PyTorch 2.5+, copy_() creates a CopySlices autograd node, so gradients
flow through. This test verifies that behavior and catches regressions if
the autograd semantics of copy_() ever change.

It also compares against the simpler direct index_select approach to confirm
both produce identical encoder gradients.

Run:
    python scripts/test/test_ibot_gradient_flow.py
"""

import torch
import torch.nn as nn


class FakeEncoder(nn.Module):
    """Minimal stand-in for the student encoder."""

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FakeHead(nn.Module):
    """Minimal stand-in for the ibot_head (DINOHead)."""

    def __init__(self, embed_dim: int = 16, out_dim: int = 8):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def ibot_path_buffer_copy(
    patch_tokens: torch.Tensor,
    head: nn.Module,
    upperbound: int,
    n_masked_patches: int,
    mask_indices: torch.Tensor,
    embed_dim: int,
) -> torch.Tensor:
    """Current iBOT student path: new_zeros + copy_ + head(full_buffer)[:n].

    Relies on copy_() being autograd-aware (CopySlices node in PyTorch 2.5+).
    """
    buffer_tensor = patch_tokens.new_zeros(upperbound, embed_dim)
    buffer_tensor[:n_masked_patches].copy_(
        torch.index_select(
            patch_tokens.flatten(0, 1),
            dim=0,
            index=mask_indices,
        )
    )
    return head(buffer_tensor)[:n_masked_patches]


def ibot_path_direct(
    patch_tokens: torch.Tensor,
    head: nn.Module,
    mask_indices: torch.Tensor,
) -> torch.Tensor:
    """Direct index_select path (no buffer). Always preserves gradients."""
    selected = torch.index_select(
        patch_tokens.flatten(0, 1),
        dim=0,
        index=mask_indices,
    )
    return head(selected)


def run_path(encoder, head, x, mask_indices, path_fn, **kwargs):
    """Run a forward+backward pass and return encoder/head gradients."""
    encoder.zero_grad()
    head.zero_grad()
    patch_tokens = encoder(x)
    output = path_fn(patch_tokens, head, mask_indices=mask_indices, **kwargs)
    loss = output.sum()
    loss.backward()
    return (
        encoder.linear.weight.grad.clone(),
        head.proj.weight.grad.clone(),
    )


def test_gradient_flow():
    torch.manual_seed(0)
    embed_dim = 16
    batch_size = 2
    n_patches = 8
    n_masked_patches = 5
    upperbound = 10

    encoder = FakeEncoder(embed_dim)
    head = FakeHead(embed_dim)

    x = torch.randn(batch_size, n_patches, embed_dim)
    mask_indices = torch.tensor([0, 3, 5, 9, 12])

    # --- Buffer + copy_ path (current code) ---
    enc_grad_buf, head_grad_buf = run_path(
        encoder, head, x, mask_indices,
        ibot_path_buffer_copy,
        upperbound=upperbound,
        n_masked_patches=n_masked_patches,
        embed_dim=embed_dim,
    )

    # --- Direct index_select path ---
    enc_grad_direct, _ = run_path(
        encoder, head, x, mask_indices,
        ibot_path_direct,
    )

    # --- Checks ---
    print("=" * 60)
    print("iBOT Student Path — Gradient Flow Regression Test")
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 60)

    all_passed = True

    # 1. Encoder must receive gradients from buffer+copy_ path
    enc_has_grad = enc_grad_buf.abs().sum().item() > 0
    status = "PASS" if enc_has_grad else "FAIL"
    if not enc_has_grad:
        all_passed = False
    print(f"[{status}] Encoder receives grad via buffer+copy_ path")
    print(f"       Encoder grad norm: {enc_grad_buf.norm().item():.6f}")

    # 2. Head must receive gradients from buffer+copy_ path
    head_has_grad = head_grad_buf.abs().sum().item() > 0
    status = "PASS" if head_has_grad else "FAIL"
    if not head_has_grad:
        all_passed = False
    print(f"[{status}] Head receives grad via buffer+copy_ path")
    print(f"       Head grad norm: {head_grad_buf.norm().item():.6f}")

    # 3. Encoder must receive gradients from direct path
    enc_has_grad_d = enc_grad_direct.abs().sum().item() > 0
    status = "PASS" if enc_has_grad_d else "FAIL"
    if not enc_has_grad_d:
        all_passed = False
    print(f"[{status}] Encoder receives grad via direct path")
    print(f"       Encoder grad norm: {enc_grad_direct.norm().item():.6f}")

    # 4. Both paths should produce identical encoder gradients
    grads_match = torch.allclose(enc_grad_buf, enc_grad_direct, atol=1e-6)
    status = "PASS" if grads_match else "FAIL"
    if not grads_match:
        all_passed = False
        diff = (enc_grad_buf - enc_grad_direct).abs().max().item()
        print(f"[{status}] Encoder grads match between paths (max diff: {diff:.2e})")
    else:
        print(f"[{status}] Encoder grads match between paths")

    # 5. Verify CopySlices autograd node exists (PyTorch 2.5+ behavior)
    patch_tokens = encoder(x)
    buffer_tensor = patch_tokens.new_zeros(upperbound, embed_dim)
    buffer_tensor[:n_masked_patches].copy_(
        torch.index_select(patch_tokens.flatten(0, 1), 0, mask_indices)
    )
    has_copy_slices = buffer_tensor.grad_fn is not None
    status = "PASS" if has_copy_slices else "FAIL"
    if not has_copy_slices:
        all_passed = False
    grad_fn_name = type(buffer_tensor.grad_fn).__name__ if buffer_tensor.grad_fn else "None"
    print(f"[{status}] Buffer has autograd node after copy_ (grad_fn={grad_fn_name})")

    print()
    print("=" * 60)
    if all_passed:
        print("ALL PASSED — iBOT gradient flow is correct")
    else:
        print("FAILED — iBOT gradient flow is broken, encoder won't learn from iBOT loss")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = test_gradient_flow()
    raise SystemExit(0 if success else 1)
