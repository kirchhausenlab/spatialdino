from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TRITON_AVAILABLE = False


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")


if TRITON_AVAILABLE:

    @triton.jit
    def _online_attn_update_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        m_ptr,
        l_ptr,
        out_ptr,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_oh,
        stride_om,
        stride_od,
        q_len,
        k_len,
        d_head,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        """Online softmax update kernel.

        Same structure as FlashAttention-2 but loads existing ``m``,
        ``l``, ``acc`` accumulators from memory and stores updated values back
        **without** final normalisation.  The caller normalises once after all
        K/V blocks have been processed.

        Grid: ``(num_heads, ceil(q_len / BLOCK_M))``.
        """
        pid_h = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < q_len
        mask_d = offs_d < d_head

        # Load Q tile and pre-scale.
        q_ptrs = (
            q_ptr
            + pid_h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        q = (q.to(tl.float32) * scale).to(q.dtype)

        # Load running accumulators.
        m_ptrs = m_ptr + pid_h * q_len + offs_m
        l_ptrs = l_ptr + pid_h * q_len + offs_m
        out_ptrs = (
            out_ptr
            + pid_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od
        )
        m_i = tl.load(m_ptrs, mask=mask_m, other=float("-inf"))
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        acc = tl.load(out_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

        for n_start in range(0, k_len, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < k_len

            k_ptrs = (
                k_ptr
                + pid_h * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + pid_h * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

            scores = tl.dot(q, tl.trans(k)).to(tl.float32)
            scores = tl.where(mask_n[None, :], scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
            m_i = m_new

        tl.store(m_ptrs, m_i, mask=mask_m)
        tl.store(l_ptrs, l_i, mask=mask_m)
        tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


    @triton.jit
    def _finalize_attn_output_kernel(
        acc_ptr,
        l_ptr,
        out_ptr,
        stride_ah,
        stride_am,
        stride_ad,
        stride_oh,
        stride_om,
        stride_od,
        q_len,
        d_head,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        pid_h = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < q_len
        mask_d = offs_d < d_head

        acc_ptrs = (
            acc_ptr
            + pid_h * stride_ah
            + offs_m[:, None] * stride_am
            + offs_d[None, :] * stride_ad
        )
        out_ptrs = (
            out_ptr
            + pid_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od
        )
        l_ptrs = l_ptr + pid_h * q_len + offs_m

        acc = tl.load(acc_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        l_i = tl.load(l_ptrs, mask=mask_m, other=1.0)
        out = acc / l_i[:, None]
        tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


def online_attn_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    row_sum: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    *,
    block_m: int = 64,
    block_n: int = 128,
    block_d: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Update online-softmax accumulators with a new K/V block.

    Args:
        q: ``[H, M, D]`` query tensor (CUDA).
        k: ``[H, N, D]`` key block (CUDA).
        v: ``[H, N, D]`` value block (CUDA, same N as ``k``).
        m: ``[H, M]`` running row-max (modified **in-place**).
        row_sum: ``[H, M]`` running row-sum (modified **in-place**).
        out: ``[H, M, D]`` running unnormalised output (modified **in-place**).
        scale: softmax scale, typically ``1 / sqrt(D)``.

    The caller must normalise with ``out /= row_sum.unsqueeze(-1)`` once all K/V
    blocks have been processed.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available.")
    H, M, D = q.shape
    N = k.shape[1]
    _require_cuda_tensor("q", q)
    _require_cuda_tensor("k", k)
    _require_cuda_tensor("v", v)
    _require_cuda_tensor("m", m)
    _require_cuda_tensor("row_sum", row_sum)
    _require_cuda_tensor("out", out)
    if k.shape[0] != H or v.shape[0] != H:
        raise ValueError("q, k, and v must have the same number of heads.")
    if k.shape[2] != D or v.shape[2] != D:
        raise ValueError("q, k, and v must have the same head_dim.")
    if v.shape[1] != N:
        raise ValueError("k and v must have the same token length.")
    if m.shape != (H, M) or row_sum.shape != (H, M):
        raise ValueError("m and row_sum must have shape [num_heads, q_len].")
    if out.shape != (H, M, D):
        raise ValueError("out must have shape [num_heads, q_len, head_dim].")
    if block_d < 16:
        raise ValueError("BLOCK_D must be at least 16 for tl.dot.")
    if D > block_d:
        raise ValueError(f"head_dim={D} exceeds BLOCK_D={block_d}. Increase block_d.")

    grid = (H, (M + block_m - 1) // block_m)
    _online_attn_update_kernel[grid](
        q,
        k,
        v,
        m,
        row_sum,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        M,
        N,
        D,
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def finalize_attn_output(
    acc: torch.Tensor,
    row_sum: torch.Tensor,
    *,
    dtype: torch.dtype,
    block_m: int = 64,
    block_d: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Normalize online-softmax accumulators into a head-major output tensor.

    Args:
        acc: ``[H, M, D]`` unnormalised attention output, float32.
        row_sum: ``[H, M]`` row sums, float32.
        dtype: output dtype.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available.")
    H, M, D = acc.shape
    _require_cuda_tensor("acc", acc)
    _require_cuda_tensor("row_sum", row_sum)
    if row_sum.shape != (H, M):
        raise ValueError("row_sum must have shape [num_heads, q_len].")
    if block_d < 16:
        raise ValueError("BLOCK_D must be at least 16 for tl.dot-compatible tiling.")
    if D > block_d:
        raise ValueError(f"head_dim={D} exceeds BLOCK_D={block_d}. Increase block_d.")

    out = torch.empty_like(acc, dtype=dtype)
    grid = (H, (M + block_m - 1) // block_m)
    _finalize_attn_output_kernel[grid](
        acc,
        row_sum,
        out,
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        M,
        D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out
