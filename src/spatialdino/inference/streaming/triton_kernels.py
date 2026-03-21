from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TRITON_AVAILABLE = False


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
        stride_mh,
        stride_mm,
        stride_lh,
        stride_lm,
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
        pid_h = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        mask_m = offs_m < q_len
        mask_n = offs_n < k_len
        mask_d = offs_d < d_head

        q_ptrs = (
            q_ptr
            + pid_h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
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

        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        q = q.to(tl.float32)
        k = k.to(tl.float32)
        v = v.to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(mask_n[None, :], scores, float("-inf"))
        block_max = tl.max(scores, axis=1)

        m_ptrs = m_ptr + pid_h * stride_mh + offs_m * stride_mm
        l_ptrs = l_ptr + pid_h * stride_lh + offs_m * stride_lm
        out_ptrs = (
            out_ptr
            + pid_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od
        )

        m_prev = tl.load(m_ptrs, mask=mask_m, other=float("-inf"))
        l_prev = tl.load(l_ptrs, mask=mask_m, other=0.0)
        out_prev = tl.load(
            out_ptrs,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        )

        m_new = tl.maximum(m_prev, block_max)
        exp_m = tl.exp(m_prev - m_new)
        exp_scores = tl.exp(scores - m_new[:, None])
        l_new = exp_m * l_prev + tl.sum(exp_scores, axis=1)
        out_new = exp_m[:, None] * out_prev + tl.dot(exp_scores, v)

        tl.store(m_ptrs, m_new, mask=mask_m)
        tl.store(l_ptrs, l_new, mask=mask_m)
        tl.store(out_ptrs, out_new, mask=mask_m[:, None] & mask_d[None, :])


def online_attn_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    block_m: int,
    block_n: int,
    block_d: int,
    num_warps: int,
    num_stages: int,
) -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available.")
    if q.device.type != "cuda":
        raise ValueError("Triton attention requires CUDA tensors.")

    q_len = q.shape[1]
    k_len = k.shape[1]
    d_head = q.shape[2]

    grid = (q.shape[0], (q_len + block_m - 1) // block_m)
    _online_attn_update_kernel[grid](
        q,
        k,
        v,
        m,
        l,
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
        m.stride(0),
        m.stride(1),
        l.stride(0),
        l.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q_len,
        k_len,
        d_head,
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
