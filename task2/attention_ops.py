"""Task 2 Part A: Dense causal, tiled online-softmax, and PyTorch SDPA operators."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def dense_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """A1: Explicit QK^T/sqrt(d) causal attention with FP32 score accumulation."""
    orig_dtype = q.dtype
    _, _, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    # Numerically stable softmax in FP32
    max_scores = scores.amax(dim=-1, keepdim=True)
    # Guard all-masked rows (where max is -inf) to avoid -inf - (-inf) = NaN
    safe_max = torch.where(torch.isfinite(max_scores), max_scores, torch.zeros_like(max_scores))
    exp_scores = torch.exp(scores - safe_max)
    exp_scores = exp_scores.masked_fill(causal_mask, 0.0)
    denom = exp_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    probs = exp_scores / denom
    out = torch.matmul(probs, v.float())
    return out.to(dtype=orig_dtype)


def tiled_online_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tile_q: int = 128,
    tile_kv: int = 128,
) -> torch.Tensor:
    """A2: Exact blockwise online-softmax causal attention without materializing L x L scores.

    Note: Backward uses standard PyTorch autograd over the Python tile loop, which
    retains per-tile intermediates in memory rather than recomputing them in SRAM
    like a fused FlashAttention CUDA kernel.
    """
    orig_dtype = q.dtype
    bsz, n_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    out_chunks: list[torch.Tensor] = []
    for q_start in range(0, seq_len, tile_q):
        q_end = min(q_start + tile_q, seq_len)
        q_blk = q_f[:, :, q_start:q_end, :]  # [B, H, Bq, d]
        bq = q_end - q_start

        m = torch.full((bsz, n_heads, bq, 1), float("-inf"), dtype=torch.float32, device=q.device)
        ell = torch.zeros((bsz, n_heads, bq, 1), dtype=torch.float32, device=q.device)
        acc = torch.zeros((bsz, n_heads, bq, head_dim), dtype=torch.float32, device=q.device)

        # Causal skip: only iterate KV tiles that have at least one key <= q_end - 1
        for kv_start in range(0, q_end, tile_kv):
            kv_end = min(kv_start + tile_kv, seq_len)
            k_blk = k_f[:, :, kv_start:kv_end, :]
            v_blk = v_f[:, :, kv_start:kv_end, :]

            s_blk = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale
            # Apply causal mask only on diagonal-straddling tiles
            if kv_end - 1 > q_start:
                q_pos = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                k_pos = torch.arange(kv_start, kv_end, device=q.device).unsqueeze(0)
                mask = k_pos > q_pos
                s_blk = s_blk.masked_fill(mask, float("-inf"))

            tile_max = s_blk.amax(dim=-1, keepdim=True)
            m_new = torch.maximum(m, tile_max)
            safe_m_new = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.where(
                torch.isfinite(m),
                torch.exp(m - safe_m_new),
                torch.zeros_like(m),
            )
            p_blk = torch.where(
                torch.isfinite(s_blk),
                torch.exp(s_blk - safe_m_new),
                torch.zeros_like(s_blk),
            )
            ell = alpha * ell + p_blk.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p_blk, v_blk)
            m = m_new

        out_blk = acc / ell.clamp_min(1e-20)
        out_chunks.append(out_blk)

    return torch.cat(out_chunks, dim=2).to(dtype=orig_dtype)


def sdpa_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """A3: PyTorch native scaled_dot_product_attention (causal)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)


def probe_sdpa_backends(
    device: torch.device,
    dtype: torch.dtype,
    bsz: int = 2,
    n_heads: int = 4,
    seq_len: int = 256,
    head_dim: int = 64,
) -> dict[str, Any]:
    """Probe which PyTorch SDPA backends are supported for (device, dtype, shape)."""
    results: dict[str, Any] = {
        "device": str(device),
        "dtype": str(dtype),
        "shape": [bsz, n_heads, seq_len, head_dim],
        "backends": {},
    }
    q = torch.randn((bsz, n_heads, seq_len, head_dim), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if device.type != "cuda":
        results["backends"]["math"] = {"supported": True, "note": "CPU execution"}
        results["dispatched_default_guess"] = "math_or_cpu_sdpa"
        return results

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend_map = [
            ("flash_attention", SDPBackend.FLASH_ATTENTION),
            ("efficient_attention", SDPBackend.EFFICIENT_ATTENTION),
            ("cudnn_attention", SDPBackend.CUDNN_ATTENTION),
            ("math", SDPBackend.MATH),
        ]
        supported_order: list[str] = []
        for name, backend in backend_map:
            try:
                with sdpa_kernel(backend):
                    _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize(device)
                results["backends"][name] = {"supported": True, "error": None}
                supported_order.append(name)
            except Exception as exc:
                results["backends"][name] = {
                    "supported": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        # PyTorch priority order on CUDA: flash -> efficient -> cudnn -> math
        results["dispatched_default_guess"] = (
            supported_order[0] if supported_order else "none"
        )
    except Exception as exc:
        results["probe_error"] = f"{type(exc).__name__}: {exc}"
        results["dispatched_default_guess"] = "unknown"
    return results


def verify_causality(
    attn_fn,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seq_len: int = 32,
    perturb_pos: int = 16,
) -> dict[str, Any]:
    """Verify causal invariance by perturbing keys/values strictly after perturb_pos."""
    g = torch.Generator(device="cpu").manual_seed(6304)
    q = torch.randn((1, 2, seq_len, 16), generator=g).to(device=device, dtype=dtype)
    k1 = torch.randn((1, 2, seq_len, 16), generator=g).to(device=device, dtype=dtype)
    v1 = torch.randn((1, 2, seq_len, 16), generator=g).to(device=device, dtype=dtype)

    k2 = k1.clone()
    v2 = v1.clone()
    k2[:, :, perturb_pos + 1 :, :] += 5.0
    v2[:, :, perturb_pos + 1 :, :] -= 5.0

    with torch.no_grad():
        o1 = attn_fn(q, k1, v1)
        o2 = attn_fn(q, k2, v2)

    prefix_diff = float((o1[:, :, : perturb_pos + 1, :] - o2[:, :, : perturb_pos + 1, :]).abs().max().item())
    suffix_diff = float((o1[:, :, perturb_pos + 1 :, :] - o2[:, :, perturb_pos + 1 :, :]).abs().max().item())
    return {
        "causal_prefix_max_abs_diff": prefix_diff,
        "future_suffix_max_abs_diff": suffix_diff,
        "passed": prefix_diff <= 1e-5 and suffix_diff > 1e-3,
    }


def verify_extreme_score_stability(
    attn_fn,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Check numerical stability under extreme QK^T magnitudes (no NaN/Inf)."""
    q = torch.full((1, 2, 16, 16), 10.0, device=device, dtype=dtype)
    k = torch.full((1, 2, 16, 16), 10.0, device=device, dtype=dtype)
    v = torch.randn((1, 2, 16, 16), device=device, dtype=dtype)
    with torch.no_grad():
        out = attn_fn(q, k, v)
    finite = bool(torch.isfinite(out).all().item())
    return {"all_finite_under_extreme_scores": finite}
