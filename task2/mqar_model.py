"""Small 2-layer decoder-only Transformer for Task 2 Part B (MQAR)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from task2.mqar_data import NUM_VALUE_CLASSES

ATTN_MODES = ("dense", "window64", "linear_elu", "local_dilated", "b3_local_prefix")


def build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def build_sparse_mask(
    attn_mode: str,
    seq_len: int,
    n_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """Build boolean attention mask [1, H, T, T] where True = allowed to attend.

    Explicitly materialized as a dense boolean mask for correctness; measured memory
    is NOT described as an efficient sparse kernel.
    """
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    dist = q_pos - k_pos
    causal = dist >= 0

    if attn_mode == "window64":
        allowed = causal & (dist < 64)
        return allowed.unsqueeze(0).unsqueeze(0).expand(1, n_heads, seq_len, seq_len)

    if attn_mode == "local_dilated":
        if n_heads != 4:
            raise ValueError(f"local_dilated requires 4 heads, got {n_heads}")
        local_m = causal & (dist < 64)
        dilated_m = causal & (dist <= 252) & ((dist % 4) == 0)
        return torch.stack([local_m, local_m, dilated_m, dilated_m], dim=0).unsqueeze(0)

    if attn_mode == "b3_local_prefix":
        if n_heads != 4:
            raise ValueError(f"b3_local_prefix requires 4 heads, got {n_heads}")
        local_m = causal & (dist < 64)
        # 32 global prefix anchors [0, 32) + 32 recent positions [i-31, i] = <=64 attended positions
        prefix_local_m = causal & ((k_pos < 32) | (dist < 32))
        return torch.stack([local_m, local_m, prefix_local_m, prefix_local_m], dim=0).unsqueeze(0)

    raise ValueError(f"No boolean mask defined for attn_mode='{attn_mode}'")


def causal_elu_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 256,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Causal ELU+1 linear attention with chunked recurrence (O(T * d^2) compute, O(d^2) state)."""
    orig_dtype = q.dtype
    bsz, n_heads, seq_len, d_h = q.shape
    qf = F.elu(q.float()) + 1.0
    kf = F.elu(k.float()) + 1.0
    vf = v.float()

    state = torch.zeros((bsz, n_heads, d_h, d_h), dtype=torch.float32, device=q.device)
    z_state = torch.zeros((bsz, n_heads, d_h), dtype=torch.float32, device=q.device)

    out_chunks: list[torch.Tensor] = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        qc = qf[:, :, start:end, :]  # [B, H, C, d]
        kc = kf[:, :, start:end, :]
        vc = vf[:, :, start:end, :]

        # Intra-chunk causal contribution via (Q K^T) masked lower-triangular
        intra_scores = torch.matmul(qc, kc.transpose(-2, -1))
        c_len = end - start
        tril = torch.tril(torch.ones((c_len, c_len), dtype=torch.bool, device=q.device))
        intra_scores = intra_scores.masked_fill(~tril, 0.0)
        num_intra = torch.matmul(intra_scores, vc)
        den_intra = intra_scores.sum(dim=-1, keepdim=True)

        # Inter-chunk contribution from recurrent state carried from previous chunks
        num_inter = torch.matmul(qc, state)
        den_inter = torch.einsum("bhcd,bhd->bhc", qc, z_state).unsqueeze(-1)

        out_c = (num_intra + num_inter) / (den_intra + den_inter + eps)
        out_chunks.append(out_c)

        # Update persistent recurrent state
        state = state + torch.einsum("bhcd,bhce->bhde", kc, vc)
        z_state = z_state + kc.sum(dim=2)

    return torch.cat(out_chunks, dim=2).to(dtype=orig_dtype)


class MQARAttention(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, attn_mode: str = "dense") -> None:
        super().__init__()
        if attn_mode not in ATTN_MODES:
            raise ValueError(f"Unsupported attn_mode '{attn_mode}'")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_mode = attn_mode
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self._mask_cache: dict[tuple[str, int, torch.device], torch.Tensor] = {}

    def _get_mask(self, mode: str, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (mode, seq_len, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = build_sparse_mask(mode, seq_len, self.n_heads, device)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor, override_attn_mode: str | None = None) -> torch.Tensor:
        mode = override_attn_mode or self.attn_mode
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, T, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mode == "dense":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif mode == "linear_elu":
            out = causal_elu_linear_attention(q, k, v)
        else:
            mask = self._get_mask(mode, seq_len, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)

        out = out.transpose(1, 2).contiguous().reshape(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class MQARBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, attn_mode: str = "dense") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MQARAttention(d_model=d_model, n_heads=n_heads, attn_mode=attn_mode)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, override_attn_mode: str | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), override_attn_mode=override_attn_mode)
        x = x + self.mlp(self.norm2(x))
        return x


class MQARTransformer(nn.Module):
    """2-layer pre-norm decoder-only Transformer (d=128, H=4, d_h=32, vocab=256)."""

    def __init__(
        self,
        attn_mode: str = "dense",
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 8192,
    ) -> None:
        super().__init__()
        self.attn_mode = attn_mode
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_emb", build_sinusoidal_pe(max_len, d_model), persistent=False)
        self.blocks = nn.ModuleList(
            [MQARBlock(d_model=d_model, n_heads=n_heads, attn_mode=attn_mode) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, NUM_VALUE_CLASSES)

    def forward(
        self,
        input_ids: torch.Tensor,
        override_attn_mode: str | None = None,
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        x = self.tok_emb(input_ids) + self.pos_emb[:seq_len].to(dtype=self.tok_emb.weight.dtype).unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, override_attn_mode=override_attn_mode)
        x = self.final_norm(x)
        return self.classifier(x)  # [B, T, 64]

    def forward_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.classifier(self.final_norm(x))


def theoretical_attention_cost(
    attn_mode: str,
    seq_len: int = 256,
    n_heads: int = 4,
    head_dim: int = 32,
    n_layers: int = 2,
    bytes_per_elem: int = 4,
) -> dict[str, Any]:
    """Compute theoretical attended KV pairs and persistent state bytes per layer."""
    if attn_mode == "dense":
        pairs_per_head = seq_len * (seq_len + 1) // 2
        total_pairs = n_layers * n_heads * pairs_per_head
        # Full KV cache across sequence
        state_bytes_per_layer = 2 * seq_len * n_heads * head_dim * bytes_per_elem
        max_kv_per_query = seq_len
    elif attn_mode in ("window64", "local_dilated", "b3_local_prefix"):
        mask = build_sparse_mask(attn_mode, seq_len, n_heads, torch.device("cpu"))
        total_pairs = int(mask.sum().item()) * n_layers
        max_kv_per_query = int(mask.sum(dim=-1).max().item())
        # Theoretical minimal KV window/buffer (note: implementation uses dense boolean mask)
        state_bytes_per_layer = 2 * min(seq_len, max_kv_per_query) * n_heads * head_dim * bytes_per_elem
    elif attn_mode == "linear_elu":
        total_pairs = 0  # Kernel feature map recurrence, no explicit pairwise QK attention
        max_kv_per_query = 0
        # Persistent recurrent state S [H, d, d] + z [H, d]
        state_bytes_per_layer = n_heads * (head_dim * head_dim + head_dim) * bytes_per_elem
    else:
        raise ValueError(f"Unknown attn_mode '{attn_mode}'")

    return {
        "attn_mode": attn_mode,
        "seq_len": int(seq_len),
        "theoretical_total_attended_pairs": int(total_pairs),
        "max_kv_positions_per_query": int(max_kv_per_query),
        "theoretical_state_bytes_per_layer": int(state_bytes_per_layer),
        "theoretical_total_state_bytes": int(state_bytes_per_layer * n_layers),
    }
