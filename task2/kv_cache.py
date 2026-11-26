"""Task 2 Part C: KV-cache compression, real INT4/INT2 bit-packing, and byte accounting."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import DynamicCache

GROUP_SIZE = 64
SINK_TOKENS = 8
RECENT_TOKENS = 64
DEFAULT_INSTRUCT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def cache_num_layers(cache: DynamicCache) -> int:
    return len(cache.layers)


def cache_get_kv(cache: DynamicCache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer = cache.layers[layer_idx]
    return layer.keys, layer.values


def cache_from_kv_list(kv_list: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_list):
        cache.update(k, v, layer_idx)
    return cache


def clone_cache(cache: DynamicCache) -> DynamicCache:
    kv = []
    for i in range(cache_num_layers(cache)):
        k, v = cache_get_kv(cache, i)
        kv.append((k.clone(), v.clone()))
    return cache_from_kv_list(kv)


def kv_full_fp16_bytes(
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
) -> int:
    """M_16(L) = 4 * B * n_l * H_kv * d_h * L."""
    return int(4 * batch_size * n_layers * n_kv_heads * head_dim * seq_len)


def kv_quant_reference_bytes(
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bits: int,
    group_size: int = GROUP_SIZE,
    batch_size: int = 1,
) -> int:
    """Exact PA4 PDF p. 8 byte formula M_b(L) with FP16 trailing (L mod 64) residual."""
    n_complete_groups = seq_len // group_size
    rem = seq_len % group_size
    bytes_per_group = (group_size * bits) // 8 + 4  # packed payload + 2B scale + 2B offset
    per_channel_bytes = n_complete_groups * bytes_per_group + 2 * rem
    return int(2 * batch_size * n_layers * n_kv_heads * head_dim * per_channel_bytes)


def max_eviction_positions(
    seq_len: int,
    budget_bytes: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
) -> dict[str, int]:
    """Largest legal FP16 token count k <= seq_len whose payload + uint16 indices <= budget."""
    bytes_per_token_kv = 4 * n_layers * n_kv_heads * head_dim
    bytes_per_token_idx = 2 * n_layers  # per-layer uint16 position indices
    bytes_per_pos = bytes_per_token_kv + bytes_per_token_idx
    k = min(seq_len, budget_bytes // bytes_per_pos)
    actual_bytes = int(k * bytes_per_pos)
    return {
        "k_kept": int(k),
        "kv_payload_bytes": int(k * bytes_per_token_kv),
        "index_bytes": int(k * bytes_per_token_idx),
        "total_bytes": actual_bytes,
        "bytes_under_budget": int(budget_bytes - actual_bytes),
    }


def solve_hybrid_allocation(
    seq_len: int,
    budget_bytes: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    group_size: int = GROUP_SIZE,
    min_fp16: int = SINK_TOKENS + RECENT_TOKENS,
) -> dict[str, int]:
    """Choose largest selective FP16 count k_fp16 >= min_fp16 with INT2 low-bit groups <= budget.

    Remaining (seq_len - k_fp16) tokens form floor((seq_len - k_fp16) / 64) complete
    INT2 groups (20 bytes/group/channel) plus an incomplete FP16 residual (<64 tokens).
    Includes per-layer uint16 position indices (2 * n_layers * seq_len bytes).
    """
    n_channels = 2 * n_layers * n_kv_heads * head_dim
    index_bytes = 2 * n_layers * seq_len
    int2_group_bytes = (group_size * 2) // 8 + 4  # 16 + 4 = 20 bytes per 64-elem group

    best: dict[str, int] | None = None
    for k_sel in range(seq_len, min(min_fp16, seq_len) - 1, -1):
        rem_for_low = seq_len - k_sel
        n_groups = rem_for_low // group_size
        residual_fp16 = rem_for_low % group_size
        total_fp16_pos = k_sel + residual_fp16
        kv_bytes = n_channels * (n_groups * int2_group_bytes + 2 * total_fp16_pos)
        total_bytes = int(kv_bytes + index_bytes)
        if total_bytes <= budget_bytes:
            best = {
                "k_selective_fp16": int(k_sel),
                "n_int2_groups": int(n_groups),
                "int2_positions": int(n_groups * group_size),
                "residual_fp16_positions": int(residual_fp16),
                "total_fp16_positions": int(total_fp16_pos),
                "kv_bytes": int(kv_bytes),
                "index_bytes": int(index_bytes),
                "total_bytes": total_bytes,
                "bytes_under_budget": int(budget_bytes - total_bytes),
            }
            break
    if best is None:
        # Fallback if seq_len is tiny (e.g. unit tests)
        n_groups = max(0, (seq_len - min(min_fp16, seq_len)) // group_size)
        residual_fp16 = seq_len - n_groups * group_size
        kv_bytes = n_channels * (n_groups * int2_group_bytes + 2 * residual_fp16)
        best = {
            "k_selective_fp16": int(min(min_fp16, seq_len)),
            "n_int2_groups": int(n_groups),
            "int2_positions": int(n_groups * group_size),
            "residual_fp16_positions": int(max(0, residual_fp16 - min(min_fp16, seq_len))),
            "total_fp16_positions": int(residual_fp16),
            "kv_bytes": int(kv_bytes),
            "index_bytes": int(index_bytes),
            "total_bytes": int(kv_bytes + index_bytes),
            "bytes_under_budget": int(budget_bytes - (kv_bytes + index_bytes)),
        }
    return best


@dataclass
class PackedKVTensor:
    """Real bit-packed INT4/INT2 representation of a single K or V tensor [B, H_kv, L, d_h]."""

    bits: int
    axis: str  # "channel" or "token"
    group_size: int
    orig_shape: tuple[int, int, int, int]
    packed_q: torch.Tensor  # uint8
    scale: torch.Tensor  # float16
    offset: torch.Tensor  # float16
    residual_fp16: torch.Tensor  # float16 [B, H_kv, r, d_h]

    def serialized_bytes(self) -> int:
        total = 0
        for t in (self.packed_q, self.scale, self.offset, self.residual_fp16):
            total += int(t.numel()) * int(t.element_size())
        return total


def _pack_uint8(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer codes in [0, 2^bits - 1] along the last dimension into uint8."""
    q_u8 = q.to(torch.uint8)
    if bits == 4:
        return (q_u8[..., 0::2] & 0x0F) | ((q_u8[..., 1::2] & 0x0F) << 4)
    if bits == 2:
        return (
            (q_u8[..., 0::4] & 0x03)
            | ((q_u8[..., 1::4] & 0x03) << 2)
            | ((q_u8[..., 2::4] & 0x03) << 4)
            | ((q_u8[..., 3::4] & 0x03) << 6)
        )
    raise ValueError(f"Unsupported KV pack bits={bits}; expected 2 or 4.")


def _unpack_uint8(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack uint8 codes along the last dimension back to float32 integer values."""
    if bits == 4:
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1).float()
    if bits == 2:
        c0 = packed & 0x03
        c1 = (packed >> 2) & 0x03
        c2 = (packed >> 4) & 0x03
        c3 = (packed >> 6) & 0x03
        return torch.stack((c0, c1, c2, c3), dim=-1).reshape(*packed.shape[:-1], -1).float()
    raise ValueError(f"Unsupported KV unpack bits={bits}; expected 2 or 4.")


def pack_kv_tensor(
    x: torch.Tensor,
    bits: int,
    axis: str,
    group_size: int = GROUP_SIZE,
) -> PackedKVTensor:
    """Physically quantize and bit-pack K or V [B, H_kv, L, d_h] with trailing FP16 residual.

    - axis == 'channel': groups of 64 along token axis L (KIVI channel-wise K).
    - axis == 'token': groups of 64 along feature axis d_h (KIVI token-wise V / token-wise K),
      applying the identical trailing (L mod 64) FP16 rule so serialized bytes match M_b(L).
    """
    bsz, n_kv, seq_len, head_dim = x.shape
    n_complete = (seq_len // group_size) * group_size
    main = x[:, :, :n_complete, :]
    residual = x[:, :, n_complete:, :].to(torch.float16).contiguous()

    if n_complete == 0:
        empty_u8 = torch.empty((0,), dtype=torch.uint8, device=x.device)
        empty_fp16 = torch.empty((0,), dtype=torch.float16, device=x.device)
        return PackedKVTensor(
            bits=bits,
            axis=axis,
            group_size=group_size,
            orig_shape=(bsz, n_kv, seq_len, head_dim),
            packed_q=empty_u8,
            scale=empty_fp16,
            offset=empty_fp16,
            residual_fp16=residual,
        )

    qmax = float((1 << bits) - 1)
    if axis == "channel":
        # [B, H_kv, n_g, G, d_h] -> transpose to [B, H_kv, n_g, d_h, G]
        n_g = n_complete // group_size
        grouped = main.float().reshape(bsz, n_kv, n_g, group_size, head_dim).transpose(-2, -1)
    elif axis == "token":
        if head_dim % group_size != 0:
            raise ValueError(f"head_dim={head_dim} must be divisible by group_size={group_size}")
        n_dg = head_dim // group_size
        grouped = main.float().reshape(bsz, n_kv, n_complete, n_dg, group_size)
    else:
        raise ValueError(f"Unknown KV quantization axis '{axis}'")

    vmin = grouped.amin(dim=-1, keepdim=True)
    vmax = grouped.amax(dim=-1, keepdim=True)
    span = vmax - vmin
    is_const = span <= 1e-12
    raw_scale = torch.where(is_const, torch.ones_like(span), span / qmax)
    scale_fp16 = raw_scale.to(torch.float16)
    scale_f = scale_fp16.float()
    scale_f = torch.where(
        (~is_const) & (scale_f == 0.0),
        torch.full_like(scale_f, float(torch.finfo(torch.float16).tiny)),
        scale_f,
    )
    offset_fp16 = vmin.to(torch.float16)
    offset_f = offset_fp16.float()

    q = torch.clamp(torch.round((grouped - offset_f) / scale_f), 0.0, qmax)
    packed_q = _pack_uint8(q, bits=bits).contiguous()

    return PackedKVTensor(
        bits=bits,
        axis=axis,
        group_size=group_size,
        orig_shape=(bsz, n_kv, seq_len, head_dim),
        packed_q=packed_q,
        scale=scale_fp16.squeeze(-1).contiguous(),
        offset=offset_fp16.squeeze(-1).contiguous(),
        residual_fp16=residual,
    )


def unpack_kv_tensor(packed: PackedKVTensor, out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Unpack and dequantize a PackedKVTensor back to [B, H_kv, L, d_h]."""
    bsz, n_kv, seq_len, head_dim = packed.orig_shape
    n_complete = (seq_len // packed.group_size) * packed.group_size
    if n_complete == 0:
        return packed.residual_fp16.to(dtype=out_dtype)

    q_f = _unpack_uint8(packed.packed_q, bits=packed.bits)
    deq = q_f * packed.scale.float().unsqueeze(-1) + packed.offset.float().unsqueeze(-1)
    if packed.axis == "channel":
        n_g = n_complete // packed.group_size
        main = deq.transpose(-2, -1).reshape(bsz, n_kv, n_complete, head_dim)
    else:
        main = deq.reshape(bsz, n_kv, n_complete, head_dim)

    main_out = main.to(dtype=out_dtype)
    if packed.residual_fp16.shape[2] == 0:
        return main_out
    return torch.cat([main_out, packed.residual_fp16.to(dtype=out_dtype)], dim=2)


def extract_five_digit_code(text: str) -> str:
    """Deterministic 5-digit target code extractor."""
    m = re.search(r"\b(\d{5})\b", text)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{5})", text)
    return m2.group(1) if m2 else ""


def split_prompt_and_map_evidence(
    tokenizer: Any,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Jointly tokenize chat prompt, split pre-question context vs suffix, and map evidence span.

    evidence_char_start / evidence_char_end are Python Unicode character slices into
    user_text, mapped onto Qwen token indices via tokenizer offset_mapping.
    """
    user_text = case["user_text"]
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            rendered = user_text
    else:
        rendered = user_text

    user_char_start = rendered.find(user_text)
    if user_char_start < 0:
        user_char_start = 0

    # Locate the final question line inside user_text for pre-question vs suffix split
    q_markers = ["\nQuestion:", "\nFinal question:", "\nQ:", "\nWhat is "]
    q_rel_pos = -1
    for mk in q_markers:
        pos = user_text.rfind(mk)
        if pos > q_rel_pos:
            q_rel_pos = pos + 1  # start at 'Question:' (after newline)
    if q_rel_pos < 0:
        last_nl = user_text.rstrip().rfind("\n")
        q_rel_pos = (last_nl + 1) if last_nl >= 0 else max(0, len(user_text) - 80)

    question_char_abs = user_char_start + q_rel_pos
    ev_start_abs = user_char_start + int(case["evidence_char_start"])
    ev_end_abs = user_char_start + int(case["evidence_char_end"])

    enc = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = [(int(s), int(e)) for s, e in enc["offset_mapping"]]

    # Split index: first token whose character end exceeds question_char_abs
    split_idx = len(input_ids) - 1
    for i, (s, e) in enumerate(offsets):
        if e > question_char_abs:
            split_idx = i
            break
    split_idx = max(1, min(split_idx, len(input_ids) - 1))

    # Evidence token indices inside pre-question context [0, split_idx)
    evidence_token_indices = [
        i for i, (s, e) in enumerate(offsets[:split_idx]) if e > ev_start_abs and s < ev_end_abs
    ]

    return {
        "rendered_prompt": rendered,
        "full_ids": input_ids,
        "pre_ids": input_ids[:split_idx],
        "question_ids": input_ids[split_idx:],
        "pre_len": split_idx,
        "question_len": len(input_ids) - split_idx,
        "full_len": len(input_ids),
        "evidence_token_indices": evidence_token_indices,
    }


def _apply_rotary_pos_emb_q(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply standard NeoX/Qwen2 rotary position embedding to query tensor [B, H, L_q, d_h]."""
    half = q.shape[-1] // 2
    q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
    return (q * cos.unsqueeze(1)) + (q_rot * sin.unsqueeze(1))


def compute_question_aware_importance(
    model: torch.nn.Module,
    pre_cache: DynamicCache,
    question_ids: torch.Tensor,
    pre_len: int,
) -> list[torch.Tensor]:
    """Compute per-layer question-aware importance scores [L] using masked FP32 QK^T attention.

    Keeps the working SDPA model intact: captures q_proj outputs via forward hooks on
    question_ids, applies RoPE at positions [pre_len .. pre_len + L_q - 1], and scores
    against the existing RoPE-positioned cached keys in FP32.
    """
    n_layers = cache_num_layers(pre_cache)
    n_q_heads = int(model.config.num_attention_heads)
    n_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.hidden_size // n_q_heads)
    gqa_groups = n_q_heads // n_kv_heads

    captured_q: dict[int, torch.Tensor] = {}
    handles = []
    layers = model.model.layers

    def _make_q_hook(l_idx: int):
        def _hook(_mod: torch.nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            captured_q[l_idx] = out.detach()

        return _hook

    for l_idx, layer in enumerate(layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(_make_q_hook(l_idx)))

    q_len = question_ids.shape[1]
    pos_ids = torch.arange(pre_len, pre_len + q_len, device=question_ids.device).unsqueeze(0)
    tmp_cache = clone_cache(pre_cache)
    try:
        with torch.no_grad():
            model(
                input_ids=question_ids,
                past_key_values=tmp_cache,
                position_ids=pos_ids,
                use_cache=True,
            )
    finally:
        for h in handles:
            h.remove()

    # Compute RoPE (cos, sin) for question positions using the model's rotary_emb
    dummy_h = torch.empty(
        (1, q_len, head_dim), device=question_ids.device, dtype=next(model.parameters()).dtype
    )
    cos, sin = model.model.rotary_emb(dummy_h, pos_ids)

    scores_per_layer: list[torch.Tensor] = []
    scale = 1.0 / math.sqrt(head_dim)
    for l_idx in range(n_layers):
        q_raw = captured_q[l_idx].reshape(1, q_len, n_q_heads, head_dim).transpose(1, 2)
        q_rope = _apply_rotary_pos_emb_q(q_raw.float(), cos.float(), sin.float())
        k_pre, _ = cache_get_kv(pre_cache, l_idx)
        k_exp = k_pre.float().repeat_interleave(gqa_groups, dim=1)  # [1, H_q, L, d_h]
        # FP32 attention over pre-question keys
        attn_logits = torch.matmul(q_rope, k_exp.transpose(-2, -1)) * scale  # [1, H_q, L_q, L]
        attn_probs = F.softmax(attn_logits, dim=-1)
        # Aggregate across question tokens and query heads -> [L]
        imp = attn_probs.mean(dim=(0, 1, 2))
        scores_per_layer.append(imp)
    return scores_per_layer


def select_eviction_indices(
    seq_len: int,
    k_kept: int,
    importance: torch.Tensor | None = None,
    n_sink: int = SINK_TOKENS,
    n_recent: int = RECENT_TOKENS,
) -> torch.Tensor:
    """Select sorted token indices to keep (8 sink + 64 recent + top remaining)."""
    if k_kept >= seq_len:
        return torch.arange(seq_len, dtype=torch.long)
    sink_end = min(n_sink, seq_len)
    recent_start = max(sink_end, seq_len - n_recent)
    mandatory = list(range(0, sink_end)) + list(range(recent_start, seq_len))
    if len(mandatory) >= k_kept:
        return torch.tensor(mandatory[:k_kept], dtype=torch.long)

    rem_budget = k_kept - len(mandatory)
    middle = list(range(sink_end, recent_start))
    if importance is None:
        # Sink + recent: fill remaining budget with most recent middle positions
        chosen_mid = middle[-rem_budget:]
    else:
        mid_scores = importance[sink_end:recent_start]
        _, top_local = torch.topk(mid_scores, k=min(rem_budget, len(middle)), largest=True)
        chosen_mid = [sink_end + int(i) for i in top_local.tolist()]

    all_kept = sorted(set(mandatory + chosen_mid))
    return torch.tensor(all_kept, dtype=torch.long)
