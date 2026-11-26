"""Weight-only PTQ (Q1), AWQ-style activation scaling (Q2), and mixed precision (Q3)."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn

from common.timing import sync_device
from task1.model_utils import (
    EligibleProjection,
    deduplicate_parameters,
    enumerate_eligible_projections,
)
from task1.pruning import restore_eligible_weights

AWQ_ALPHA_GRID: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
Q3_ALLOWED_BITS: tuple[int, ...] = (2, 3, 4, 8)
QUANT_CONVENTIONS = {
    "scheme": "group_wise_asymmetric_weight_only_ptq",
    "group_axis": "input_columns",
    "include_zero_in_range": True,
    "rounding": "torch.round (ties_to_even)",
    "zero_point_from_stored_fp16_scale": True,
    "all_zero_group_scale": 1.0,
    "metadata_per_group": "FP16 scale (2 bytes) + uint8 zero-point (1 byte) = 3 bytes/group",
    "payload_packing": "ceil(total_groups * group_size * bits / 8) bytes per matrix",
    "awq_folding": (
        "down_proj inverse input scales folded algebraically into up_proj output rows "
        "before up_proj quantization; non-foldable projections (q/k/v/o/gate/up) "
        "serialize FP16 per-channel compensation vectors (2 * in_features bytes) "
        "and include them in the matched packed-byte budget."
    ),
}


class ScaledLinearWrapper(nn.Module):
    """Wraps an nn.Linear with a per-input-channel FP16 inverse compensation scale."""

    def __init__(self, linear: nn.Linear, inv_scale: torch.Tensor) -> None:
        super().__init__()
        self.linear = linear
        self.register_buffer(
            "inv_scale",
            inv_scale.detach().to(device=linear.weight.device, dtype=linear.weight.dtype),
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.linear.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x * self.inv_scale)


def unwrap_scaled_linears(model: nn.Module) -> int:
    """Restore any ScaledLinearWrapper modules back to their underlying nn.Linear."""
    unwrapped = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, ScaledLinearWrapper):
                setattr(parent, child_name, child.linear)
                unwrapped += 1
    return unwrapped


def _set_module_by_name(model: nn.Module, full_name: str, new_module: nn.Module) -> None:
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def quantize_dequantize_weight(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 128,
) -> torch.Tensor:
    """Group-wise asymmetric weight-only fake quantization along input columns.

    Fixed conventions across Q1/Q2/Q3:
    - Pad input columns with zeros to a multiple of group_size if needed.
    - Include 0.0 in [w_min, w_max].
    - Store scale in FP16 (all-zero groups -> scale = 1.0).
    - Compute integer zero-point from the stored FP16 scale with ties-to-even rounding.
    - Reconstruct to weight.dtype (FP16 on GPU, FP32 in CPU unit tests).
    """
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"Unsupported bit width {bits}; expected one of (2, 3, 4, 8).")
    orig_dtype = weight.dtype
    d_out, d_in = weight.shape
    w = weight.detach().float()

    pad = (group_size - (d_in % group_size)) % group_size
    if pad > 0:
        w = torch.nn.functional.pad(w, (0, pad), value=0.0)
    d_in_padded = w.shape[1]
    n_groups = d_in_padded // group_size
    w_g = w.reshape(d_out, n_groups, group_size)

    qmax = float((1 << bits) - 1)
    w_min = torch.minimum(w_g.amin(dim=-1, keepdim=True), torch.zeros(1, device=w.device))
    w_max = torch.maximum(w_g.amax(dim=-1, keepdim=True), torch.zeros(1, device=w.device))
    span = w_max - w_min

    # Safe handling for all-zero / degenerate groups
    is_zero_group = span <= 1e-12
    raw_scale = torch.where(is_zero_group, torch.ones_like(span), span / qmax)
    scale_fp16 = raw_scale.to(torch.float16).float()
    # Guard against FP16 underflow to 0 on tiny nonzero groups
    scale_fp16 = torch.where(
        (~is_zero_group) & (scale_fp16 == 0.0),
        torch.full_like(scale_fp16, float(torch.finfo(torch.float16).tiny)),
        scale_fp16,
    )

    zero_point = torch.clamp(torch.round(-w_min / scale_fp16), 0.0, qmax)
    q = torch.clamp(torch.round(w_g / scale_fp16) + zero_point, 0.0, qmax)
    w_hat = scale_fp16 * (q - zero_point)
    w_hat = w_hat.reshape(d_out, d_in_padded)[:, :d_in]
    return w_hat.to(dtype=orig_dtype)


def compute_matrix_packed_bytes(
    d_out: int,
    d_in: int,
    bits: int,
    group_size: int = 128,
    has_compensation_vector: bool = False,
) -> dict[str, int]:
    """Exact serialized packed byte accounting for one quantized eligible matrix."""
    n_groups_per_row = (d_in + group_size - 1) // group_size
    total_groups = d_out * n_groups_per_row
    padded_elems = total_groups * group_size
    payload_bytes = (padded_elems * bits + 7) // 8
    scale_bytes = total_groups * 2  # FP16 scale
    zp_bytes = total_groups * 1  # uint8 zero-point
    metadata_bytes = scale_bytes + zp_bytes
    comp_bytes = (d_in * 2) if has_compensation_vector else 0
    return {
        "n_groups": int(total_groups),
        "padded_elements": int(padded_elems),
        "payload_bytes": int(payload_bytes),
        "scale_bytes": int(scale_bytes),
        "zp_bytes": int(zp_bytes),
        "metadata_bytes": int(metadata_bytes),
        "compensation_bytes": int(comp_bytes),
        "total_bytes": int(payload_bytes + metadata_bytes + comp_bytes),
    }


def compute_quantized_model_storage(
    model: nn.Module,
    bits_by_name: dict[str, int],
    group_size: int = 128,
    non_folded_comp_names: set[str] | None = None,
) -> dict[str, Any]:
    """Compute exact packed weight storage including metadata, compensation, and FP16 tail."""
    non_folded = non_folded_comp_names or set()
    eligible = enumerate_eligible_projections(model)
    unique_params, tied_aliases = deduplicate_parameters(model)
    eligible_ptrs = {int(item.module.weight.data_ptr()) for item in eligible}

    uncompressed_unique_params = 0
    for name, p in unique_params:
        if int(p.data_ptr()) in eligible_ptrs:
            continue
        # Exclude temporary ScaledLinearWrapper buffers (none are nn.Parameters, only buffers)
        uncompressed_unique_params += int(p.numel())
    uncompressed_fp16_bytes = uncompressed_unique_params * 2

    total_eligible_params = 0
    total_payload_bytes = 0
    total_metadata_bytes = 0
    total_comp_bytes = 0
    weighted_bits_sum = 0.0

    for item in eligible:
        b = int(bits_by_name[item.full_name])
        m = compute_matrix_packed_bytes(
            item.out_features,
            item.in_features,
            bits=b,
            group_size=group_size,
            has_compensation_vector=(item.full_name in non_folded),
        )
        total_eligible_params += item.numel
        total_payload_bytes += m["payload_bytes"]
        total_metadata_bytes += m["metadata_bytes"]
        total_comp_bytes += m["compensation_bytes"]
        weighted_bits_sum += b * item.numel

    eligible_packed_bytes = total_payload_bytes + total_metadata_bytes + total_comp_bytes
    total_packed_bytes = eligible_packed_bytes + uncompressed_fp16_bytes
    dense_fp16_total_bytes = (total_eligible_params + uncompressed_unique_params) * 2

    return {
        "storage_accounting_type": "logical_serialized_packed_estimate_not_gpu_vram",
        "quant_conventions": QUANT_CONVENTIONS,
        "group_size": int(group_size),
        "tied_weights_counted_once": True,
        "tied_alias_count": len(tied_aliases),
        "unique_total_params": int(total_eligible_params + uncompressed_unique_params),
        "eligible_weight_params": int(total_eligible_params),
        "uncompressed_unique_params": int(uncompressed_unique_params),
        "nominal_bits_per_eligible_weight": float(weighted_bits_sum / max(total_eligible_params, 1)),
        "effective_bits_per_eligible_weight_incl_meta": float(
            8.0 * eligible_packed_bytes / max(total_eligible_params, 1)
        ),
        "eligible_payload_bytes": int(total_payload_bytes),
        "eligible_metadata_bytes": int(total_metadata_bytes),
        "compensation_scale_bytes": int(total_comp_bytes),
        "eligible_packed_total_bytes": int(eligible_packed_bytes),
        "uncompressed_fp16_bytes": int(uncompressed_fp16_bytes),
        "dense_fp16_total_bytes": int(dense_fp16_total_bytes),
        "serialized_packed_total_bytes": int(total_packed_bytes),
        "compression_ratio_vs_dense_fp16": float(
            dense_fp16_total_bytes / max(total_packed_bytes, 1)
        ),
    }


def collect_calibration_mean_abs_inputs(
    model: nn.Module,
    calib_records: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    """Collect per-input-channel mean(|X_{:, j}|) across all calibration tokens (AWQ)."""
    model.eval()
    eligible = enumerate_eligible_projections(model)
    abs_sums: dict[str, torch.Tensor] = {
        item.full_name: torch.zeros(item.in_features, dtype=torch.float64, device=device)
        for item in eligible
    }
    token_counts: dict[str, int] = {item.full_name: 0 for item in eligible}
    handles = []

    def _make_hook(name: str):
        def _hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0].detach()
            x_2d = x.reshape(-1, x.shape[-1]).to(dtype=torch.float64)
            abs_sums[name].add_(x_2d.abs().sum(dim=0))
            token_counts[name] += int(x_2d.shape[0])

        return _hook

    for item in eligible:
        handles.append(item.module.register_forward_pre_hook(_make_hook(item.full_name)))

    sync_device(device)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            for rec in calib_records:
                ids = torch.tensor([rec["token_ids"]], dtype=torch.long, device=device)
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    sync_device(device)
    elapsed_sec = time.perf_counter() - t0

    mean_abs = {
        name: (val / max(token_counts[name], 1)).to(dtype=torch.float32, device="cpu")
        for name, val in abs_sums.items()
    }
    return mean_abs, float(elapsed_sec)


def compute_awq_channel_scale(
    mean_abs_x: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute AWQ channel scale s = s_X^alpha / sqrt(max * min) in float32."""
    if alpha == 0.0:
        return torch.ones_like(mean_abs_x, dtype=torch.float32)
    s = mean_abs_x.float().clamp_min(1e-5).pow(alpha)
    norm_factor = torch.sqrt(s.max() * s.min()).clamp_min(1e-5)
    return (s / norm_factor).clamp(1e-4, 1e4)


def search_awq_scales_for_bits(
    model: nn.Module,
    clean_weights: dict[str, torch.Tensor],
    mean_abs_x: dict[str, torch.Tensor],
    calib_records: list[dict[str, Any]],
    device: torch.device,
    bits: int,
    group_size: int = 128,
    alpha_grid: tuple[float, ...] = AWQ_ALPHA_GRID,
) -> dict[str, Any]:
    """Search best AWQ exponent alpha per eligible projection on calibration activations."""
    unwrap_scaled_linears(model)
    restore_eligible_weights(model, clean_weights)
    model.eval()
    eligible = enumerate_eligible_projections(model)

    # Evaluate each alpha across all 196 projections in a single calibration pass per alpha
    err_sq_by_alpha: dict[float, dict[str, float]] = {
        float(a): {item.full_name: 0.0 for item in eligible} for a in alpha_grid
    }
    ref_sq: dict[str, float] = {item.full_name: 0.0 for item in eligible}

    sync_device(device)
    t0 = time.perf_counter()
    for alpha_idx, alpha in enumerate(alpha_grid):
        alpha_f = float(alpha)
        delta_w_map: dict[str, torch.Tensor] = {}
        for item in eligible:
            w = item.module.weight.detach()
            s = compute_awq_channel_scale(mean_abs_x[item.full_name], alpha_f).to(
                device=w.device, dtype=torch.float32
            )
            w_scaled = (w.float() * s.unsqueeze(0)).to(dtype=w.dtype)
            w_q = quantize_dequantize_weight(w_scaled, bits=bits, group_size=group_size)
            w_eff = (w_q.float() / s.unsqueeze(0)).to(dtype=w.dtype)
            delta_w_map[item.full_name] = w_eff - w

        handles = []

        def _make_hook(name: str, record_ref: bool):
            dw = delta_w_map[name]

            def _hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                x = inputs[0].detach()
                dy = torch.nn.functional.linear(x, dw)
                err_sq_by_alpha[alpha_f][name] += float(dy.float().pow(2).sum().item())
                if record_ref:
                    ref_sq[name] += float(output.detach().float().pow(2).sum().item())

            return _hook

        for item in eligible:
            handles.append(
                item.module.register_forward_hook(
                    _make_hook(item.full_name, record_ref=(alpha_idx == 0))
                )
            )
        try:
            with torch.no_grad():
                for rec in calib_records:
                    ids = torch.tensor([rec["token_ids"]], dtype=torch.long, device=device)
                    model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        del delta_w_map

    sync_device(device)
    search_time_sec = time.perf_counter() - t0

    best_alpha: dict[str, float] = {}
    best_err_sq: dict[str, float] = {}
    best_rel_l2: dict[str, float] = {}
    per_proj: list[dict[str, Any]] = []
    for item in eligible:
        name = item.full_name
        chosen_a = min(alpha_grid, key=lambda a: err_sq_by_alpha[float(a)][name])
        e2 = err_sq_by_alpha[float(chosen_a)][name]
        r2 = ref_sq[name]
        rel = math.sqrt(e2 / max(r2, 1e-12))
        best_alpha[name] = float(chosen_a)
        best_err_sq[name] = float(e2)
        best_rel_l2[name] = float(rel)
        per_proj.append(
            {
                "layer_idx": item.layer_idx,
                "full_name": name,
                "proj_name": item.proj_name,
                "best_alpha": float(chosen_a),
                "err_sq": float(e2),
                "ref_sq": float(r2),
                "rel_l2_error": float(rel),
                "alpha_0_rel_l2_error": float(
                    math.sqrt(err_sq_by_alpha[float(alpha_grid[0])][name] / max(r2, 1e-12))
                ),
            }
        )

    return {
        "bits": int(bits),
        "group_size": int(group_size),
        "alpha_grid": [float(a) for a in alpha_grid],
        "search_time_sec": float(search_time_sec),
        "best_alpha_by_name": best_alpha,
        "best_err_sq_by_name": best_err_sq,
        "best_rel_l2_by_name": best_rel_l2,
        "mean_rel_l2_error": float(sum(best_rel_l2.values()) / max(len(best_rel_l2), 1)),
        "per_projection": per_proj,
    }


def apply_q1_uniform_quantization(
    model: nn.Module,
    bits: int,
    group_size: int = 128,
) -> dict[str, Any]:
    """Apply Q1 uniform group-wise asymmetric PTQ in-place (fake-quantized FP16)."""
    unwrap_scaled_linears(model)
    eligible = enumerate_eligible_projections(model)
    t0 = time.perf_counter()
    bits_by_name: dict[str, int] = {}
    with torch.no_grad():
        for item in eligible:
            w_hat = quantize_dequantize_weight(
                item.module.weight, bits=bits, group_size=group_size
            )
            item.module.weight.copy_(w_hat)
            bits_by_name[item.full_name] = int(bits)
    elapsed_sec = time.perf_counter() - t0
    return {
        "bits_by_name": bits_by_name,
        "non_folded_comp_names": set(),
        "quant_apply_time_sec": float(elapsed_sec),
    }


def apply_awq_quantization(
    model: nn.Module,
    clean_weights: dict[str, torch.Tensor],
    mean_abs_x: dict[str, torch.Tensor],
    bits_by_name: dict[str, int],
    alpha_by_name: dict[str, float],
    group_size: int = 128,
    verify_sample_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Apply AWQ scaling, algebraic down_proj->up_proj folding, equivalence check, and PTQ.

    Folding rule:
    - For every Transformer block i, down_proj's inverse input scale (1 / s_down) is
      folded into up_proj's output rows BEFORE up_proj is quantized.
    - For all other projections (q/k/v/o/gate/up), inverse scales are applied via
      ScaledLinearWrapper and their FP16 channel vectors are counted in packed bytes.
    """
    unwrap_scaled_linears(model)
    restore_eligible_weights(model, clean_weights)
    model.eval()
    eligible = enumerate_eligible_projections(model)
    by_layer_proj: dict[tuple[int, str], EligibleProjection] = {
        (item.layer_idx, item.proj_name): item for item in eligible
    }

    # Reference outputs BEFORE rescaling (for full-precision equivalence verification)
    ref_logits = None
    if verify_sample_ids is not None:
        with torch.no_grad():
            ref_logits = model(input_ids=verify_sample_ids, use_cache=False).logits.detach().float()

    t0 = time.perf_counter()
    scales_by_name: dict[str, torch.Tensor] = {}
    for item in eligible:
        alpha = float(alpha_by_name[item.full_name])
        s = compute_awq_channel_scale(mean_abs_x[item.full_name], alpha)
        scales_by_name[item.full_name] = s

    folded_names: list[str] = []
    non_folded_names: set[str] = set()

    # Step A: Apply full-precision rescaling + folding / compensation wrappers
    with torch.no_grad():
        for item in eligible:
            w = item.module.weight
            s = scales_by_name[item.full_name].to(device=w.device, dtype=torch.float32)
            inv_s = 1.0 / s
            # Rescale input columns of W: W' = W * diag(s)
            w.copy_((w.float() * s.unsqueeze(0)).to(dtype=w.dtype))

            if item.proj_name == "down_proj" and (item.layer_idx, "up_proj") in by_layer_proj:
                # Fold inv_s into up_proj output rows algebraically!
                up_item = by_layer_proj[(item.layer_idx, "up_proj")]
                up_w = up_item.module.weight
                up_w.copy_((up_w.float() * inv_s.unsqueeze(1)).to(dtype=up_w.dtype))
                if up_item.module.bias is not None:
                    up_item.module.bias.copy_(
                        (up_item.module.bias.float() * inv_s).to(dtype=up_item.module.bias.dtype)
                    )
                folded_names.append(item.full_name)
            else:
                non_folded_names.add(item.full_name)
                wrapper = ScaledLinearWrapper(item.module, inv_s)
                _set_module_by_name(model, item.full_name, wrapper)

    # Step B: Verify full-precision rescaling preserves outputs BEFORE rounding
    fp_equiv_check: dict[str, Any] = {"checked": False}
    if verify_sample_ids is not None and ref_logits is not None:
        with torch.no_grad():
            rescaled_logits = model(input_ids=verify_sample_ids, use_cache=False).logits.detach().float()
        diff = (rescaled_logits - ref_logits).abs()
        fp_equiv_check = {
            "checked": True,
            "max_abs_logit_diff": float(diff.max().item()),
            "mean_abs_logit_diff": float(diff.mean().item()),
            "rel_l2_logit_diff": float(
                diff.pow(2).sum().sqrt().item()
                / max(ref_logits.pow(2).sum().sqrt().item(), 1e-12)
            ),
        }

    # Step C: Apply identical Q1 group-wise asymmetric quantizer to all rescaled weights
    # (note up_proj already includes folded 1/s_down BEFORE quantization!)
    with torch.no_grad():
        for item in eligible:
            b = int(bits_by_name[item.full_name])
            w_hat = quantize_dequantize_weight(
                item.module.weight, bits=b, group_size=group_size
            )
            item.module.weight.copy_(w_hat)

    elapsed_sec = time.perf_counter() - t0
    return {
        "bits_by_name": {k: int(v) for k, v in bits_by_name.items()},
        "folded_projections": folded_names,
        "non_folded_comp_names": non_folded_names,
        "fp_rescaling_equivalence": fp_equiv_check,
        "quant_apply_time_sec": float(elapsed_sec),
    }


def allocate_q3_mixed_precision(
    projections: list[EligibleProjection],
    errors_by_bits: dict[int, dict[str, float]],
    budget_packed_bytes: int,
    uncompressed_fp16_bytes: int,
    non_folded_comp_names: set[str],
    group_size: int = 128,
    min_bits: int = 2,
    allowed_bits: tuple[int, ...] = Q3_ALLOWED_BITS,
) -> dict[str, Any]:
    """Greedy marginal error-reduction-per-byte mixed-precision bit allocation.

    Guarantees total packed storage (including metadata, compensation vectors, and
    uncompressed FP16 parameters) <= budget_packed_bytes.
    """
    valid_bits = sorted(b for b in allowed_bits if b >= min_bits)
    if not valid_bits:
        raise ValueError(f"No valid bit widths >= min_bits={min_bits}.")
    base_b = valid_bits[0]

    # Precompute packed bytes for each (projection, bits)
    mat_bytes: dict[tuple[str, int], int] = {}
    for p in projections:
        has_comp = p.full_name in non_folded_comp_names
        for b in valid_bits:
            mat_bytes[(p.full_name, b)] = compute_matrix_packed_bytes(
                p.out_features, p.in_features, b, group_size, has_comp
            )["total_bytes"]

    bits_by_name = {p.full_name: base_b for p in projections}
    current_bytes = uncompressed_fp16_bytes + sum(
        mat_bytes[(p.full_name, base_b)] for p in projections
    )
    if current_bytes > budget_packed_bytes:
        raise ValueError(
            f"Minimum bit width {base_b} requires {current_bytes} bytes > budget {budget_packed_bytes}."
        )

    # Greedily upgrade to the next bit level with maximum error reduction per additional byte
    while True:
        best_gain_per_byte = -1.0
        best_move: tuple[str, int, int] | None = None
        for p in projections:
            cur_b = bits_by_name[p.full_name]
            cur_idx = valid_bits.index(cur_b)
            if cur_idx + 1 >= len(valid_bits):
                continue
            for nxt_b in valid_bits[cur_idx + 1 :]:
                delta_bytes = mat_bytes[(p.full_name, nxt_b)] - mat_bytes[(p.full_name, cur_b)]
                if delta_bytes <= 0 or current_bytes + delta_bytes > budget_packed_bytes:
                    continue
                err_drop = errors_by_bits[cur_b][p.full_name] - errors_by_bits[nxt_b][p.full_name]
                gain_per_byte = err_drop / float(delta_bytes)
                if gain_per_byte > best_gain_per_byte:
                    best_gain_per_byte = gain_per_byte
                    best_move = (p.full_name, nxt_b, delta_bytes)
        if best_move is None:
            break
        name, nxt_b, delta_bytes = best_move
        bits_by_name[name] = nxt_b
        current_bytes += delta_bytes

    if current_bytes > budget_packed_bytes:
        raise AssertionError(
            f"Q3 allocation exceeded budget: {current_bytes} > {budget_packed_bytes}."
        )

    bit_counts = {str(b): 0 for b in allowed_bits}
    for b in bits_by_name.values():
        bit_counts[str(b)] = bit_counts.get(str(b), 0) + 1

    return {
        "bits_by_name": bits_by_name,
        "budget_packed_bytes": int(budget_packed_bytes),
        "achieved_packed_bytes": int(current_bytes),
        "bytes_under_budget": int(budget_packed_bytes - current_bytes),
        "bit_histogram": bit_counts,
        "min_bits_floor": int(min_bits),
    }
