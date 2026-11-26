"""Core pruning primitives for Task 1: P1 (Magnitude), P2 (Wanda), 2:4, and P3 (Nonuniform)."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn

from common.timing import sync_device
from task1.model_utils import EligibleProjection, enumerate_eligible_projections

P3_MIN_SPARSITY = 0.10
P3_MAX_SPARSITY = 0.85


def snapshot_eligible_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone all eligible projection weights onto CPU for clean restoration between runs."""
    eligible = enumerate_eligible_projections(model)
    return {
        item.full_name: item.module.weight.detach().cpu().clone()
        for item in eligible
    }


def restore_eligible_weights(
    model: nn.Module,
    clean_weights: dict[str, torch.Tensor],
) -> None:
    """Restore unchanged FP16 eligible weights from CPU snapshot."""
    eligible = enumerate_eligible_projections(model)
    if len(eligible) != len(clean_weights):
        raise ValueError(
            f"Snapshot size {len(clean_weights)} != eligible count {len(eligible)}."
        )
    with torch.no_grad():
        for item in eligible:
            src = clean_weights[item.full_name]
            item.module.weight.copy_(src.to(device=item.module.weight.device, dtype=item.module.weight.dtype))


def collect_calibration_input_norms(
    model: nn.Module,
    calib_records: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    """Collect per-input-channel L2 norms ||X_{j,:}||_2 for all eligible projections.

    Uses forward pre-hooks on the unchanged FP16 model over the locked C4 calibration
    sequences, accumulating sum(x^2) in float64 to avoid precision loss.
    """
    model.eval()
    eligible = enumerate_eligible_projections(model)
    sq_sums: dict[str, torch.Tensor] = {
        item.full_name: torch.zeros(item.in_features, dtype=torch.float64, device=device)
        for item in eligible
    }
    handles = []

    def _make_hook(name: str):
        def _hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0].detach()
            x_2d = x.reshape(-1, x.shape[-1]).to(dtype=torch.float64)
            sq_sums[name].add_(x_2d.pow(2).sum(dim=0))

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

    act_norms = {
        name: torch.sqrt(val.clamp_min(0.0)).to(dtype=torch.float32, device="cpu")
        for name, val in sq_sums.items()
    }
    return act_norms, float(elapsed_sec)


def compute_wanda_scores(
    weight: torch.Tensor,
    input_norm: torch.Tensor,
) -> torch.Tensor:
    """Compute Wanda importance score S_ij = |W_ij| * ||X_{j,:}||_2 in float32."""
    w_abs = weight.detach().float().abs()
    norm = input_norm.to(device=w_abs.device, dtype=torch.float32).unsqueeze(0)
    return w_abs * norm


def compute_magnitude_scores(weight: torch.Tensor) -> torch.Tensor:
    """Compute magnitude importance score S_ij = |W_ij| in float32."""
    return weight.detach().float().abs()


def summarize_wanda_distributions(
    model: nn.Module,
    act_norms: dict[str, torch.Tensor],
    outlier_multipliers: tuple[float, ...] = (3.0, 5.0, 7.0),
) -> list[dict[str, Any]]:
    """Compute per-projection Wanda score distribution & OWL outlier statistics."""
    eligible = enumerate_eligible_projections(model)
    summaries: list[dict[str, Any]] = []
    for item in eligible:
        scores = compute_wanda_scores(item.module.weight, act_norms[item.full_name])
        flat = scores.reshape(-1)
        mean_val = float(flat.mean().item())
        std_val = float(flat.std(unbiased=False).item())
        max_val = float(flat.max().item())
        # Subsample deterministically if huge so torch.quantile stays fast & memory-safe
        q_input = flat[:: max(1, flat.numel() // 200_000)]
        q50, q90, q99 = torch.quantile(
            q_input, torch.tensor([0.50, 0.90, 0.99], device=q_input.device)
        ).tolist()

        lod: dict[str, float] = {}
        for m in outlier_multipliers:
            thresh = m * mean_val
            lod[f"lod_m{int(m)}"] = float((flat > thresh).float().mean().item())

        summaries.append(
            {
                "layer_idx": item.layer_idx,
                "full_name": item.full_name,
                "proj_name": item.proj_name,
                "shape": [item.out_features, item.in_features],
                "numel": item.numel,
                "mean": mean_val,
                "std": std_val,
                "max": max_val,
                "q50": float(q50),
                "q90": float(q90),
                "q99": float(q99),
                **lod,
            }
        )
    return summaries


def prune_matrix_per_row(
    weight: torch.Tensor,
    scores: torch.Tensor,
    sparsity: float,
    exact_zero_count: int | None = None,
) -> int:
    """Apply unstructured per-output-row pruning in-place and return zeroed count.

    If exact_zero_count is provided, distributes floor(exact_zero_count / d_out) zeros
    per row plus 1 extra zero to the first (exact_zero_count % d_out) rows so the
    matrix matches exact_zero_count exactly while preserving per-row ranking.
    """
    d_out, d_in = weight.shape
    if exact_zero_count is None:
        k_base = int(round(sparsity * d_in))
        k_base = max(0, min(d_in, k_base))
        extra_rows = 0
    else:
        if exact_zero_count < 0 or exact_zero_count > d_out * d_in:
            raise ValueError(
                f"exact_zero_count={exact_zero_count} out of range [0, {d_out * d_in}]."
            )
        k_base = exact_zero_count // d_out
        extra_rows = exact_zero_count % d_out

    with torch.no_grad():
        if k_base == 0 and extra_rows == 0:
            return 0
        if extra_rows == 0:
            _, idx = torch.topk(scores, k=k_base, dim=1, largest=False, sorted=False)
            weight.scatter_(1, idx, 0.0)
            return d_out * k_base

        # First extra_rows rows get k_base + 1 zeros; remaining rows get k_base zeros.
        _, idx_hi = torch.topk(
            scores[:extra_rows], k=k_base + 1, dim=1, largest=False, sorted=False
        )
        weight[:extra_rows].scatter_(1, idx_hi, 0.0)
        if k_base > 0 and extra_rows < d_out:
            _, idx_lo = torch.topk(
                scores[extra_rows:], k=k_base, dim=1, largest=False, sorted=False
            )
            weight[extra_rows:].scatter_(1, idx_lo, 0.0)
        return extra_rows * (k_base + 1) + (d_out - extra_rows) * k_base


def prune_matrix_2to4(weight: torch.Tensor, scores: torch.Tensor) -> int:
    """Apply 2:4 semi-structured pruning in-place along contiguous groups of 4 input channels."""
    d_out, d_in = weight.shape
    if d_in % 4 != 0:
        raise ValueError(f"2:4 pruning requires in_features divisible by 4, got {d_in}.")
    with torch.no_grad():
        grouped_scores = scores.reshape(d_out, d_in // 4, 4)
        _, drop_idx = torch.topk(grouped_scores, k=2, dim=-1, largest=False, sorted=False)
        w_grouped = weight.reshape(d_out, d_in // 4, 4)
        w_grouped.scatter_(-1, drop_idx, 0.0)
    return (d_out * d_in) // 2


def apply_uniform_pruning(
    model: nn.Module,
    method: str,
    sparsity: float,
    act_norms: dict[str, torch.Tensor] | None = None,
    structured_2to4: bool = False,
) -> dict[str, Any]:
    """Apply P1 (magnitude), P2 (Wanda), or 2:4 Wanda pruning to all eligible projections."""
    eligible = enumerate_eligible_projections(model)
    t0 = time.perf_counter()
    total_eligible = 0
    total_zeroed = 0
    per_proj: list[dict[str, Any]] = []

    for item in eligible:
        w = item.module.weight
        if method == "p1_magnitude":
            scores = compute_magnitude_scores(w)
        elif method == "p2_wanda":
            if act_norms is None:
                raise ValueError("act_norms is required for p2_wanda pruning.")
            scores = compute_wanda_scores(w, act_norms[item.full_name])
        else:
            raise ValueError(f"Unsupported pruning method '{method}'.")

        if structured_2to4:
            if abs(sparsity - 0.50) > 1e-9:
                raise ValueError("2:4 structured pruning is only defined at 50% sparsity.")
            zeroed = prune_matrix_2to4(w, scores)
        else:
            zeroed = prune_matrix_per_row(w, scores, sparsity=sparsity)

        total_eligible += item.numel
        total_zeroed += zeroed
        per_proj.append(
            {
                "layer_idx": item.layer_idx,
                "full_name": item.full_name,
                "proj_name": item.proj_name,
                "numel": item.numel,
                "zeroed": zeroed,
                "achieved_sparsity": zeroed / item.numel,
            }
        )

    elapsed_sec = time.perf_counter() - t0
    return {
        "method": method,
        "structured_2to4": structured_2to4,
        "target_sparsity": float(sparsity),
        "total_eligible_weights": int(total_eligible),
        "total_zeroed_weights": int(total_zeroed),
        "achieved_sparsity": float(total_zeroed / total_eligible),
        "pruning_apply_time_sec": float(elapsed_sec),
        "per_projection": per_proj,
    }


def measure_calibration_layer_output_error(
    model_fp16_or_pruned: nn.Module,
    clean_weights: dict[str, torch.Tensor],
    calib_records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Measure local projection output error ||(W_pruned - W_fp16) X||_F / ||W_fp16 X||_F.

    Evaluated on the exact calibration inputs seen by the unchanged FP16 model:
    temporarily restores clean FP16 weights for the forward pass and computes exact
    local output distortion Delta_Y = (W_pruned - W_clean) @ X^T inside forward pre-hooks.
    """
    eligible = enumerate_eligible_projections(model_fp16_or_pruned)
    pruned_weights = {
        item.full_name: item.module.weight.detach().clone() for item in eligible
    }
    delta_weights = {
        item.full_name: (
            pruned_weights[item.full_name]
            - clean_weights[item.full_name].to(
                device=pruned_weights[item.full_name].device,
                dtype=pruned_weights[item.full_name].dtype,
            )
        )
        for item in eligible
    }

    # Restore clean FP16 weights so every projection sees the exact FP16 calibration inputs X
    restore_eligible_weights(model_fp16_or_pruned, clean_weights)
    model_fp16_or_pruned.eval()

    err_sq: dict[str, float] = {item.full_name: 0.0 for item in eligible}
    ref_sq: dict[str, float] = {item.full_name: 0.0 for item in eligible}
    n_elems: dict[str, int] = {item.full_name: 0 for item in eligible}
    handles = []

    def _make_hook(item: EligibleProjection):
        name = item.full_name
        dw = delta_weights[name]

        def _hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            x = inputs[0].detach()
            y_ref = output.detach()
            # Local linear output error: (W_pruned - W_fp16) @ x
            dy = torch.nn.functional.linear(x, dw)
            err_sq[name] += float(dy.float().pow(2).sum().item())
            ref_sq[name] += float(y_ref.float().pow(2).sum().item())
            n_elems[name] += int(y_ref.numel())

        return _hook

    for item in eligible:
        handles.append(item.module.register_forward_hook(_make_hook(item)))

    sync_device(device)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            for rec in calib_records:
                ids = torch.tensor([rec["token_ids"]], dtype=torch.long, device=device)
                model_fp16_or_pruned(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
        # Restore pruned weights back onto the model
        with torch.no_grad():
            for item in eligible:
                item.module.weight.copy_(pruned_weights[item.full_name])
    sync_device(device)
    elapsed_sec = time.perf_counter() - t0

    per_proj: list[dict[str, Any]] = []
    rel_errors: list[float] = []
    total_err_sq = 0.0
    total_ref_sq = 0.0
    for item in eligible:
        name = item.full_name
        e2 = err_sq[name]
        r2 = ref_sq[name]
        rel = math.sqrt(e2 / max(r2, 1e-12))
        mse = e2 / max(n_elems[name], 1)
        rel_errors.append(rel)
        total_err_sq += e2
        total_ref_sq += r2
        per_proj.append(
            {
                "layer_idx": item.layer_idx,
                "full_name": name,
                "proj_name": item.proj_name,
                "rel_l2_error": float(rel),
                "mse": float(mse),
            }
        )

    return {
        "eval_time_sec": float(elapsed_sec),
        "mean_projection_rel_l2_error": float(sum(rel_errors) / max(len(rel_errors), 1)),
        "max_projection_rel_l2_error": float(max(rel_errors)) if rel_errors else 0.0,
        "pooled_rel_l2_error": float(math.sqrt(total_err_sq / max(total_ref_sq, 1e-12))),
        "per_projection": per_proj,
    }


def allocate_nonuniform_sparsities(
    projections: list[EligibleProjection],
    sensitivities: dict[str, float],
    target_sparsity: float,
    lam: float,
    s_min: float = P3_MIN_SPARSITY,
    s_max: float = P3_MAX_SPARSITY,
) -> dict[str, Any]:
    """Convert per-projection sensitivities into an exact matched-budget sparsity map.

    Higher sensitivity -> lower sparsity (more weights retained).
    Constraints (ATML PA4 PDF p. 4):
    - Every unit satisfies s_min (0.10) <= s_l <= s_max (0.85).
    - Total zeroed eligible weights match uniform pruning at target_sparsity exactly
      up to integer-weight rounding (here matched to the exact integer target).
    """
    if not (s_min <= target_sparsity <= s_max):
        raise ValueError(
            f"target_sparsity={target_sparsity} must lie within [{s_min}, {s_max}]."
        )

    raw_vals = [float(sensitivities[p.full_name]) for p in projections]
    mean_v = sum(raw_vals) / len(raw_vals)
    var_v = sum((v - mean_v) ** 2 for v in raw_vals) / len(raw_vals)
    std_v = math.sqrt(var_v)
    if std_v < 1e-12:
        norm_vals = [0.0 for _ in raw_vals]
    else:
        # Normalize to roughly [-1, 1] via clipped z-score / max-abs so lam directly
        # controls the half-spread around target_sparsity.
        z = [(v - mean_v) / std_v for v in raw_vals]
        max_abs = max(abs(x) for x in z)
        norm_vals = [x / max_abs for x in z]

    # Target total zeroed weights matches uniform per-row rounding at target_sparsity
    target_total_zeros = sum(
        p.out_features * int(round(target_sparsity * p.in_features))
        for p in projections
    )
    total_eligible = sum(p.numel for p in projections)

    base_rates = [target_sparsity - lam * u for u in norm_vals]

    # Bisection for scalar offset delta so sum_l N_l * clip(base_l + delta, s_min, s_max) == target_total_zeros
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        est_zeros = sum(
            p.numel * min(s_max, max(s_min, r + mid))
            for p, r in zip(projections, base_rates)
        )
        if est_zeros < target_total_zeros:
            lo = mid
        else:
            hi = mid
    delta = 0.5 * (lo + hi)

    continuous_rates = [min(s_max, max(s_min, r + delta)) for r in base_rates]

    # Integer zero counts per projection respecting [ceil(s_min * N_l), floor(s_max * N_l)]
    min_zeros = [math.ceil(s_min * p.numel) for p in projections]
    max_zeros = [math.floor(s_max * p.numel) for p in projections]
    zero_counts = [
        min(hi_z, max(lo_z, int(round(rate * p.numel))))
        for p, rate, lo_z, hi_z in zip(projections, continuous_rates, min_zeros, max_zeros)
    ]

    # Exact integer-weight residual adjustment (distribute remaining +/- weights to
    # projections with largest fractional rounding slack while staying inside [s_min, s_max])
    diff = target_total_zeros - sum(zero_counts)
    if diff != 0:
        step = 1 if diff > 0 else -1
        remaining = abs(diff)
        # Prioritize projections ordered by how much room they have
        order = sorted(
            range(len(projections)),
            key=lambda i: (
                (continuous_rates[i] * projections[i].numel - zero_counts[i]) * step
            ),
            reverse=True,
        )
        idx_ptr = 0
        while remaining > 0 and idx_ptr < len(order):
            i = order[idx_ptr]
            room = (max_zeros[i] - zero_counts[i]) if step > 0 else (zero_counts[i] - min_zeros[i])
            if room > 0:
                take = min(remaining, room)
                zero_counts[i] += step * take
                remaining -= take
            idx_ptr += 1

    achieved_total_zeros = sum(zero_counts)
    per_unit: list[dict[str, Any]] = []
    allocation_map: dict[str, int] = {}
    for p, sens, zc in zip(projections, raw_vals, zero_counts):
        s_l = zc / p.numel
        if s_l < s_min - 1e-9 or s_l > s_max + 1e-9:
            raise AssertionError(
                f"Unit {p.full_name} sparsity {s_l:.6f} violated [{s_min}, {s_max}]."
            )
        allocation_map[p.full_name] = int(zc)
        per_unit.append(
            {
                "layer_idx": p.layer_idx,
                "full_name": p.full_name,
                "proj_name": p.proj_name,
                "numel": p.numel,
                "sensitivity": float(sens),
                "zeroed_weights": int(zc),
                "sparsity": float(s_l),
            }
        )

    return {
        "unit_definition": "individual_eligible_projection",
        "target_sparsity": float(target_sparsity),
        "lambda": float(lam),
        "s_min": float(s_min),
        "s_max": float(s_max),
        "total_eligible_weights": int(total_eligible),
        "target_total_zeros": int(target_total_zeros),
        "achieved_total_zeros": int(achieved_total_zeros),
        "zero_count_diff": int(achieved_total_zeros - target_total_zeros),
        "achieved_global_sparsity": float(achieved_total_zeros / total_eligible),
        "min_unit_sparsity": float(min(u["sparsity"] for u in per_unit)),
        "max_unit_sparsity": float(max(u["sparsity"] for u in per_unit)),
        "zero_counts_by_name": allocation_map,
        "per_unit": per_unit,
    }


def apply_nonuniform_wanda_pruning(
    model: nn.Module,
    act_norms: dict[str, torch.Tensor],
    zero_counts_by_name: dict[str, int],
) -> dict[str, Any]:
    """Apply Wanda per-row pruning using exact per-projection zero counts."""
    eligible = enumerate_eligible_projections(model)
    t0 = time.perf_counter()
    total_eligible = 0
    total_zeroed = 0
    for item in eligible:
        scores = compute_wanda_scores(item.module.weight, act_norms[item.full_name])
        target_z = int(zero_counts_by_name[item.full_name])
        zeroed = prune_matrix_per_row(
            item.module.weight,
            scores,
            sparsity=target_z / item.numel,
            exact_zero_count=target_z,
        )
        total_eligible += item.numel
        total_zeroed += zeroed
    elapsed_sec = time.perf_counter() - t0
    return {
        "total_eligible_weights": int(total_eligible),
        "total_zeroed_weights": int(total_zeroed),
        "achieved_sparsity": float(total_zeroed / total_eligible),
        "pruning_apply_time_sec": float(elapsed_sec),
    }
