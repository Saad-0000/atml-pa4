"""Exact speculative sampling (Leviathan et al., 2023) with one-pending-token KV invariant."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from common.timing import sync_device


def crop_dynamic_cache(cache: DynamicCache, target_len: int) -> None:
    """Truncate every layer of a transformers==4.57.1 DynamicCache to target_len."""
    if target_len < 0:
        raise ValueError(f"target_len must be >= 0, got {target_len}")
    if hasattr(cache, "crop"):
        cache.crop(target_len)
        return
    for layer in cache.layers:
        if layer.keys is not None and layer.keys.shape[2] > target_len:
            layer.keys = layer.keys[:, :, :target_len, :].contiguous()
            layer.values = layer.values[:, :, :target_len, :].contiguous()


def _sample_from_probs(probs: torch.Tensor, generator: torch.Generator) -> int:
    """Sample a single token ID from a 1D FP32 probability vector."""
    probs = probs.clamp_min(0.0)
    s = probs.sum()
    if s <= 0 or not torch.isfinite(s):
        probs = torch.full_like(probs, 1.0 / probs.numel())
    else:
        probs = probs / s
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def _sample_positive_residual(
    p: torch.Tensor,
    q: torch.Tensor,
    generator: torch.Generator,
) -> int:
    """Sample from normalized positive residual r(y) = [p(y) - q(y)]_+ / Z."""
    diff = torch.clamp(p - q, min=0.0)
    z = diff.sum()
    if float(z.item()) <= 1e-12 or not torch.isfinite(z):
        return _sample_from_probs(p, generator)
    return _sample_from_probs(diff / z, generator)


def target_only_generate(
    target_model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_ids: set[int],
    device: torch.device,
    seed: int = 6304,
    temperature: float = 1.0,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    """Ordinary autoregressive target sampling with separated prefill and decode timers."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    sync_device(device)
    t_pre0 = time.perf_counter()
    with torch.no_grad():
        out = target_model(input_ids=inp, use_cache=True)
    sync_device(device)
    prefill_sec = time.perf_counter() - t_pre0

    cache = out.past_key_values
    logits = out.logits[0, -1, :].float()
    generated: list[int] = []
    target_passes = 1  # prefill
    hit_eos = False

    sync_device(device)
    t_dec0 = time.perf_counter()
    with torch.no_grad():
        for step in range(max_new_tokens):
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            tok = _sample_from_probs(probs, gen)
            generated.append(tok)
            if (not ignore_eos) and (tok in eos_ids):
                hit_eos = True
                break
            if step + 1 < max_new_tokens:
                next_inp = torch.tensor([[tok]], dtype=torch.long, device=device)
                out = target_model(input_ids=next_inp, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = out.logits[0, -1, :].float()
                target_passes += 1
    sync_device(device)
    decode_sec = time.perf_counter() - t_dec0

    return {
        "generated_ids": generated,
        "num_tokens": len(generated),
        "hit_eos": hit_eos,
        "truncated": (not hit_eos) and (len(generated) == max_new_tokens),
        "target_passes": target_passes,
        "prefill_sec": float(prefill_sec),
        "decode_sec": float(decode_sec),
        "total_sec": float(prefill_sec + decode_sec),
    }


def exact_speculative_generate(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    gamma: int,
    eos_ids: set[int],
    device: torch.device,
    seed: int = 6304,
    temperature: float = 1.0,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    """Exact speculative decoding enforcing the one-pending-token KV-cache invariant.

    Invariant before every speculative round:
    - Let C be the full emitted sequence so far (prompt + generated), of length L >= 1.
    - Both draft_cache and target_cache hold exactly C[:-1] (length L - 1).
    - pending_token = C[-1] has NOT yet been fed into either cache.
    - Round proposes k = min(gamma, remaining) tokens starting from pending_token,
      verifies [pending_token, x_1, ..., x_k] in one target pass (k+1 tokens),
      emits n_emitted in [1, k+1] tokens, sets pending_token = last emitted token,
      and crops both caches to length (L - 1) + n_emitted = new_L - 1.
    """
    if len(prompt_ids) < 1:
        raise ValueError("prompt_ids must be non-empty.")
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1, got {gamma}")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Prefill C[:-1] into both caches; keep C[-1] as pending_token
    sync_device(device)
    t_pre0 = time.perf_counter()
    with torch.no_grad():
        if len(prompt_ids) > 1:
            prefix_t = torch.tensor([prompt_ids[:-1]], dtype=torch.long, device=device)
            t_out = target_model(input_ids=prefix_t, use_cache=True)
            d_out = draft_model(input_ids=prefix_t, use_cache=True)
            target_cache = t_out.past_key_values
            draft_cache = d_out.past_key_values
        else:
            target_cache = DynamicCache()
            draft_cache = DynamicCache()
    sync_device(device)
    prefill_sec = time.perf_counter() - t_pre0

    pending_token = int(prompt_ids[-1])
    cached_len = len(prompt_ids) - 1  # invariant: both caches have exact length cached_len

    generated: list[int] = []
    hit_eos = False
    rounds = 0
    total_proposed = 0
    total_examined = 0
    total_accepted = 0
    draft_forward_passes = 0
    target_verify_passes = 0

    # Four-part per-round synchronized timing breakdown
    t_draft_sec = 0.0
    t_verify_sec = 0.0
    t_accept_sec = 0.0
    t_cache_sec = 0.0

    sync_device(device)
    t_dec0 = time.perf_counter()

    with torch.no_grad():
        while len(generated) < max_new_tokens and not hit_eos:
            remaining = max_new_tokens - len(generated)
            k = min(gamma, remaining)
            rounds += 1

            # --- 1. Draft proposal passes ---
            sync_device(device)
            t0 = time.perf_counter()
            proposals: list[int] = []
            draft_probs_list: list[torch.Tensor] = []
            cur_tok = pending_token
            for _ in range(k):
                d_inp = torch.tensor([[cur_tok]], dtype=torch.long, device=device)
                d_out = draft_model(input_ids=d_inp, past_key_values=draft_cache, use_cache=True)
                draft_cache = d_out.past_key_values
                draft_forward_passes += 1
                q_probs = F.softmax(d_out.logits[0, -1, :].float() / max(temperature, 1e-6), dim=-1)
                x_j = _sample_from_probs(q_probs, gen)
                proposals.append(x_j)
                draft_probs_list.append(q_probs)
                cur_tok = x_j
            sync_device(device)
            t_draft_sec += time.perf_counter() - t0
            total_proposed += k

            # --- 2. Single target verification pass on [pending_token, x_1, ..., x_k] ---
            sync_device(device)
            t0 = time.perf_counter()
            verify_seq = [pending_token] + proposals
            t_inp = torch.tensor([verify_seq], dtype=torch.long, device=device)
            t_out = target_model(input_ids=t_inp, past_key_values=target_cache, use_cache=True)
            target_cache = t_out.past_key_values
            target_verify_passes += 1
            t_logits = t_out.logits[0].float()  # [k + 1, V]
            sync_device(device)
            t_verify_sec += time.perf_counter() - t0

            # --- 3. Exact accept/reject, positive-residual, & bonus token ---
            sync_device(device)
            t0 = time.perf_counter()
            round_emitted: list[int] = []
            accepted_in_round = 0
            rejected = False

            for j in range(k):
                total_examined += 1
                x_j = proposals[j]
                p_j = F.softmax(t_logits[j] / max(temperature, 1e-6), dim=-1)
                q_j = draft_probs_list[j]
                q_x = float(q_j[x_j].item())
                p_x = float(p_j[x_j].item())
                acc_prob = 1.0 if q_x <= 0.0 else min(1.0, p_x / q_x)
                u = float(torch.rand((), generator=gen, device=device).item())

                if u < acc_prob:
                    accepted_in_round += 1
                    total_accepted += 1
                    round_emitted.append(x_j)
                    if (not ignore_eos) and (x_j in eos_ids):
                        hit_eos = True
                        break
                else:
                    y = _sample_positive_residual(p_j, q_j, gen)
                    round_emitted.append(y)
                    if (not ignore_eos) and (y in eos_ids):
                        hit_eos = True
                    rejected = True
                    break

            if (not rejected) and (not hit_eos) and (len(generated) + len(round_emitted) < max_new_tokens):
                # All k proposals accepted -> sample bonus token from p_{k+1}
                p_bonus = F.softmax(t_logits[k] / max(temperature, 1e-6), dim=-1)
                bonus_tok = _sample_from_probs(p_bonus, gen)
                round_emitted.append(bonus_tok)
                if (not ignore_eos) and (bonus_tok in eos_ids):
                    hit_eos = True

            generated.extend(round_emitted)
            sync_device(device)
            t_accept_sec += time.perf_counter() - t0

            # --- 4. Cache rollback / advancement preserving one-pending-token invariant ---
            sync_device(device)
            t0 = time.perf_counter()
            n_emitted = len(round_emitted)
            new_cached_len = cached_len + n_emitted
            # Target cache currently has cached_len + k + 1 entries; crop to new_cached_len
            crop_dynamic_cache(target_cache, new_cached_len)
            # Draft cache currently has cached_len + k entries; if n_emitted <= k crop to new_cached_len.
            # If n_emitted == k + 1 (full accept + bonus), draft cache has cached_len + k == new_cached_len - 1,
            # so we feed x_k into draft_cache (1 token) or wait: draft already processed pending_token, x_1..x_{k-1}
            # (k passes) -> draft_cache has cached_len + k entries! The last emitted token is bonus_tok,
            # and x_k was NOT fed into draft_cache yet! So feed x_k once so draft_cache holds [..., x_k] (len new_cached_len).
            if draft_cache.get_seq_length() > new_cached_len:
                crop_dynamic_cache(draft_cache, new_cached_len)
            elif draft_cache.get_seq_length() < new_cached_len:
                # Full acceptance + bonus: advance draft_cache by x_k (1 token) to reach new_cached_len
                adv_tok = torch.tensor([[proposals[-1]]], dtype=torch.long, device=device)
                d_adv = draft_model(input_ids=adv_tok, past_key_values=draft_cache, use_cache=True)
                draft_cache = d_adv.past_key_values
                draft_forward_passes += 1

            pending_token = round_emitted[-1]
            cached_len = new_cached_len
            sync_device(device)
            t_cache_sec += time.perf_counter() - t0

    sync_device(device)
    decode_sec = time.perf_counter() - t_dec0
    host_overhead_sec = max(0.0, decode_sec - (t_draft_sec + t_verify_sec + t_accept_sec + t_cache_sec))

    return {
        "generated_ids": generated,
        "num_tokens": len(generated),
        "hit_eos": hit_eos,
        "truncated": (not hit_eos) and (len(generated) == max_new_tokens),
        "gamma": int(gamma),
        "rounds": rounds,
        "total_proposed": total_proposed,
        "total_examined": total_examined,
        "total_accepted": total_accepted,
        "accepted_over_examined": float(total_accepted / max(total_examined, 1)),
        "accepted_over_proposed": float(total_accepted / max(total_proposed, 1)),
        "mean_accepted_per_verify": float(total_accepted / max(target_verify_passes, 1)),
        "target_verify_passes": target_verify_passes,
        "target_passes_per_output_token": float(target_verify_passes / max(len(generated), 1)),
        "draft_forward_passes": draft_forward_passes,
        "prefill_sec": float(prefill_sec),
        "decode_sec": float(decode_sec),
        "total_sec": float(prefill_sec + decode_sec),
        "round_timing_sec": {
            "draft_proposals_sec": float(t_draft_sec),
            "target_verification_sec": float(t_verify_sec),
            "accept_residual_sec": float(t_accept_sec),
            "cache_rollback_advance_sec": float(t_cache_sec),
            "host_overhead_sec": float(host_overhead_sec),
        },
        "final_target_cache_len": int(target_cache.get_seq_length()),
        "final_draft_cache_len": int(draft_cache.get_seq_length()),
    }


def run_markov_chi2_validation(
    n_trials: int = 100_000,
    seq_len: int = 3,
    gamma: int = 2,
    seed: int = 6304,
) -> dict[str, Any]:
    """100,000-trial 3-symbol Markov joint-sequence Pearson chi^2 validation & negative control.

    - Exact sampler (positive residual) must NOT be rejected at p < 0.01.
    - Negative control (replacing positive residual with p) MUST be rejected at p < 0.01.
    - All 27 joint 3-token sequence expected counts are >= 10.
    """
    from scipy.stats import chisquare

    # 3-symbol Markov transitions (states 0, 1, 2; initial state = 0)
    # Chosen so target and draft differ clearly while every 3-token sequence has prob >= 0.005 (>= 500 expected in 100k)
    p_init = np.array([0.40, 0.35, 0.25], dtype=np.float64)
    p_trans = np.array(
        [
            [0.50, 0.30, 0.20],
            [0.25, 0.45, 0.30],
            [0.30, 0.25, 0.45],
        ],
        dtype=np.float64,
    )
    q_init = np.array([0.20, 0.55, 0.25], dtype=np.float64)
    q_trans = np.array(
        [
            [0.20, 0.55, 0.25],
            [0.50, 0.20, 0.30],
            [0.55, 0.25, 0.20],
        ],
        dtype=np.float64,
    )

    def _p_dist(prefix: list[int]) -> np.ndarray:
        return p_init if not prefix else p_trans[prefix[-1]]

    def _q_dist(prefix: list[int]) -> np.ndarray:
        return q_init if not prefix else q_trans[prefix[-1]]

    # Exact joint probability over all 3^3 = 27 sequences
    exact_probs = np.zeros(27, dtype=np.float64)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                idx = a * 9 + b * 3 + c
                exact_probs[idx] = p_init[a] * p_trans[a, b] * p_trans[b, c]
    expected_counts = exact_probs * n_trials
    min_expected = float(expected_counts.min())
    if min_expected < 10.0:
        raise AssertionError(f"Expected count {min_expected} < 10")

    def _simulate(use_buggy_residual: bool, sim_seed: int) -> np.ndarray:
        rng = np.random.default_rng(sim_seed)
        counts = np.zeros(27, dtype=np.int64)
        for _ in range(n_trials):
            seq: list[int] = []
            while len(seq) < seq_len:
                k = min(gamma, seq_len - len(seq))
                props: list[int] = []
                q_list: list[np.ndarray] = []
                tmp_prefix = list(seq)
                for _j in range(k):
                    qj = _q_dist(tmp_prefix)
                    xj = int(rng.choice(3, p=qj))
                    props.append(xj)
                    q_list.append(qj)
                    tmp_prefix.append(xj)

                rejected = False
                for j in range(k):
                    pj = _p_dist(seq)
                    qj = q_list[j]
                    xj = props[j]
                    acc = min(1.0, pj[xj] / qj[xj])
                    if rng.random() < acc:
                        seq.append(xj)
                    else:
                        if use_buggy_residual:
                            res_dist = pj
                        else:
                            pos = np.maximum(pj - qj, 0.0)
                            res_dist = pos / pos.sum()
                        seq.append(int(rng.choice(3, p=res_dist)))
                        rejected = True
                        break
                if (not rejected) and len(seq) < seq_len:
                    p_bonus = _p_dist(seq)
                    seq.append(int(rng.choice(3, p=p_bonus)))
            counts[seq[0] * 9 + seq[1] * 3 + seq[2]] += 1
        return counts

    exact_counts = _simulate(use_buggy_residual=False, sim_seed=seed)
    buggy_counts = _simulate(use_buggy_residual=True, sim_seed=seed + 1)

    chi2_exact, p_exact = chisquare(exact_counts, f_exp=expected_counts)
    chi2_buggy, p_buggy = chisquare(buggy_counts, f_exp=expected_counts)

    return {
        "n_trials": int(n_trials),
        "seq_len": int(seq_len),
        "gamma": int(gamma),
        "min_expected_count": min_expected,
        "exact_sampler": {
            "chi2_stat": float(chi2_exact),
            "p_value": float(p_exact),
            "passed_not_rejected_at_0p01": bool(p_exact >= 0.01),
            "max_abs_joint_prob_err": float(np.abs(exact_counts / n_trials - exact_probs).max()),
        },
        "negative_control_p_instead_of_residual": {
            "chi2_stat": float(chi2_buggy),
            "p_value": float(p_buggy),
            "passed_rejected_at_0p01": bool(p_buggy < 0.01),
            "max_abs_joint_prob_err": float(np.abs(buggy_counts / n_trials - exact_probs).max()),
        },
        "passed": bool(p_exact >= 0.01 and p_buggy < 0.01),
    }


def analytical_speedup(alpha: float, c: float, v_gamma: float, gamma: int) -> float:
    """S_hat_gamma = (1 + sum_{j=1}^gamma alpha^j) / (gamma * c + v_gamma)."""
    num = 1.0 + sum(alpha ** j for j in range(1, gamma + 1))
    den = gamma * c + v_gamma
    return float(num / max(den, 1e-12))


def compute_breakeven_alpha(c: float, v_gamma: float, gamma: int) -> float | None:
    """Smallest alpha in [0, 1] with 1 + sum_{j=1}^gamma alpha^j > gamma*c + v_gamma, or None."""
    rhs = gamma * c + v_gamma
    if 1.0 > rhs:
        return 0.0
    if (1.0 + gamma) <= rhs:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lhs = 1.0 + sum(mid ** j for j in range(1, gamma + 1))
        if lhs > rhs:
            hi = mid
        else:
            lo = mid
    return float(hi)
