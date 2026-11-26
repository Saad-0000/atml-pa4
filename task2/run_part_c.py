"""Task 2 Part C: Pretrained KV Cache Validation (C1), Compression (C2-C4), & Benchmarks (C5).

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task2.run_part_c --stage c1 --data-dir <DATA_DIR> --out-dir <T2_OUT> --seed 6304
  python -m task2.run_part_c --stage eval_all --resume --data-dir <DATA_DIR> --out-dir <T2_OUT> --seed 6304
  python -m task2.run_part_c --stage bench_packed --batches 1 8 --lengths 2048 4096 --data-dir <DATA_DIR> --out-dir <T2_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from common.env import collect_env_metadata, resolve_model_revision
from common.io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    ensure_task_dirs,
    load_local_source_revisions,
    read_json,
    read_jsonl,
    resolve_smoke_out_dir,
    update_condition_status,
)
from common.seeding import DEFAULT_SEED, set_seed
from common.timing import get_memory_stats, measure_latency, reset_peak_memory, sync_device
from task2.kv_cache import (
    DEFAULT_INSTRUCT_MODEL_ID,
    GROUP_SIZE,
    cache_from_kv_list,
    cache_get_kv,
    cache_num_layers,
    clone_cache,
    compute_question_aware_importance,
    extract_five_digit_code,
    kv_full_fp16_bytes,
    kv_quant_reference_bytes,
    max_eviction_positions,
    pack_kv_tensor,
    select_eviction_indices,
    solve_hybrid_allocation,
    split_prompt_and_map_evidence,
    unpack_kv_tensor,
)

POLICIES = (
    "fp16_full",
    "evict_sink_recent",
    "evict_question_aware",
    "quant_int4_tokenwise",
    "quant_int4_channelK_tokenV",
    "hybrid_fp16_int2",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 2 Part C: Pretrained KV cache validation, compression, and benchmarks."
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=("c1", "eval_all", "bench_packed"),
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--model-id", type=str, default=DEFAULT_INSTRUCT_MODEL_ID)
    parser.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--device", type=str, default=None)
    return parser


def _load_kv_cases(data_dir: str | Path, limit: int | None = None) -> tuple[Path, list[dict[str, Any]]]:
    root = Path(data_dir)
    for c in (
        root / "kv_retrieval" / "kv_retrieval_cases.jsonl.gz",
        root / "data" / "kv_retrieval" / "kv_retrieval_cases.jsonl.gz",
    ):
        if c.exists():
            return c, read_jsonl(c, limit=limit)
    raise FileNotFoundError(f"kv_retrieval_cases.jsonl.gz not found under {data_dir}")


def _load_model(model_id: str, dtype_name: str, device_str: str | None):
    dev = (
        torch.device(device_str)
        if device_str is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dt = torch.float16 if (dtype_name == "fp16" and dev.type == "cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt)
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, dev, dt


def _quantize_cache_layout(
    cache: DynamicCache,
    bits: int,
    k_axis: str,
    v_axis: str,
    group_size: int = GROUP_SIZE,
) -> tuple[DynamicCache, dict[str, Any]]:
    """Pack and immediately unpack every layer's K/V while recording exact serialized bytes & errors."""
    n_layers = cache_num_layers(cache)
    new_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    total_packed_bytes = 0
    k_err_sq = k_ref_sq = 0.0
    v_err_sq = v_ref_sq = 0.0

    for i in range(n_layers):
        k, v = cache_get_kv(cache, i)
        pk = pack_kv_tensor(k, bits=bits, axis=k_axis, group_size=group_size)
        pv = pack_kv_tensor(v, bits=bits, axis=v_axis, group_size=group_size)
        total_packed_bytes += pk.serialized_bytes() + pv.serialized_bytes()
        k_hat = unpack_kv_tensor(pk, out_dtype=k.dtype)
        v_hat = unpack_kv_tensor(pv, out_dtype=v.dtype)
        k_err_sq += float((k_hat.float() - k.float()).pow(2).sum().item())
        k_ref_sq += float(k.float().pow(2).sum().item())
        v_err_sq += float((v_hat.float() - v.float()).pow(2).sum().item())
        v_ref_sq += float(v.float().pow(2).sum().item())
        new_kv.append((k_hat, v_hat))

    return cache_from_kv_list(new_kv), {
        "serialized_bytes": int(total_packed_bytes),
        "k_rel_l2_err": float(math.sqrt(k_err_sq / max(k_ref_sq, 1e-12))),
        "v_rel_l2_err": float(math.sqrt(v_err_sq / max(v_ref_sq, 1e-12))),
    }


def _apply_policy_to_pre_cache(
    policy: str,
    pre_cache: DynamicCache,
    pre_len: int,
    evidence_indices: list[int],
    importance_per_layer: list[torch.Tensor] | None,
    budget_bytes: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[DynamicCache, dict[str, Any]]:
    """Apply a Part C compression policy to pre_cache under the exact matched INT4 budget."""
    ev_set = set(evidence_indices)
    full_bytes = kv_full_fp16_bytes(pre_len, n_layers, n_kv_heads, head_dim)

    if policy == "fp16_full":
        return clone_cache(pre_cache), {
            "pre_compressed_bytes": full_bytes,
            "realized_frac_of_fp16": 1.0,
            "evidence_retained_selective_fp16": 1.0 if ev_set else 1.0,
            "evidence_retained_any_fp16": 1.0 if ev_set else 1.0,
            "k_rel_l2_err": 0.0,
            "v_rel_l2_err": 0.0,
            "stream_quant": None,
        }

    if policy in ("evict_sink_recent", "evict_question_aware"):
        alloc = max_eviction_positions(pre_len, budget_bytes, n_layers, n_kv_heads, head_dim)
        k_kept = alloc["k_kept"]
        new_kv = []
        retained_fracs = []
        for i in range(n_layers):
            k, v = cache_get_kv(pre_cache, i)
            imp = importance_per_layer[i] if (policy == "evict_question_aware" and importance_per_layer is not None) else None
            idx = select_eviction_indices(pre_len, k_kept, importance=imp).to(k.device)
            new_kv.append((k.index_select(2, idx).clone(), v.index_select(2, idx).clone()))
            if ev_set:
                kept_set = set(idx.tolist())
                retained_fracs.append(len(ev_set & kept_set) / len(ev_set))
        ret_mean = float(sum(retained_fracs) / len(retained_fracs)) if retained_fracs else 1.0
        return cache_from_kv_list(new_kv), {
            "pre_compressed_bytes": int(alloc["total_bytes"]),
            "realized_frac_of_fp16": float(alloc["total_bytes"] / max(full_bytes, 1)),
            "k_kept": int(k_kept),
            "evidence_retained_selective_fp16": ret_mean,
            "evidence_retained_any_fp16": ret_mean,
            "k_rel_l2_err": 0.0,
            "v_rel_l2_err": 0.0,
            "stream_quant": None,
        }

    if policy == "quant_int4_tokenwise":
        q_cache, q_info = _quantize_cache_layout(pre_cache, bits=4, k_axis="token", v_axis="token")
        return q_cache, {
            "pre_compressed_bytes": q_info["serialized_bytes"],
            "realized_frac_of_fp16": float(q_info["serialized_bytes"] / max(full_bytes, 1)),
            "evidence_retained_selective_fp16": 0.0,
            "evidence_retained_any_fp16": 0.0,
            "k_rel_l2_err": q_info["k_rel_l2_err"],
            "v_rel_l2_err": q_info["v_rel_l2_err"],
            "stream_quant": {"bits": 4, "k_axis": "token", "v_axis": "token"},
        }

    if policy == "quant_int4_channelK_tokenV":
        q_cache, q_info = _quantize_cache_layout(pre_cache, bits=4, k_axis="channel", v_axis="token")
        return q_cache, {
            "pre_compressed_bytes": q_info["serialized_bytes"],
            "realized_frac_of_fp16": float(q_info["serialized_bytes"] / max(full_bytes, 1)),
            "evidence_retained_selective_fp16": 0.0,
            "evidence_retained_any_fp16": 0.0,
            "k_rel_l2_err": q_info["k_rel_l2_err"],
            "v_rel_l2_err": q_info["v_rel_l2_err"],
            "stream_quant": {"bits": 4, "k_axis": "channel", "v_axis": "token"},
        }

    if policy == "hybrid_fp16_int2":
        alloc = solve_hybrid_allocation(pre_len, budget_bytes, n_layers, n_kv_heads, head_dim)
        k_sel = alloc["k_selective_fp16"]
        new_kv = []
        sel_fracs, any_fracs = [], []
        k_err_sq = k_ref_sq = v_err_sq = v_ref_sq = 0.0
        for i in range(n_layers):
            k, v = cache_get_kv(pre_cache, i)
            imp = importance_per_layer[i] if importance_per_layer is not None else None
            sel_idx = select_eviction_indices(pre_len, k_sel, importance=imp)
            sel_set = set(sel_idx.tolist())
            low_idx = torch.tensor([p for p in range(pre_len) if p not in sel_set], dtype=torch.long)

            k_out = k.clone()
            v_out = v.clone()
            residual_indices: set[int] = set()
            if len(low_idx) > 0:
                n_comp = (len(low_idx) // GROUP_SIZE) * GROUP_SIZE
                if n_comp > 0:
                    comp_idx = low_idx[:n_comp].to(k.device)
                    pk = pack_kv_tensor(k.index_select(2, comp_idx), bits=2, axis="channel")
                    pv = pack_kv_tensor(v.index_select(2, comp_idx), bits=2, axis="token")
                    k_out[:, :, comp_idx, :] = unpack_kv_tensor(pk, out_dtype=k.dtype)
                    v_out[:, :, comp_idx, :] = unpack_kv_tensor(pv, out_dtype=v.dtype)
                residual_indices = set(low_idx[n_comp:].tolist())

            k_err_sq += float((k_out.float() - k.float()).pow(2).sum().item())
            k_ref_sq += float(k.float().pow(2).sum().item())
            v_err_sq += float((v_out.float() - v.float()).pow(2).sum().item())
            v_ref_sq += float(v.float().pow(2).sum().item())
            new_kv.append((k_out, v_out))

            if ev_set:
                sel_fracs.append(len(ev_set & sel_set) / len(ev_set))
                any_fracs.append(len(ev_set & (sel_set | residual_indices)) / len(ev_set))

        return cache_from_kv_list(new_kv), {
            "pre_compressed_bytes": int(alloc["total_bytes"]),
            "realized_frac_of_fp16": float(alloc["total_bytes"] / max(full_bytes, 1)),
            "k_selective_fp16": int(k_sel),
            "n_int2_groups": int(alloc["n_int2_groups"]),
            "residual_fp16_positions": int(alloc["residual_fp16_positions"]),
            "evidence_retained_selective_fp16": float(sum(sel_fracs) / len(sel_fracs)) if sel_fracs else 1.0,
            "evidence_retained_any_fp16": float(sum(any_fracs) / len(any_fracs)) if any_fracs else 1.0,
            "k_rel_l2_err": float(math.sqrt(k_err_sq / max(k_ref_sq, 1e-12))),
            "v_rel_l2_err": float(math.sqrt(v_err_sq / max(v_ref_sq, 1e-12))),
            "stream_quant": {"bits": 2, "k_axis": "channel", "v_axis": "token"},
        }

    raise ValueError(f"Unknown Part C policy '{policy}'")


def _sequential_suffix_and_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    cache: DynamicCache,
    question_ids: list[int],
    pre_len: int,
    max_new_tokens: int,
    device: torch.device,
    stream_quant: dict[str, Any] | None = None,
    ref_answer_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Append question suffix and generate greedily one token at a time.

    Quantizes newly completed 64-token trailing groups immediately on group completion
    (preventing future-token leakage) and scores reference-answer NLL sequentially.
    """
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and getattr(gen_cfg, "eos_token_id", None) is not None:
        e = gen_cfg.eos_token_id
        if isinstance(e, list):
            eos_ids.update(int(x) for x in e)
        else:
            eos_ids.add(int(e))

    n_layers = cache_num_layers(cache)
    unquantized_tail = pre_len % GROUP_SIZE if stream_quant is not None else 0

    def _maybe_quantize_completed_group(cur_cache: DynamicCache, tail_len: int) -> int:
        if stream_quant is None or tail_len < GROUP_SIZE:
            return tail_len
        for i in range(n_layers):
            k, v = cache_get_kv(cur_cache, i)
            pk = pack_kv_tensor(k[:, :, -GROUP_SIZE:, :], bits=stream_quant["bits"], axis=stream_quant["k_axis"])
            pv = pack_kv_tensor(v[:, :, -GROUP_SIZE:, :], bits=stream_quant["bits"], axis=stream_quant["v_axis"])
            k[:, :, -GROUP_SIZE:, :] = unpack_kv_tensor(pk, out_dtype=k.dtype)
            v[:, :, -GROUP_SIZE:, :] = unpack_kv_tensor(pv, out_dtype=v.dtype)
        return 0

    logical_pos = pre_len
    last_logits: torch.Tensor | None = None

    with torch.no_grad():
        for tok_id in question_ids:
            inp = torch.tensor([[tok_id]], dtype=torch.long, device=device)
            pos = torch.tensor([[logical_pos]], dtype=torch.long, device=device)
            out = model(input_ids=inp, past_key_values=cache, position_ids=pos, use_cache=True)
            cache = out.past_key_values
            last_logits = out.logits[:, -1, :]
            logical_pos += 1
            if stream_quant is not None:
                unquantized_tail += 1
                unquantized_tail = _maybe_quantize_completed_group(cache, unquantized_tail)

        assert last_logits is not None
        post_prefill_logits = last_logits.detach().float().clone()

        # Score reference-answer NLL sequentially on a cloned cache (without polluting free-running decode)
        ref_ans_nll = None
        if ref_answer_ids:
            nll_cache = clone_cache(cache)
            nll_pos = logical_pos
            nll_tail = unquantized_tail
            cur_logits = last_logits
            nll_sum = 0.0
            for t_idx, ans_tok in enumerate(ref_answer_ids):
                nll_sum += float(
                    F.cross_entropy(
                        cur_logits.float(),
                        torch.tensor([ans_tok], dtype=torch.long, device=device),
                        reduction="sum",
                    ).item()
                )
                if t_idx + 1 < len(ref_answer_ids):
                    inp = torch.tensor([[ans_tok]], dtype=torch.long, device=device)
                    pos = torch.tensor([[nll_pos]], dtype=torch.long, device=device)
                    out = model(input_ids=inp, past_key_values=nll_cache, position_ids=pos, use_cache=True)
                    nll_cache = out.past_key_values
                    cur_logits = out.logits[:, -1, :]
                    nll_pos += 1
                    if stream_quant is not None:
                        nll_tail += 1
                        nll_tail = _maybe_quantize_completed_group(nll_cache, nll_tail)
            ref_ans_nll = float(nll_sum / len(ref_answer_ids))
            del nll_cache

        # Free-running greedy decode
        t_dec0 = time.perf_counter()
        generated_ids: list[int] = []
        cur_logits = last_logits
        for _ in range(max_new_tokens):
            next_tok = int(cur_logits.argmax(dim=-1).item())
            generated_ids.append(next_tok)
            if next_tok in eos_ids:
                break
            inp = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            pos = torch.tensor([[logical_pos]], dtype=torch.long, device=device)
            out = model(input_ids=inp, past_key_values=cache, position_ids=pos, use_cache=True)
            cache = out.past_key_values
            cur_logits = out.logits[:, -1, :]
            logical_pos += 1
            if stream_quant is not None:
                unquantized_tail += 1
                unquantized_tail = _maybe_quantize_completed_group(cache, unquantized_tail)
        sync_device(device)
        decode_sec = time.perf_counter() - t_dec0

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "generated_ids": generated_ids,
        "generated_text": gen_text,
        "extracted_code": extract_five_digit_code(gen_text),
        "post_prefill_logits": post_prefill_logits,
        "ref_answer_nll": ref_ans_nll,
        "decode_sec": float(decode_sec),
        "generated_token_count": len(generated_ids),
        "final_physical_cache_len": int(cache.get_seq_length()),
    }


def _logit_kl_from_fp16(fp16_logits: torch.Tensor, comp_logits: torch.Tensor) -> float:
    log_p = F.log_softmax(fp16_logits.float(), dim=-1)
    log_q = F.log_softmax(comp_logits.float(), dim=-1)
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum(dim=-1).mean().item())


def run_stage_c1(args: argparse.Namespace, model=None, tokenizer=None) -> dict[str, Any]:
    """C1: Validate split-prefill vs full-forward logits, 100% retention, RoPE & finite logits."""
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    if model is None or tokenizer is None:
        model, tokenizer, device, dt = _load_model(args.model_id, args.dtype, args.device)
    else:
        device = next(model.parameters()).device
        dt = next(model.parameters()).dtype

    _, cases = _load_kv_cases(args.data_dir, limit=2)
    parsed = split_prompt_and_map_evidence(tokenizer, cases[0])
    full_ids = torch.tensor([parsed["full_ids"]], dtype=torch.long, device=device)
    pre_ids = torch.tensor([parsed["pre_ids"]], dtype=torch.long, device=device)
    q_ids = torch.tensor([parsed["question_ids"]], dtype=torch.long, device=device)

    with torch.no_grad():
        full_out = model(input_ids=full_ids, use_cache=False)
        full_last = full_out.logits[:, -1, :].float()

        pre_out = model(input_ids=pre_ids, use_cache=True)
        q_pos = torch.arange(parsed["pre_len"], parsed["full_len"], device=device).unsqueeze(0)
        split_out = model(
            input_ids=q_ids,
            past_key_values=pre_out.past_key_values,
            position_ids=q_pos,
            use_cache=True,
        )
        split_last = split_out.logits[:, -1, :].float()

    max_diff = float((full_last - split_last).abs().max().item())
    all_finite = bool(torch.isfinite(split_last).all().item())
    phys_len_ok = int(split_out.past_key_values.get_seq_length()) == parsed["full_len"]

    report = {
        "status": "completed",
        "model_id": args.model_id,
        "model_revision": resolve_model_revision(model, args.model_id),
        "dtype": str(dt),
        "device": str(device),
        "case_id": cases[0]["id"],
        "full_len": parsed["full_len"],
        "pre_len": parsed["pre_len"],
        "question_len": parsed["question_len"],
        "evidence_token_count": len(parsed["evidence_token_indices"]),
        "split_vs_full_max_abs_logit_diff": max_diff,
        "all_logits_finite": all_finite,
        "physical_cache_len_matches_full": phys_len_ok,
        "passed": all_finite and phys_len_ok and max_diff < 5e-2,
    }
    atomic_write_json(dirs["results"] / "partC_c1_validation.json", report)
    update_condition_status(
        dirs["results"] / "condition_status.json", "partC_c1", "completed", report
    )
    print(
        f"[Part C1] max_abs_diff={max_diff:.6f}, finite={all_finite}, "
        f"phys_len_ok={phys_len_ok}, passed={report['passed']}",
        flush=True,
    )
    return report


def run_stage_eval_all(args: argparse.Namespace, model=None, tokenizer=None) -> dict[str, Any]:
    """C2-C5: Evaluate FP16 reference and all 5 matched-INT4-budget policies on all prompts."""
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    if model is None or tokenizer is None:
        model, tokenizer, device, dt = _load_model(args.model_id, args.dtype, args.device)
    else:
        device = next(model.parameters()).device
        dt = next(model.parameters()).dtype

    limit = 6 if args.smoke else None
    cases_path, cases = _load_kv_cases(args.data_dir, limit=limit)
    n_layers = int(model.config.num_hidden_layers)
    n_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)

    atomic_write_json(
        dirs["configs"] / "partC_env_metadata.json",
        collect_env_metadata(
            data_dir=args.data_dir,
            model_revisions={args.model_id: resolve_model_revision(model, args.model_id)},
            seed=args.seed,
            extra={
                "stage": "task2_partC",
                "smoke": bool(args.smoke),
                "dtype": str(dt),
                "device": str(device),
                "cases_path": str(cases_path),
                "num_cases": len(cases),
            },
        ),
    )

    per_prompt_path = dirs["results"] / "partC_per_prompt_results.jsonl"
    completed_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and per_prompt_path.exists():
        for rec in read_jsonl(per_prompt_path):
            completed_by_id[str(rec["id"])] = rec
        print(f"[RESUME] Loaded {len(completed_by_id)} completed Part C prompts.", flush=True)

    ordered_records: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        cid = str(case["id"])
        if cid in completed_by_id:
            ordered_records.append(completed_by_id[cid])
            continue

        parsed = split_prompt_and_map_evidence(tokenizer, case)
        pre_ids_t = torch.tensor([parsed["pre_ids"]], dtype=torch.long, device=device)
        q_ids_t = torch.tensor([parsed["question_ids"]], dtype=torch.long, device=device)
        pre_len = parsed["pre_len"]
        budget_bytes = kv_quant_reference_bytes(pre_len, n_layers, n_kv_heads, head_dim, bits=4)

        # Shared prefill of pre-question context
        reset_peak_memory(device)
        sync_device(device)
        t_pre0 = time.perf_counter()
        with torch.no_grad():
            pre_out = model(input_ids=pre_ids_t, use_cache=True)
        sync_device(device)
        prefill_sec = time.perf_counter() - t_pre0
        prefill_mem = get_memory_stats(device)
        pre_cache = pre_out.past_key_values

        # Question-aware importance scoring (used by C2b and C4)
        sync_device(device)
        t_imp0 = time.perf_counter()
        imp_scores = compute_question_aware_importance(model, pre_cache, q_ids_t, pre_len)
        sync_device(device)
        scoring_sec = time.perf_counter() - t_imp0

        # Run FP16 reference first to obtain reference logits & reference answer tokens
        fp16_cache, fp16_info = _apply_policy_to_pre_cache(
            "fp16_full", pre_cache, pre_len, parsed["evidence_token_indices"],
            None, budget_bytes, n_layers, n_kv_heads, head_dim,
        )
        fp16_run = _sequential_suffix_and_decode(
            model, tokenizer, fp16_cache, parsed["question_ids"], pre_len,
            args.max_new_tokens, device, stream_quant=None, ref_answer_ids=None,
        )
        ref_ans_ids = fp16_run["generated_ids"]
        # Also score FP16's own NLL on ref_ans_ids
        fp16_cache2 = clone_cache(pre_cache)
        fp16_nll_run = _sequential_suffix_and_decode(
            model, tokenizer, fp16_cache2, parsed["question_ids"], pre_len,
            1, device, stream_quant=None, ref_answer_ids=ref_ans_ids,
        )
        fp16_correct = int(fp16_run["extracted_code"] == str(case["code"]))
        shared_decode_fp16_bytes = kv_full_fp16_bytes(
            parsed["question_len"] + fp16_run["generated_token_count"],
            n_layers, n_kv_heads, head_dim,
        )

        policy_outcomes: dict[str, Any] = {
            "fp16_full": {
                "exact_match": fp16_correct,
                "extracted_code": fp16_run["extracted_code"],
                "generated_text": fp16_run["generated_text"],
                "ref_answer_nll": fp16_nll_run["ref_answer_nll"],
                "post_prefill_logit_kl": 0.0,
                "evidence_retained_selective_fp16": 1.0,
                "evidence_retained_any_fp16": 1.0,
                "pre_compressed_bytes": fp16_info["pre_compressed_bytes"],
                "shared_decode_fp16_bytes": shared_decode_fp16_bytes,
                "total_live_cache_bytes": fp16_info["pre_compressed_bytes"] + shared_decode_fp16_bytes,
                "realized_frac_of_fp16": 1.0,
                "k_rel_l2_err": 0.0,
                "v_rel_l2_err": 0.0,
                "scoring_sec": 0.0,
                "prefill_sec": float(prefill_sec),
                "decode_sec": fp16_run["decode_sec"],
                "peak_allocated_bytes": prefill_mem["peak_allocated_bytes"],
            }
        }

        for policy in POLICIES[1:]:
            reset_peak_memory(device)
            t_pol0 = time.perf_counter()
            pol_cache, pol_info = _apply_policy_to_pre_cache(
                policy, pre_cache, pre_len, parsed["evidence_token_indices"],
                imp_scores, budget_bytes, n_layers, n_kv_heads, head_dim,
            )
            pol_compress_sec = time.perf_counter() - t_pol0
            pol_run = _sequential_suffix_and_decode(
                model, tokenizer, pol_cache, parsed["question_ids"], pre_len,
                args.max_new_tokens, device,
                stream_quant=pol_info["stream_quant"],
                ref_answer_ids=ref_ans_ids,
            )
            pol_mem = get_memory_stats(device)
            kl = _logit_kl_from_fp16(fp16_run["post_prefill_logits"], pol_run["post_prefill_logits"])
            dec_fp16_bytes = kv_full_fp16_bytes(
                parsed["question_len"] + pol_run["generated_token_count"],
                n_layers, n_kv_heads, head_dim,
            )
            policy_outcomes[policy] = {
                "exact_match": int(pol_run["extracted_code"] == str(case["code"])),
                "extracted_code": pol_run["extracted_code"],
                "generated_text": pol_run["generated_text"],
                "ref_answer_nll": pol_run["ref_answer_nll"],
                "post_prefill_logit_kl": kl,
                "evidence_retained_selective_fp16": pol_info["evidence_retained_selective_fp16"],
                "evidence_retained_any_fp16": pol_info["evidence_retained_any_fp16"],
                "pre_compressed_bytes": pol_info["pre_compressed_bytes"],
                "shared_decode_fp16_bytes": dec_fp16_bytes,
                "total_live_cache_bytes": pol_info["pre_compressed_bytes"] + dec_fp16_bytes,
                "realized_frac_of_fp16": pol_info["realized_frac_of_fp16"],
                "k_rel_l2_err": pol_info["k_rel_l2_err"],
                "v_rel_l2_err": pol_info["v_rel_l2_err"],
                "scoring_sec": float(
                    (scoring_sec if policy in ("evict_question_aware", "hybrid_fp16_int2") else 0.0)
                    + pol_compress_sec
                ),
                "prefill_sec": float(prefill_sec),
                "decode_sec": pol_run["decode_sec"],
                "peak_allocated_bytes": max(
                    prefill_mem["peak_allocated_bytes"], pol_mem["peak_allocated_bytes"]
                ),
            }

        prompt_rec = {
            "id": cid,
            "gold_code": str(case["code"]),
            "target_tokens": int(case["target_tokens"]),
            "position": str(case["position"]),
            "distractors": int(case["distractors"]),
            "encoded_full_len": parsed["full_len"],
            "encoded_pre_len": pre_len,
            "encoded_question_len": parsed["question_len"],
            "evidence_token_count": len(parsed["evidence_token_indices"]),
            "int4_ref_budget_bytes": int(budget_bytes),
            "fp16_full_pre_bytes": int(kv_full_fp16_bytes(pre_len, n_layers, n_kv_heads, head_dim)),
            "policies": policy_outcomes,
        }
        ordered_records.append(prompt_rec)
        atomic_write_jsonl(per_prompt_path, ordered_records)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(cases):
            print(f"  [Part C] Completed {idx + 1}/{len(cases)} prompts.", flush=True)

    _aggregate_part_c_outputs(dirs, ordered_records, seed=args.seed)
    update_condition_status(
        dirs["results"] / "condition_status.json",
        "partC_eval_all",
        "completed",
        {"seed": int(args.seed), "smoke": bool(args.smoke), "num_prompts": len(ordered_records)},
    )
    return {"num_prompts": len(ordered_records)}


def _aggregate_part_c_outputs(
    dirs: dict[str, Path],
    records: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
    n_resamples: int = 2000,
) -> None:
    """Write matched-budget summary CSV, factorial breakdown CSV, bootstrap CIs, and qualitative JSON."""
    res_dir = dirs["results"]
    fp16_correct_mask = [r["policies"]["fp16_full"]["exact_match"] == 1 for r in records]

    # 1. Single matched-INT4-budget summary table
    summary_rows: list[dict[str, Any]] = []
    for pol in POLICIES:
        ems = [r["policies"][pol]["exact_match"] for r in records]
        cond_ems = [em for em, ok in zip(ems, fp16_correct_mask) if ok]
        nlls = [r["policies"][pol]["ref_answer_nll"] for r in records if r["policies"][pol]["ref_answer_nll"] is not None]
        kls = [r["policies"][pol]["post_prefill_logit_kl"] for r in records]
        ev_sel = [r["policies"][pol]["evidence_retained_selective_fp16"] for r in records]
        ev_any = [r["policies"][pol]["evidence_retained_any_fp16"] for r in records]
        pre_b = [r["policies"][pol]["pre_compressed_bytes"] for r in records]
        fracs = [r["policies"][pol]["realized_frac_of_fp16"] for r in records]
        k_err = [r["policies"][pol]["k_rel_l2_err"] for r in records]
        v_err = [r["policies"][pol]["v_rel_l2_err"] for r in records]
        sc_t = [r["policies"][pol]["scoring_sec"] * 1000.0 for r in records]
        pr_t = [r["policies"][pol]["prefill_sec"] * 1000.0 for r in records]
        dc_t = [r["policies"][pol]["decode_sec"] * 1000.0 for r in records]
        vram = [r["policies"][pol]["peak_allocated_bytes"] for r in records]

        summary_rows.append(
            {
                "policy": pol,
                "num_prompts": len(records),
                "unconditional_exact_match": float(np.mean(ems)),
                "conditional_on_fp16_correct": float(np.mean(cond_ems)) if cond_ems else 0.0,
                "fp16_correct_count": int(sum(fp16_correct_mask)),
                "mean_ref_answer_nll": float(np.mean(nlls)) if nlls else 0.0,
                "mean_post_prefill_kl": float(np.mean(kls)),
                "mean_ev_retained_selective_fp16": float(np.mean(ev_sel)),
                "mean_ev_retained_any_fp16": float(np.mean(ev_any)),
                "mean_pre_packed_bytes": float(np.mean(pre_b)),
                "mean_realized_frac_fp16": float(np.mean(fracs)),
                "mean_k_rel_l2_err": float(np.mean(k_err)),
                "mean_v_rel_l2_err": float(np.mean(v_err)),
                "mean_scoring_ms": float(np.mean(sc_t)),
                "mean_prefill_ms": float(np.mean(pr_t)),
                "mean_decode_ms": float(np.mean(dc_t)),
                "max_peak_allocated_bytes": int(max(vram)),
            }
        )

    def _dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        buf = StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
        atomic_write_text(path, buf.getvalue())

    _dump_csv(res_dir / "partC_kv_fidelity_matched_budget.csv", summary_rows)

    # 2. Factorial breakdown by (target_tokens, position, distractors)
    cells: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for r in records:
        cells.setdefault((r["target_tokens"], r["position"], r["distractors"]), []).append(r)
    fact_rows: list[dict[str, Any]] = []
    for (t_len, pos, dist), cell_recs in sorted(cells.items()):
        for pol in POLICIES:
            ems = [r["policies"][pol]["exact_match"] for r in cell_recs]
            fact_rows.append(
                {
                    "target_tokens": t_len,
                    "position": pos,
                    "distractors": dist,
                    "n_cell": len(cell_recs),
                    "policy": pol,
                    "exact_match": float(np.mean(ems)),
                    "mean_ref_answer_nll": float(
                        np.mean([r["policies"][pol]["ref_answer_nll"] for r in cell_recs])
                    ),
                    "mean_ev_retained_selective_fp16": float(
                        np.mean([r["policies"][pol]["evidence_retained_selective_fp16"] for r in cell_recs])
                    ),
                }
            )
    _dump_csv(res_dir / "partC_kv_breakdown_factorial.csv", fact_rows)

    # 3. Paired bootstrap 95% CIs over prompts (>= 2,000 resamples)
    pairs = [
        ("evict_question_aware_minus_sink_recent", "evict_question_aware", "evict_sink_recent"),
        ("quant_channelK_minus_tokenwise", "quant_int4_channelK_tokenV", "quant_int4_tokenwise"),
        ("hybrid_minus_question_aware", "hybrid_fp16_int2", "evict_question_aware"),
        ("hybrid_minus_quant_channelK", "hybrid_fp16_int2", "quant_int4_channelK_tokenV"),
        ("quant_channelK_minus_fp16", "quant_int4_channelK_tokenV", "fp16_full"),
    ]
    rng = np.random.default_rng(seed)
    n = len(records)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_out: dict[str, Any] = {"seed": int(seed), "n_prompts": n, "n_resamples": int(n_resamples), "comparisons": {}}
    for key, pol_a, pol_b in pairs:
        a = np.array([r["policies"][pol_a]["exact_match"] for r in records], dtype=np.float64)
        b = np.array([r["policies"][pol_b]["exact_match"] for r in records], dtype=np.float64)
        diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        lo, hi = np.percentile(diffs, [2.5, 97.5]).tolist()
        boot_out["comparisons"][key] = {
            "policy_a": pol_a,
            "policy_b": pol_b,
            "acc_a": float(a.mean()),
            "acc_b": float(b.mean()),
            "diff_a_minus_b": float(a.mean() - b.mean()),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
        }
    atomic_write_json(res_dir / "partC_bootstrap_ci.json", boot_out)

    # 4. Qualitative aligned generations (fixed deterministic selection of up to 8 prompts)
    qual: list[dict[str, Any]] = []
    for r in records[:8]:
        qual.append(
            {
                "id": r["id"],
                "gold_code": r["gold_code"],
                "target_tokens": r["target_tokens"],
                "position": r["position"],
                "distractors": r["distractors"],
                "encoded_pre_len": r["encoded_pre_len"],
                "evidence_token_count": r["evidence_token_count"],
                "generations": {
                    pol: {
                        "exact_match": r["policies"][pol]["exact_match"],
                        "extracted_code": r["policies"][pol]["extracted_code"],
                        "generated_text": r["policies"][pol]["generated_text"],
                        "evidence_retained_selective_fp16": r["policies"][pol]["evidence_retained_selective_fp16"],
                    }
                    for pol in POLICIES
                },
            }
        )
    atomic_write_json(
        dirs["evidence"] / "partC_qualitative_generations.json",
        {"selection_rule": "First 8 prompts in locked kv_retrieval_cases.jsonl.gz", "cases": qual},
    )


def run_stage_bench_packed(args: argparse.Namespace) -> list[dict[str, Any]]:
    """C5: Independently benchmark packed-cache encode, decode/unpack, and decode-step latency."""
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # Match Qwen2.5-1.5B-Instruct KV geometry: n_layers=28, H_kv=2, d_h=128
    n_layers = 28
    n_kv_heads = 2
    n_q_heads = 12
    head_dim = 128

    batches = [1] if args.smoke else list(args.batches)
    lengths = [256] if args.smoke else list(args.lengths)
    warmup = 1 if args.smoke else 3
    reps = 2 if args.smoke else 10

    configs = [
        ("fp16_ref", 16, "none", "none"),
        ("int4_tokenwise", 4, "token", "token"),
        ("int4_channelK_tokenV", 4, "channel", "token"),
        ("int2_channelK_tokenV", 2, "channel", "token"),
    ]

    rows: list[dict[str, Any]] = []
    for bsz in batches:
        for seq_len in lengths:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Synthetic full-layer K, V, and 1-token query Q for matched decode-step attention
            k_one = torch.randn((bsz, n_kv_heads, seq_len, head_dim), device=device, dtype=dtype)
            v_one = torch.randn((bsz, n_kv_heads, seq_len, head_dim), device=device, dtype=dtype)
            q_step = torch.randn((bsz, n_q_heads, 1, head_dim), device=device, dtype=dtype)
            gqa = n_q_heads // n_kv_heads

            for cfg_name, bits, k_ax, v_ax in configs:
                if bits == 16:
                    ser_bytes = kv_full_fp16_bytes(seq_len, n_layers, n_kv_heads, head_dim, batch_size=bsz)
                    enc_ms = dec_ms = 0.0

                    def _step_fp16() -> torch.Tensor:
                        ke = k_one.repeat_interleave(gqa, dim=1)
                        ve = v_one.repeat_interleave(gqa, dim=1)
                        return F.scaled_dot_product_attention(q_step, ke, ve, is_causal=False)

                    step_stats = measure_latency(_step_fp16, warmup=warmup, reps=reps, device=device)
                else:
                    pk_ref = pack_kv_tensor(k_one, bits=bits, axis=k_ax)
                    pv_ref = pack_kv_tensor(v_one, bits=bits, axis=v_ax)
                    ser_bytes = (pk_ref.serialized_bytes() + pv_ref.serialized_bytes()) * n_layers

                    enc_stats = measure_latency(
                        lambda: (
                            pack_kv_tensor(k_one, bits=bits, axis=k_ax),
                            pack_kv_tensor(v_one, bits=bits, axis=v_ax),
                        ),
                        warmup=warmup, reps=reps, device=device,
                    )
                    dec_stats = measure_latency(
                        lambda: (
                            unpack_kv_tensor(pk_ref, out_dtype=dtype),
                            unpack_kv_tensor(pv_ref, out_dtype=dtype),
                        ),
                        warmup=warmup, reps=reps, device=device,
                    )
                    enc_ms = enc_stats["mean_ms"] * n_layers
                    dec_ms = dec_stats["mean_ms"] * n_layers

                    def _step_unpacked() -> torch.Tensor:
                        ku = unpack_kv_tensor(pk_ref, out_dtype=dtype).repeat_interleave(gqa, dim=1)
                        vu = unpack_kv_tensor(pv_ref, out_dtype=dtype).repeat_interleave(gqa, dim=1)
                        return F.scaled_dot_product_attention(q_step, ku, vu, is_causal=False)

                    step_stats = measure_latency(_step_unpacked, warmup=warmup, reps=reps, device=device)

                row = {
                    "config": cfg_name,
                    "bits": bits,
                    "batch_size": int(bsz),
                    "seq_len": int(seq_len),
                    "n_layers_scaled": n_layers,
                    "timed_reps": int(reps),
                    "serialized_total_bytes": int(ser_bytes),
                    "encode_all_layers_ms": round(enc_ms, 4),
                    "decode_unpack_all_layers_ms": round(dec_ms, 4),
                    "decode_step_all_layers_ms": round(step_stats["mean_ms"] * n_layers, 4),
                    "decode_step_per_layer_ms": round(step_stats["mean_ms"], 4),
                    "peak_allocated_bytes_single_layer": step_stats["peak_allocated_bytes"],
                }
                rows.append(row)
                print(
                    f"  [Part C5 Bench] B={bsz} L={seq_len:4d} {cfg_name:22s} -> "
                    f"ser={ser_bytes/1e6:.2f}MB enc={row['encode_all_layers_ms']:.2f}ms "
                    f"unpack={row['decode_unpack_all_layers_ms']:.2f}ms "
                    f"step={row['decode_step_all_layers_ms']:.2f}ms",
                    flush=True,
                )

    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    atomic_write_text(dirs["results"] / "partC_packed_encode_decode_bench.csv", buf.getvalue())
    update_condition_status(
        dirs["results"] / "condition_status.json",
        "partC_bench_packed",
        "completed",
        {"seed": int(args.seed), "smoke": bool(args.smoke), "rows": len(rows)},
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.stage == "c1":
        run_stage_c1(args)
    elif args.stage == "eval_all":
        run_stage_eval_all(args)
    elif args.stage == "bench_packed":
        run_stage_bench_packed(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
