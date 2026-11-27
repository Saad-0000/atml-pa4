"""Task 3 Steps 2, 6 & 7: Speculative Decoding Evaluation, Diagnostics & Standalone Quality.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task3.run_spec_eval --draft pretrained --gammas 2 4 --data-dir <DATA_DIR> --out-dir <T3_OUT> --seed 6304
  python -m task3.run_spec_eval --draft all_distilled --gammas 2 4 --include-unmerged-diag --include-fixed64-diag --resume --data-dir <DATA_DIR> --out-dir <T3_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.env import collect_env_metadata, resolve_model_revision
from common.io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    ensure_task_dirs,
    read_json,
    read_jsonl,
    resolve_smoke_out_dir,
    update_condition_status,
)
from common.seeding import DEFAULT_SEED, set_seed
from common.timing import get_memory_stats, measure_latency, reset_peak_memory
from task3.data import (
    DRAFT_MODEL_ID,
    TARGET_MODEL_ID,
    check_short_form_match,
    extract_gsm8k_prediction,
    format_chat_prompt,
    get_eos_token_ids,
    load_task3_test_prompts,
    verify_tokenizer_agreement,
)
from task3.sampler import (
    analytical_speedup,
    compute_breakeven_alpha,
    exact_speculative_generate,
    target_only_generate,
)
from task3.train_distill import _load_lora_state_dict, _wrap_student_with_lora


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 3 Steps 2, 6 & 7: Counterbalanced exact speculative decoding eval & diagnostics."
    )
    parser.add_argument(
        "--draft",
        type=str,
        required=True,
        choices=("pretrained", "all_distilled"),
    )
    parser.add_argument("--gammas", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--include-unmerged-diag", action="store_true")
    parser.add_argument("--include-fixed64-diag", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


def _measure_isolated_step_costs(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    gammas: list[int],
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    """Measure isolated single-step draft/target decode cost and gamma-token verification cost."""
    prompt_lens = [32] if smoke else [64, 256, 512]
    warmup = 1 if smoke else 3
    reps = 2 if smoke else 10
    vocab = int(getattr(target_model.config, "vocab_size", 1000))

    per_len: dict[str, Any] = {}
    c_vals, v_vals_by_g = [], {g: [] for g in gammas}

    with torch.no_grad():
        for plen in prompt_lens:
            pre_ids = torch.randint(10, min(vocab, 500), (1, plen), device=device)
            t_pre = target_model(input_ids=pre_ids, use_cache=True).past_key_values
            d_pre = draft_model(input_ids=pre_ids, use_cache=True).past_key_values
            one_tok = torch.randint(10, min(vocab, 500), (1, 1), device=device)

            # Single-step target & draft latency (using cloned/cropped cache via non-mutating fresh prefill or no-cache?
            # Even cleaner: copy cache tensors inside a small wrapper or pass fresh KV of length plen)
            def _t_step() -> Any:
                out = target_model(input_ids=one_tok, past_key_values=t_pre, use_cache=True)
                # Crop back to plen so every rep sees exact same KV length!
                for layer in t_pre.layers:
                    layer.keys = layer.keys[:, :, :plen, :]
                    layer.values = layer.values[:, :, :plen, :]
                return out

            def _d_step() -> Any:
                out = draft_model(input_ids=one_tok, past_key_values=d_pre, use_cache=True)
                for layer in d_pre.layers:
                    layer.keys = layer.keys[:, :, :plen, :]
                    layer.values = layer.values[:, :, :plen, :]
                return out

            t1 = measure_latency(_t_step, warmup=warmup, reps=reps, device=device)["mean_ms"]
            d1 = measure_latency(_d_step, warmup=warmup, reps=reps, device=device)["mean_ms"]
            c_ratio = d1 / max(t1, 1e-6)
            c_vals.append(c_ratio)

            vg_map = {}
            for g in gammas:
                ver_toks = torch.randint(10, min(vocab, 500), (1, g + 1), device=device)

                def _t_ver(vt: torch.Tensor = ver_toks) -> Any:
                    out = target_model(input_ids=vt, past_key_values=t_pre, use_cache=True)
                    for layer in t_pre.layers:
                        layer.keys = layer.keys[:, :, :plen, :]
                        layer.values = layer.values[:, :, :plen, :]
                    return out

                tg = measure_latency(_t_ver, warmup=warmup, reps=reps, device=device)["mean_ms"]
                vg = tg / max(t1, 1e-6)
                vg_map[str(g)] = {"verify_ms": tg, "v_gamma": vg}
                v_vals_by_g[g].append(vg)

            per_len[str(plen)] = {
                "target_step_ms": t1,
                "draft_step_ms": d1,
                "c": c_ratio,
                "verify_by_gamma": vg_map,
            }

    mean_c = float(np.mean(c_vals))
    mean_v = {str(g): float(np.mean(v_vals_by_g[g])) for g in gammas}
    breakeven = {
        str(g): compute_breakeven_alpha(mean_c, mean_v[str(g)], g) for g in gammas
    }
    return {
        "mean_c": mean_c,
        "mean_v_gamma": mean_v,
        "breakeven_alpha": breakeven,
        "by_prompt_length": per_len,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    atomic_write_text(path, buf.getvalue())


def _rebuild_spec_csvs(results_dir: Path, cost_model: dict[str, Any]) -> None:
    """Aggregate per-condition JSONL files into summary & 4-part timing decomposition CSVs."""
    summary_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for jf in sorted(results_dir.glob("spec_per_prompt_*.jsonl")):
        recs = read_jsonl(jf)
        if not recs:
            continue
        draft_name = recs[0]["draft"]
        gamma = int(recs[0]["gamma"])
        tag = recs[0].get("eval_tag", "eos_aware_64")

        acc_exam = float(np.mean([r["accepted_over_examined"] for r in recs]))
        acc_prop = float(np.mean([r["accepted_over_proposed"] for r in recs]))
        mean_acc_ver = float(np.mean([r["mean_accepted_per_verify"] for r in recs]))
        t_pass_tok = float(np.mean([r["target_passes_per_output_token"] for r in recs]))
        d_passes = float(np.mean([r["draft_forward_passes"] for r in recs]))
        spec_tps = float(np.mean([r["spec_tokens_per_sec"] for r in recs]))
        tgt_tps = float(np.mean([r["target_tokens_per_sec"] for r in recs]))
        meas_speedup = float(np.mean([r["throughput_ratio_spec_over_target"] for r in recs]))

        c_val = float(cost_model.get("mean_c", 0.5))
        vg_val = float(cost_model.get("mean_v_gamma", {}).get(str(gamma), 1.0))
        proj_s = analytical_speedup(acc_exam, c_val, vg_val, gamma)
        be_a = cost_model.get("breakeven_alpha", {}).get(str(gamma), None)

        summary_rows.append(
            {
                "draft": draft_name,
                "gamma": gamma,
                "eval_tag": tag,
                "num_prompt_reps": len(recs),
                "accepted_over_examined": acc_exam,
                "accepted_over_proposed": acc_prop,
                "mean_accepted_per_verify": mean_acc_ver,
                "target_passes_per_output_token": t_pass_tok,
                "mean_draft_passes": d_passes,
                "mean_output_tokens": float(np.mean([r["spec_num_tokens"] for r in recs])),
                "eos_rate": float(np.mean([int(r["spec_hit_eos"]) for r in recs])),
                "truncation_rate": float(np.mean([int(r["spec_truncated"]) for r in recs])),
                "spec_tokens_per_sec": spec_tps,
                "target_tokens_per_sec": tgt_tps,
                "measured_speedup_S_gamma": meas_speedup,
                "analytical_speedup_S_hat": proj_s,
                "c_draft_over_target": c_val,
                "v_gamma": vg_val,
                "breakeven_alpha": "" if be_a is None else be_a,
                "max_peak_allocated_bytes": int(max(r.get("peak_allocated_bytes", 0) for r in recs)),
            }
        )
        timing_rows.append(
            {
                "draft": draft_name,
                "gamma": gamma,
                "eval_tag": tag,
                "mean_prefill_ms": float(np.mean([r["spec_prefill_sec"] * 1000.0 for r in recs])),
                "mean_draft_proposals_ms": float(np.mean([r["round_timing_sec"]["draft_proposals_sec"] * 1000.0 for r in recs])),
                "mean_target_verification_ms": float(np.mean([r["round_timing_sec"]["target_verification_sec"] * 1000.0 for r in recs])),
                "mean_accept_residual_ms": float(np.mean([r["round_timing_sec"]["accept_residual_sec"] * 1000.0 for r in recs])),
                "mean_cache_rollback_ms": float(np.mean([r["round_timing_sec"]["cache_rollback_advance_sec"] * 1000.0 for r in recs])),
                "mean_host_overhead_ms": float(np.mean([r["round_timing_sec"]["host_overhead_sec"] * 1000.0 for r in recs])),
                "mean_total_decode_ms": float(np.mean([r["spec_decode_sec"] * 1000.0 for r in recs])),
                "mean_target_only_decode_ms": float(np.mean([r["target_decode_sec"] * 1000.0 for r in recs])),
            }
        )

    _write_csv(results_dir / "speculative_decoding_summary.csv", summary_rows)
    _write_csv(results_dir / "speculative_round_timing_decomposition.csv", timing_rows)


def run_spec_eval(
    args: argparse.Namespace,
    target_model: torch.nn.Module | None = None,
    draft_models: dict[str, torch.nn.Module] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dt = torch.float16 if device.type == "cuda" else torch.float32

    test_prompts = load_task3_test_prompts(args.data_dir)
    if args.smoke:
        test_prompts = test_prompts[:4]
        n_reps = 1
        speed_cap = 12
    else:
        n_reps = 2
        speed_cap = 64

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, use_fast=True)
        draft_tok = AutoTokenizer.from_pretrained(DRAFT_MODEL_ID, use_fast=True)
        tok_agree = verify_tokenizer_agreement(
            tokenizer, draft_tok, [p["prompt"] for p in test_prompts]
        )
        atomic_write_json(dirs["configs"] / "tokenizer_agreement.json", tok_agree)

    if target_model is None:
        target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, dtype=dt).to(device).eval()
    eos_ids = get_eos_token_ids(tokenizer, target_model)

    if draft_models is None:
        draft_models = {
            "original_0.5b": AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_ID, dtype=dt).to(device).eval()
        }
        if args.draft == "all_distilled":
            for run_id in ("off_policy_s6304", "off_policy_s6305", "on_policy_s6304", "on_policy_s6305"):
                mdir = dirs["checkpoints"] / run_id / "merged_fp16"
                if mdir.exists():
                    draft_models[run_id] = AutoModelForCausalLM.from_pretrained(str(mdir), dtype=dt).to(device).eval()

    atomic_write_json(
        dirs["configs"] / "env_metadata.json",
        collect_env_metadata(
            data_dir=args.data_dir,
            model_revisions={
                TARGET_MODEL_ID: resolve_model_revision(target_model, TARGET_MODEL_ID),
                DRAFT_MODEL_ID: resolve_model_revision(draft_models["original_0.5b"], DRAFT_MODEL_ID),
            },
            seed=args.seed,
            extra={"stage": f"task3_spec_eval_{args.draft}", "smoke": bool(args.smoke)},
        ),
    )

    # Measure isolated cost model parameters c, v_gamma, break-even alpha
    cost_model = _measure_isolated_step_costs(
        target_model, draft_models["original_0.5b"], list(args.gammas), device, smoke=args.smoke
    )
    atomic_write_json(dirs["results"] / "isolated_step_cost_model.json", cost_model)

    # Counterbalanced evaluation across drafts, reps, and gammas
    draft_names = list(draft_models.keys())
    qual_continuations: list[dict[str, Any]] = []

    for rep in range(n_reps):
        ordered_drafts = draft_names if rep % 2 == 0 else list(reversed(draft_names))
        ordered_gammas = list(args.gammas) if rep % 2 == 0 else list(reversed(args.gammas))

        for d_name in ordered_drafts:
            d_mod = draft_models[d_name]
            for g in ordered_gammas:
                cond_tag = f"{d_name}_g{g}_eos64"
                out_jsonl = dirs["results"] / f"spec_per_prompt_{cond_tag}.jsonl"
                done_keys: dict[tuple[str, int], dict[str, Any]] = {}
                if args.resume and out_jsonl.exists():
                    for rec in read_jsonl(out_jsonl):
                        done_keys[(str(rec["prompt_id"]), int(rec["rep"]))] = rec

                records_list = list(done_keys.values())
                for idx, p in enumerate(test_prompts):
                    if (p["prompt_id"], rep) in done_keys:
                        continue
                    chat = format_chat_prompt(tokenizer, p["prompt"])
                    p_ids = tokenizer.encode(chat, add_special_tokens=False)
                    run_seed = int(args.seed) + rep * 10000 + idx * 13

                    # Counterbalance adjacent target-only vs speculative pair order
                    reset_peak_memory(device)
                    if (idx + rep) % 2 == 0:
                        tgt = target_only_generate(target_model, p_ids, speed_cap, eos_ids, device, seed=run_seed)
                        spec = exact_speculative_generate(target_model, d_mod, p_ids, speed_cap, g, eos_ids, device, seed=run_seed)
                    else:
                        spec = exact_speculative_generate(target_model, d_mod, p_ids, speed_cap, g, eos_ids, device, seed=run_seed)
                        tgt = target_only_generate(target_model, p_ids, speed_cap, eos_ids, device, seed=run_seed)
                    mem = get_memory_stats(device)

                    spec_tps = spec["num_tokens"] / max(spec["decode_sec"], 1e-6)
                    tgt_tps = tgt["num_tokens"] / max(tgt["decode_sec"], 1e-6)
                    rec = {
                        "draft": d_name,
                        "gamma": int(g),
                        "eval_tag": "eos_aware_64",
                        "rep": int(rep),
                        "prompt_id": p["prompt_id"],
                        "workload": p["workload"],
                        "spec_num_tokens": spec["num_tokens"],
                        "spec_hit_eos": spec["hit_eos"],
                        "spec_truncated": spec["truncated"],
                        "accepted_over_examined": spec["accepted_over_examined"],
                        "accepted_over_proposed": spec["accepted_over_proposed"],
                        "mean_accepted_per_verify": spec["mean_accepted_per_verify"],
                        "target_passes_per_output_token": spec["target_passes_per_output_token"],
                        "draft_forward_passes": spec["draft_forward_passes"],
                        "spec_prefill_sec": spec["prefill_sec"],
                        "spec_decode_sec": spec["decode_sec"],
                        "target_decode_sec": tgt["decode_sec"],
                        "spec_tokens_per_sec": float(spec_tps),
                        "target_tokens_per_sec": float(tgt_tps),
                        "throughput_ratio_spec_over_target": float(spec_tps / max(tgt_tps, 1e-6)),
                        "round_timing_sec": spec["round_timing_sec"],
                        "peak_allocated_bytes": mem["peak_allocated_bytes"],
                    }
                    records_list.append(rec)
                    if rep == 0 and g == args.gammas[0] and idx < 4:
                        qual_continuations.append(
                            {
                                "draft": d_name,
                                "gamma": int(g),
                                "prompt_id": p["prompt_id"],
                                "workload": p["workload"],
                                "spec_text": tokenizer.decode(spec["generated_ids"], skip_special_tokens=True),
                                "target_text": tokenizer.decode(tgt["generated_ids"], skip_special_tokens=True),
                                "accepted_over_examined": spec["accepted_over_examined"],
                            }
                        )
                atomic_write_jsonl(out_jsonl, records_list)
                print(f"  [Spec Eval] Completed {cond_tag} rep={rep} ({len(records_list)} rows).", flush=True)

    # Optional fixed-64-token throughput diagnostic (24 math prompts, ignoring EOS for timing only)
    if args.include_fixed64_diag:
        math_prompts = [p for p in test_prompts if p["category"] == "math"][: (2 if args.smoke else 24)]
        for d_name, d_mod in draft_models.items():
            g = args.gammas[-1]
            out_jsonl = dirs["results"] / f"spec_per_prompt_{d_name}_g{g}_fixed64.jsonl"
            f_recs = []
            for idx, p in enumerate(math_prompts):
                p_ids = tokenizer.encode(format_chat_prompt(tokenizer, p["prompt"]), add_special_tokens=False)
                tgt = target_only_generate(target_model, p_ids, speed_cap, eos_ids, device, seed=args.seed + idx, ignore_eos=True)
                spec = exact_speculative_generate(target_model, d_mod, p_ids, speed_cap, g, eos_ids, device, seed=args.seed + idx, ignore_eos=True)
                spec_tps = spec["num_tokens"] / max(spec["decode_sec"], 1e-6)
                tgt_tps = tgt["num_tokens"] / max(tgt["decode_sec"], 1e-6)
                f_recs.append(
                    {
                        "draft": d_name,
                        "gamma": int(g),
                        "eval_tag": "fixed64_ignore_eos",
                        "rep": 0,
                        "prompt_id": p["prompt_id"],
                        "workload": p["workload"],
                        "spec_num_tokens": spec["num_tokens"],
                        "spec_hit_eos": False,
                        "spec_truncated": True,
                        "accepted_over_examined": spec["accepted_over_examined"],
                        "accepted_over_proposed": spec["accepted_over_proposed"],
                        "mean_accepted_per_verify": spec["mean_accepted_per_verify"],
                        "target_passes_per_output_token": spec["target_passes_per_output_token"],
                        "draft_forward_passes": spec["draft_forward_passes"],
                        "spec_prefill_sec": spec["prefill_sec"],
                        "spec_decode_sec": spec["decode_sec"],
                        "target_decode_sec": tgt["decode_sec"],
                        "spec_tokens_per_sec": float(spec_tps),
                        "target_tokens_per_sec": float(tgt_tps),
                        "throughput_ratio_spec_over_target": float(spec_tps / max(tgt_tps, 1e-6)),
                        "round_timing_sec": spec["round_timing_sec"],
                        "peak_allocated_bytes": 0,
                    }
                )
            atomic_write_jsonl(out_jsonl, f_recs)

    # Optional unmerged LoRA wrapper overhead diagnostic (1 adapter)
    if args.include_unmerged_diag:
        for run_id in ("off_policy_s6304", "on_policy_s6304"):
            lora_pt = dirs["checkpoints"] / run_id / "lora_final.pt"
            if lora_pt.exists():
                base_05 = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_ID, dtype=dt).to(device)
                unmerged, _, _ = _wrap_student_with_lora(base_05)
                _load_lora_state_dict(unmerged, torch.load(lora_pt, map_location=device, weights_only=False)["lora_state"])
                unmerged.eval()
                sub = test_prompts[: (2 if args.smoke else 16)]
                u_times, m_times = [], []
                for idx, p in enumerate(sub):
                    p_ids = tokenizer.encode(format_chat_prompt(tokenizer, p["prompt"]), add_special_tokens=False)
                    u_out = exact_speculative_generate(target_model, unmerged, p_ids, speed_cap, 4, eos_ids, device, seed=args.seed + idx)
                    m_out = exact_speculative_generate(target_model, draft_models[run_id], p_ids, speed_cap, 4, eos_ids, device, seed=args.seed + idx)
                    u_times.append(u_out["decode_sec"] * 1000.0)
                    m_times.append(m_out["decode_sec"] * 1000.0)
                merge_path = dirs["results"] / "merged_vs_unmerged_check.json"
                mdoc = read_json(merge_path) if merge_path.exists() else {}
                mdoc["unmerged_wrapper_timing_diag"] = {
                    "run_id": run_id,
                    "num_prompts": len(sub),
                    "unmerged_mean_decode_ms": float(np.mean(u_times)),
                    "merged_mean_decode_ms": float(np.mean(m_times)),
                    "wrapper_overhead_ratio": float(np.mean(u_times) / max(np.mean(m_times), 1e-6)),
                }
                atomic_write_json(merge_path, mdoc)
                break

    # Standalone student answer-quality eval (512 math / 32 extraction)
    if args.draft == "all_distilled" or args.smoke:
        standalone_rows: list[dict[str, Any]] = []
        for d_name, d_mod in draft_models.items():
            for wl in ("math", "short_familiar", "short_shifted"):
                sub = [p for p in test_prompts if p["workload"] == wl]
                if not sub:
                    continue
                correct = extractable = compliant = truncated = 0
                for idx, p in enumerate(sub):
                    p_ids = tokenizer.encode(format_chat_prompt(tokenizer, p["prompt"]), add_special_tokens=False)
                    cap = 16 if args.smoke else int(p["standalone_cap"])
                    gen = target_only_generate(d_mod, p_ids, cap, eos_ids, device, seed=args.seed + idx)
                    text = tokenizer.decode(gen["generated_ids"], skip_special_tokens=True)
                    truncated += int(gen["truncated"])
                    if p["category"] == "math":
                        pred, is_ext = extract_gsm8k_prediction(text)
                        extractable += int(is_ext)
                        correct += int(is_ext and pred == p["gold"])
                        compliant += int(is_ext)
                    else:
                        em, comp = check_short_form_match(text, p["raw"])
                        extractable += int(bool(text.strip()))
                        correct += int(em)
                        compliant += int(comp)
                standalone_rows.append(
                    {
                        "draft": d_name,
                        "workload": wl,
                        "num_prompts": len(sub),
                        "accuracy": float(correct / len(sub)),
                        "extractable_rate": float(extractable / len(sub)),
                        "format_compliance_rate": float(compliant / len(sub)),
                        "truncation_rate": float(truncated / len(sub)),
                    }
                )
        _write_csv(dirs["results"] / "standalone_student_accuracy.csv", standalone_rows)

    _rebuild_spec_csvs(dirs["results"], cost_model)

    # Update qualitative continuations
    qual_path = dirs["evidence"] / "qualitative_disagreements_and_continuations.json"
    qdoc = read_json(qual_path) if qual_path.exists() else {"prefix_level_disagreements": [], "matched_prompt_continuations": []}
    qdoc["matched_prompt_continuations"] = qual_continuations
    atomic_write_json(qual_path, qdoc)

    update_condition_status(
        dirs["results"] / "condition_status.json",
        f"spec_eval_{args.draft}",
        "completed",
        {"drafts": draft_names, "gammas": list(args.gammas)},
    )
    return {"cost_model": cost_model, "drafts": draft_names}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_spec_eval(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
