"""Task 1 Step 7: Q3 Mixed-Precision AWQ Quantization under Matched Packed-Byte Budgets.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task1.run_q3_mixed --budgets int4 int3 --group-size 128 --data-dir <DATA_DIR> --out-dir <T1_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from common.env import collect_env_metadata, resolve_model_revision
from common.io import (
    atomic_write_json,
    atomic_write_jsonl,
    ensure_task_dirs,
    load_local_source_revisions,
    read_json,
    resolve_smoke_out_dir,
    update_condition_status,
)
from common.seeding import DEFAULT_SEED, set_seed
from task1.benchmark import benchmark_prefill_and_decode
from task1.data import load_c4_sequences, load_hellaswag_val, load_ppl_windows
from task1.eval_hellaswag import evaluate_hellaswag
from task1.eval_ppl import evaluate_perplexity
from task1.model_utils import (
    DEFAULT_MODEL_ID,
    enumerate_eligible_projections,
    load_base_model_and_tokenizer,
    verify_model_structure,
)
from task1.pruning import restore_eligible_weights, snapshot_eligible_weights
from task1.quantization import (
    AWQ_ALPHA_GRID,
    Q3_ALLOWED_BITS,
    QUANT_CONVENTIONS,
    allocate_q3_mixed_precision,
    apply_awq_quantization,
    compute_quantized_model_storage,
    search_awq_scales_for_bits,
    unwrap_scaled_linears,
)
from task1.run_p3_nonuniform import _c4_dev_as_ppl_windows
from task1.run_pruning import _condition_is_complete
from task1.run_quantization import _save_or_load_mean_abs_x


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 1 Step 7 (Q3): Sensitivity-guided mixed-precision AWQ over "
            "{2, 3, 4, 8} bits matching Q2 uniform-INT4 and uniform-INT3 packed bytes."
        )
    )
    parser.add_argument(
        "--budgets",
        type=str,
        nargs="+",
        default=["int4", "int3"],
        choices=("int4", "int3"),
        help="Matched Q2 packed-byte budgets (default: int4 int3).",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup-reps", type=int, default=None)
    parser.add_argument("--timing-reps", type=int, default=None)
    parser.add_argument("--prefill-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--decode-tokens", type=int, default=None)
    parser.add_argument("--vocab-chunk-tokens", type=int, default=128)
    return parser


def _load_or_run_awq_search(
    model: torch.nn.Module,
    clean_weights: dict[str, torch.Tensor],
    mean_abs_x: dict[str, torch.Tensor],
    calib_records: list[dict[str, Any]],
    device: torch.device,
    bits: int,
    group_size: int,
    alpha_grid: tuple[float, ...],
    results_dir: Path,
) -> dict[str, Any]:
    cache_json = results_dir / f"awq_scale_search_int{bits}.json"
    if cache_json.exists():
        try:
            doc = read_json(cache_json)
            if (
                int(doc.get("bits", -1)) == bits
                and int(doc.get("group_size", -1)) == group_size
                and "best_alpha_by_name" in doc
                and "best_err_sq_by_name" in doc
            ):
                return doc
        except Exception:
            pass
    doc = search_awq_scales_for_bits(
        model=model,
        clean_weights=clean_weights,
        mean_abs_x=mean_abs_x,
        calib_records=calib_records,
        device=device,
        bits=bits,
        group_size=group_size,
        alpha_grid=alpha_grid,
    )
    atomic_write_json(cache_json, doc)
    return doc


def run_q3_mixed(
    args: argparse.Namespace,
    model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Select Q3 mixed-precision allocations on C4-dev and evaluate held-out."""
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    status_path = dirs["results"] / "condition_status.json"

    if args.smoke:
        calib_limit: int | None = 4
        dev_limit: int | None = 4
        ppl_limit: int | None = 4
        hs_full_limit: int | None = 16
        alpha_grid = (0.0, 0.5, 1.0)
        warmup = args.warmup_reps if args.warmup_reps is not None else 1
        reps = args.timing_reps if args.timing_reps is not None else 2
        prefill_lengths = list(args.prefill_lengths) if args.prefill_lengths else [256, 512]
        decode_prompt_len = 256
        decode_tokens = args.decode_tokens if args.decode_tokens is not None else 16
    else:
        calib_limit = None
        dev_limit = None
        ppl_limit = None
        hs_full_limit = None
        alpha_grid = AWQ_ALPHA_GRID
        warmup = args.warmup_reps if args.warmup_reps is not None else 2
        reps = args.timing_reps if args.timing_reps is not None else 5
        prefill_lengths = list(args.prefill_lengths) if args.prefill_lengths else [512, 2048]
        decode_prompt_len = 512
        decode_tokens = args.decode_tokens if args.decode_tokens is not None else 64

    if model is None or tokenizer is None:
        model, tokenizer, device, torch_dtype = load_base_model_and_tokenizer(
            model_id=args.model_id,
            dtype_name=args.dtype,
            device=args.device,
        )
    else:
        device = (
            torch.device(args.device)
            if args.device is not None
            else next(model.parameters()).device
        )
        torch_dtype = next(model.parameters()).dtype
        model.eval()

    model_revision = resolve_model_revision(model, args.model_id)
    verify_model_structure(model)
    projections = enumerate_eligible_projections(model)
    clean_weights = snapshot_eligible_weights(model)

    calib_path, calib_records = load_c4_sequences(
        args.data_dir, "c4_calibration", limit=calib_limit
    )
    dev_path, dev_records = load_c4_sequences(
        args.data_dir, "c4_development", limit=dev_limit
    )
    dev_windows = _c4_dev_as_ppl_windows(dev_records)
    wt2_path, wt2_windows = load_ppl_windows(args.data_dir, "wikitext2", limit=ppl_limit)
    c4_path, c4_windows = load_ppl_windows(args.data_dir, "c4_validation", limit=ppl_limit)
    hs_path, hs_records_all = load_hellaswag_val(args.data_dir, limit=hs_full_limit)

    atomic_write_json(
        dirs["configs"] / "q3_mixed_env_metadata.json",
        collect_env_metadata(
            data_dir=args.data_dir,
            model_revisions={args.model_id: model_revision},
            seed=args.seed,
            extra={
                "stage": "task1_q3_mixed",
                "smoke": bool(args.smoke),
                "dtype": str(torch_dtype),
                "device": str(device),
            },
        ),
    )

    cache_path = dirs["checkpoints"] / "c4_calib_mean_abs_x.pt"
    mean_abs_x, calib_act_sec, _ = _save_or_load_mean_abs_x(
        model, calib_records, device, cache_path, args.seed, args.smoke
    )

    # Collect AWQ scale search & calibration error profiles for all allowed bits {2, 3, 4, 8}
    search_by_bits: dict[int, dict[str, Any]] = {}
    t_probe0 = time.perf_counter()
    for b in Q3_ALLOWED_BITS:
        print(f"[Task 1 Q3] Loading/running AWQ sensitivity profile for b={b}...", flush=True)
        search_by_bits[b] = _load_or_run_awq_search(
            model=model,
            clean_weights=clean_weights,
            mean_abs_x=mean_abs_x,
            calib_records=calib_records,
            device=device,
            bits=b,
            group_size=args.group_size,
            alpha_grid=alpha_grid,
            results_dir=dirs["results"],
        )
    probe_time_sec = time.perf_counter() - t_probe0

    # Non-folded projections (all except down_proj, which folds into up_proj)
    non_folded_comp_names = {
        p.full_name for p in projections if p.proj_name != "down_proj"
    }

    # Exact Q2 uniform packed-byte budgets (using identical eligible set, metadata & comp bytes)
    q2_uniform_storage: dict[str, dict[str, Any]] = {}
    for b_name, b_val in (("int4", 4), ("int3", 3)):
        q2_uniform_storage[b_name] = compute_quantized_model_storage(
            model=model,
            bits_by_name={p.full_name: b_val for p in projections},
            group_size=args.group_size,
            non_folded_comp_names=non_folded_comp_names,
        )
    uncompressed_fp16_bytes = q2_uniform_storage["int4"]["uncompressed_fp16_bytes"]

    # Three sensitivity error metrics for greedy marginal-gain-per-byte allocation
    err_metrics: dict[str, dict[int, dict[str, float]]] = {
        "mse_per_byte": {
            b: {k: float(v) for k, v in search_by_bits[b]["best_err_sq_by_name"].items()}
            for b in Q3_ALLOWED_BITS
        },
        "rel_l2_per_byte": {
            b: {k: float(v) for k, v in search_by_bits[b]["best_rel_l2_by_name"].items()}
            for b in Q3_ALLOWED_BITS
        },
        "l2_per_byte": {
            b: {
                k: math.sqrt(max(float(v), 0.0))
                for k, v in search_by_bits[b]["best_err_sq_by_name"].items()
            }
            for b in Q3_ALLOWED_BITS
        },
    }

    verify_sample_ids = torch.tensor(
        [calib_records[0]["token_ids"][: min(64, len(calib_records[0]["token_ids"]))]],
        dtype=torch.long,
        device=device,
    )

    q3_summary_all: dict[str, Any] = {
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "model_id": args.model_id,
        "model_revision": model_revision,
        "group_size": int(args.group_size),
        "allowed_bits": list(Q3_ALLOWED_BITS),
        "unit_definition": "individual_eligible_projection",
        "quant_conventions": QUANT_CONVENTIONS,
        "selection_split": str(dev_path),
        "selection_criterion": "lowest C4-development perplexity under exact Q2 packed-byte budget (no test data accessed)",
        "calibration_split": str(calib_path),
        "local_source_revisions": load_local_source_revisions(args.data_dir),
        "by_budget": {},
    }

    all_results: dict[str, Any] = {}
    for budget_name in args.budgets:
        cond_id = f"q3_mixed_{budget_name}"
        res_path = dirs["results"] / f"condition_{cond_id}.json"

        if args.resume and _condition_is_complete(res_path, args.seed, args.smoke):
            print(f"[RESUME] Skipping completed condition {cond_id}.", flush=True)
            all_results[cond_id] = read_json(res_path)
            continue

        update_condition_status(
            status_path,
            cond_id,
            "pending",
            {"seed": int(args.seed), "smoke": bool(args.smoke), "budget": budget_name},
        )
        try:
            t_sel0 = time.perf_counter()
            budget_bytes = int(q2_uniform_storage[budget_name]["serialized_packed_total_bytes"])
            ref_bits = 4 if budget_name == "int4" else 3
            min_bits_candidates = (3, 2) if budget_name == "int4" else (2,)

            print(
                f"\n[Task 1 Q3] Selecting mixed-precision allocation for {budget_name} "
                f"(budget <= {budget_bytes:,} bytes) on C4-dev...",
                flush=True,
            )

            # Evaluate uniform Q2 on C4-dev as reference
            uniform_q2_bits = {p.full_name: ref_bits for p in projections}
            apply_awq_quantization(
                model=model,
                clean_weights=clean_weights,
                mean_abs_x=mean_abs_x,
                bits_by_name=uniform_q2_bits,
                alpha_by_name=search_by_bits[ref_bits]["best_alpha_by_name"],
                group_size=args.group_size,
            )
            uniform_dev, _ = evaluate_perplexity(
                model,
                dev_windows,
                device=device,
                suite_name=f"c4_dev_q2_{budget_name}",
                vocab_chunk_tokens=args.vocab_chunk_tokens,
            )

            candidates: list[dict[str, Any]] = []
            best_cand: dict[str, Any] | None = None
            best_alloc: dict[str, Any] | None = None

            for metric_name, err_dict in err_metrics.items():
                for min_b in min_bits_candidates:
                    alloc = allocate_q3_mixed_precision(
                        projections=projections,
                        errors_by_bits=err_dict,
                        budget_packed_bytes=budget_bytes,
                        uncompressed_fp16_bytes=uncompressed_fp16_bytes,
                        non_folded_comp_names=non_folded_comp_names,
                        group_size=args.group_size,
                        min_bits=min_b,
                    )
                    # Use each projection's optimal AWQ alpha for its assigned bit width
                    alpha_map = {
                        p.full_name: search_by_bits[alloc["bits_by_name"][p.full_name]][
                            "best_alpha_by_name"
                        ][p.full_name]
                        for p in projections
                    }
                    apply_awq_quantization(
                        model=model,
                        clean_weights=clean_weights,
                        mean_abs_x=mean_abs_x,
                        bits_by_name=alloc["bits_by_name"],
                        alpha_by_name=alpha_map,
                        group_size=args.group_size,
                    )
                    dev_summary, _ = evaluate_perplexity(
                        model,
                        dev_windows,
                        device=device,
                        suite_name=f"c4_dev_q3_{metric_name}_min{min_b}",
                        vocab_chunk_tokens=args.vocab_chunk_tokens,
                    )
                    cand_rec = {
                        "sensitivity_metric": metric_name,
                        "min_bits_floor": int(min_b),
                        "c4_dev_ppl": float(dev_summary["perplexity"]),
                        "c4_dev_mean_nll": float(dev_summary["mean_nll"]),
                        "achieved_packed_bytes": alloc["achieved_packed_bytes"],
                        "budget_packed_bytes": alloc["budget_packed_bytes"],
                        "bytes_under_budget": alloc["bytes_under_budget"],
                        "bit_histogram": alloc["bit_histogram"],
                    }
                    candidates.append(cand_rec)
                    print(
                        f"  [C4-dev] metric={metric_name:16s} min_b={min_b} -> "
                        f"dev_PPL={dev_summary['perplexity']:.4f} "
                        f"(Q2_dev={uniform_dev['perplexity']:.4f}) hist={alloc['bit_histogram']}",
                        flush=True,
                    )
                    if (
                        best_cand is None
                        or cand_rec["c4_dev_mean_nll"] < best_cand["c4_dev_mean_nll"]
                    ):
                        best_cand = cand_rec
                        best_alloc = alloc

            selection_sec = time.perf_counter() - t_sel0
            assert best_cand is not None and best_alloc is not None

            print(
                f"  [FROZEN Q3 {budget_name}] Selected {best_cand['sensitivity_metric']} "
                f"(min_b={best_cand['min_bits_floor']}, C4-dev PPL={best_cand['c4_dev_ppl']:.4f}, "
                f"bytes={best_alloc['achieved_packed_bytes']:,} <= {budget_bytes:,}). "
                f"Evaluating held-out splits...",
                flush=True,
            )

            # Apply frozen winning mixed-precision allocation cleanly and evaluate held-out
            t_eval0 = time.perf_counter()
            chosen_alpha_map = {
                p.full_name: search_by_bits[best_alloc["bits_by_name"][p.full_name]][
                    "best_alpha_by_name"
                ][p.full_name]
                for p in projections
            }
            q_apply = apply_awq_quantization(
                model=model,
                clean_weights=clean_weights,
                mean_abs_x=mean_abs_x,
                bits_by_name=best_alloc["bits_by_name"],
                alpha_by_name=chosen_alpha_map,
                group_size=args.group_size,
                verify_sample_ids=verify_sample_ids,
            )
            storage_report = compute_quantized_model_storage(
                model=model,
                bits_by_name=best_alloc["bits_by_name"],
                group_size=args.group_size,
                non_folded_comp_names=q_apply["non_folded_comp_names"],
            )
            if storage_report["serialized_packed_total_bytes"] > budget_bytes:
                raise AssertionError(
                    f"Q3 {budget_name} exceeded budget: "
                    f"{storage_report['serialized_packed_total_bytes']} > {budget_bytes}"
                )

            per_unit_assignment = [
                {
                    "layer_idx": p.layer_idx,
                    "full_name": p.full_name,
                    "proj_name": p.proj_name,
                    "numel": p.numel,
                    "assigned_bits": int(best_alloc["bits_by_name"][p.full_name]),
                    "awq_alpha": float(chosen_alpha_map[p.full_name]),
                    "calib_rel_l2_err_at_assigned_bits": float(
                        search_by_bits[best_alloc["bits_by_name"][p.full_name]][
                            "best_rel_l2_by_name"
                        ][p.full_name]
                    ),
                }
                for p in projections
            ]
            q3_summary_all["by_budget"][budget_name] = {
                "budget_name": budget_name,
                "budget_packed_bytes": int(budget_bytes),
                "uniform_q2_c4_dev_ppl": float(uniform_dev["perplexity"]),
                "selection_time_sec": float(selection_sec),
                "all_candidates_retained": candidates,
                "selected_candidate": best_cand,
                "achieved_packed_bytes": int(storage_report["serialized_packed_total_bytes"]),
                "bytes_under_budget": int(
                    budget_bytes - storage_report["serialized_packed_total_bytes"]
                ),
                "bit_histogram": best_alloc["bit_histogram"],
                "per_unit": per_unit_assignment,
            }

            wt2_summary, wt2_per_win = evaluate_perplexity(
                model,
                wt2_windows,
                device=device,
                suite_name="wikitext2",
                vocab_chunk_tokens=args.vocab_chunk_tokens,
            )
            atomic_write_jsonl(
                dirs["results"] / f"ppl_windows_wikitext2_{cond_id}.jsonl", wt2_per_win
            )

            c4_summary, c4_per_win = evaluate_perplexity(
                model,
                c4_windows,
                device=device,
                suite_name="c4_validation",
                vocab_chunk_tokens=args.vocab_chunk_tokens,
            )
            atomic_write_jsonl(
                dirs["results"] / f"ppl_windows_c4_{cond_id}.jsonl", c4_per_win
            )

            # Both Q3 INT4 and Q3 INT3 require full 2,000 HellaSwag IDs (PDF p. 3)
            hs_summary, hs_per_ex = evaluate_hellaswag(
                model, tokenizer, hs_records_all, device=device, diag_limit=512
            )
            atomic_write_jsonl(
                dirs["results"] / f"hellaswag_per_example_{cond_id}.jsonl", hs_per_ex
            )

            bench_report = benchmark_prefill_and_decode(
                model=model,
                device=device,
                prefill_lengths=prefill_lengths,
                decode_prompt_len=decode_prompt_len,
                decode_tokens=decode_tokens,
                warmup=warmup,
                reps=reps,
                seed=args.seed,
            )

            eval_sec = time.perf_counter() - t_eval0
            rel_errs = [u["calib_rel_l2_err_at_assigned_bits"] for u in per_unit_assignment]
            payload: dict[str, Any] = {
                "status": "completed",
                "condition": cond_id,
                "family": "quantization",
                "method": "q3_mixed",
                "budget_name": budget_name,
                "bits": float(storage_report["nominal_bits_per_eligible_weight"]),
                "group_size": int(args.group_size),
                "smoke": bool(args.smoke),
                "seed": int(args.seed),
                "model_id": args.model_id,
                "model_revision": model_revision,
                "dtype": str(torch_dtype),
                "device": str(device),
                "selected_metric": best_cand["sensitivity_metric"],
                "selected_min_bits_floor": best_cand["min_bits_floor"],
                "c4_dev_ppl": best_cand["c4_dev_ppl"],
                "uniform_q2_c4_dev_ppl": float(uniform_dev["perplexity"]),
                "bit_histogram": best_alloc["bit_histogram"],
                "matched_budget_check": {
                    "budget_packed_bytes": int(budget_bytes),
                    "achieved_packed_bytes": int(storage_report["serialized_packed_total_bytes"]),
                    "bytes_under_budget": int(
                        budget_bytes - storage_report["serialized_packed_total_bytes"]
                    ),
                },
                "fp_rescaling_equivalence": q_apply["fp_rescaling_equivalence"],
                "calibration_output_error": {
                    "eval_time_sec": float(probe_time_sec),
                    "mean_projection_rel_l2_error": float(sum(rel_errs) / max(len(rel_errs), 1)),
                    "max_projection_rel_l2_error": float(max(rel_errs)),
                    "pooled_rel_l2_error": float(sum(rel_errs) / max(len(rel_errs), 1)),
                },
                "storage": storage_report,
                "quality": {
                    "wikitext2": wt2_summary,
                    "c4_validation": c4_summary,
                    "hellaswag": hs_summary,
                },
                "benchmark": bench_report,
                "timing_sec": {
                    "calibration_act_sec": float(calib_act_sec),
                    "awq_probe_all_bits_sec": float(probe_time_sec),
                    "dev_selection_sec": float(selection_sec),
                    "quant_apply_sec": float(q_apply["quant_apply_time_sec"]),
                    "wikitext2_eval_sec": float(wt2_summary["eval_time_sec"]),
                    "c4_eval_sec": float(c4_summary["eval_time_sec"]),
                    "hellaswag_eval_sec": float(hs_summary["eval_time_sec"]),
                    "total_wall_sec": float(selection_sec + eval_sec),
                },
                "hardware_acceleration_note": (
                    "Fake-quantized FP16 reconstruction for numerical evaluation; "
                    "packed bytes including metadata and compensation scales are "
                    "computed separately. Not a hardware-accelerated low-bit kernel."
                ),
            }
            atomic_write_json(res_path, payload)
            update_condition_status(
                status_path,
                cond_id,
                "completed",
                {
                    "seed": int(args.seed),
                    "smoke": bool(args.smoke),
                    "budget": budget_name,
                    "packed_bytes": storage_report["serialized_packed_total_bytes"],
                    "wikitext2_ppl": wt2_summary["perplexity"],
                    "c4_ppl": c4_summary["perplexity"],
                    "hellaswag_acc_norm": hs_summary["acc_norm"],
                    "summary_path": str(res_path),
                },
            )
            print(
                f"  [{cond_id}] packed={storage_report['serialized_packed_total_bytes']/1e6:.2f}MB "
                f"<= {budget_bytes/1e6:.2f}MB | WT2 PPL={wt2_summary['perplexity']:.4f} | "
                f"C4 PPL={c4_summary['perplexity']:.4f} | HS(n={hs_summary['num_examples']}) "
                f"acc_norm={hs_summary['acc_norm']:.4f}",
                flush=True,
            )
            all_results[cond_id] = payload

        except Exception as exc:
            update_condition_status(
                status_path,
                cond_id,
                "failed",
                {
                    "seed": int(args.seed),
                    "smoke": bool(args.smoke),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    atomic_write_json(
        dirs["results"] / "q3_allocation_and_sensitivity.json",
        q3_summary_all,
    )
    unwrap_scaled_linears(model)
    restore_eligible_weights(model, clean_weights)
    return all_results


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_q3_mixed(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
