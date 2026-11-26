"""Task 1 Steps 5 & 6: Q1 (Uniform PTQ) and Q2 (AWQ-Style PTQ).

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task1.run_quantization --method q1_uniform --bits 8 4 3 --group-size 128 --data-dir <DATA_DIR> --out-dir <T1_OUT> --seed 6304
  python -m task1.run_quantization --method q2_awq --bits 8 4 3 --group-size 128 --data-dir <DATA_DIR> --out-dir <T1_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
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
    load_base_model_and_tokenizer,
    verify_model_structure,
)
from task1.pruning import (
    measure_calibration_layer_output_error,
    restore_eligible_weights,
    snapshot_eligible_weights,
)
from task1.quantization import (
    AWQ_ALPHA_GRID,
    QUANT_CONVENTIONS,
    apply_awq_quantization,
    apply_q1_uniform_quantization,
    collect_calibration_mean_abs_inputs,
    compute_quantized_model_storage,
    search_awq_scales_for_bits,
    unwrap_scaled_linears,
)
from task1.run_pruning import _condition_is_complete


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 1 Steps 5 & 6: Group-wise asymmetric weight-only PTQ "
            "(Q1 uniform and Q2 AWQ-style activation-aware scaling)."
        )
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=("q1_uniform", "q2_awq"),
        help="Quantization method: q1_uniform or q2_awq.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[8, 4, 3],
        help="Bit widths to evaluate (default: 8 4 3).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Quantization group size along input columns (default: 128).",
    )
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


def _save_or_load_mean_abs_x(
    model: torch.nn.Module,
    calib_records: list[dict[str, Any]],
    device: torch.device,
    cache_path: Path,
    seed: int,
    smoke: bool,
) -> tuple[dict[str, torch.Tensor], float, bool]:
    if cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                payload.get("seed") == seed
                and bool(payload.get("smoke")) == smoke
                and payload.get("num_sequences") == len(calib_records)
            ):
                return payload["mean_abs_x"], float(payload.get("calib_time_sec", 0.0)), True
        except Exception:
            pass
    mean_abs_x, calib_sec = collect_calibration_mean_abs_inputs(
        model, calib_records, device
    )
    tmp_path = cache_path.parent / f".{cache_path.name}.tmp"
    torch.save(
        {
            "seed": seed,
            "smoke": smoke,
            "num_sequences": len(calib_records),
            "calib_time_sec": calib_sec,
            "mean_abs_x": mean_abs_x,
        },
        tmp_path,
    )
    tmp_path.replace(cache_path)
    return mean_abs_x, calib_sec, False


def run_quantization(
    args: argparse.Namespace,
    model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Run Q1 uniform or Q2 AWQ-style weight-only PTQ."""
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    status_path = dirs["results"] / "condition_status.json"

    if args.smoke:
        calib_limit: int | None = 4
        ppl_limit: int | None = 4
        hs_full_limit: int | None = 16
        hs_diag_limit = 8
        alpha_grid = (0.0, 0.5, 1.0)
        warmup = args.warmup_reps if args.warmup_reps is not None else 1
        reps = args.timing_reps if args.timing_reps is not None else 2
        prefill_lengths = list(args.prefill_lengths) if args.prefill_lengths else [256, 512]
        decode_prompt_len = 256
        decode_tokens = args.decode_tokens if args.decode_tokens is not None else 16
    else:
        calib_limit = None
        ppl_limit = None
        hs_full_limit = None
        hs_diag_limit = 512
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
    clean_weights = snapshot_eligible_weights(model)

    calib_path, calib_records = load_c4_sequences(
        args.data_dir, "c4_calibration", limit=calib_limit
    )
    wt2_path, wt2_windows = load_ppl_windows(args.data_dir, "wikitext2", limit=ppl_limit)
    c4_path, c4_windows = load_ppl_windows(args.data_dir, "c4_validation", limit=ppl_limit)
    hs_path, hs_records_all = load_hellaswag_val(args.data_dir, limit=hs_full_limit)

    atomic_write_json(
        dirs["configs"] / f"{args.method}_env_metadata.json",
        collect_env_metadata(
            data_dir=args.data_dir,
            model_revisions={args.model_id: model_revision},
            seed=args.seed,
            extra={
                "stage": f"task1_{args.method}",
                "smoke": bool(args.smoke),
                "dtype": str(torch_dtype),
                "device": str(device),
            },
        ),
    )
    atomic_write_json(
        dirs["configs"] / f"{args.method}_config.json",
        {
            "method": args.method,
            "bits": [int(b) for b in args.bits],
            "group_size": int(args.group_size),
            "alpha_grid": list(alpha_grid),
            "quant_conventions": QUANT_CONVENTIONS,
            "seed": int(args.seed),
            "smoke": bool(args.smoke),
            "model_id": args.model_id,
            "model_revision": model_revision,
            "manifests": {
                "c4_calibration": str(calib_path),
                "wikitext2": str(wt2_path),
                "c4_validation": str(c4_path),
                "hellaswag": str(hs_path),
            },
            "local_source_revisions": load_local_source_revisions(args.data_dir),
        },
    )

    mean_abs_x: dict[str, torch.Tensor] | None = None
    calib_act_sec = 0.0
    if args.method == "q2_awq":
        cache_path = dirs["checkpoints"] / "c4_calib_mean_abs_x.pt"
        print(
            f"[Task 1 q2_awq] Collecting/loading AWQ calibration mean(|X|) "
            f"({len(calib_records)} sequences)...",
            flush=True,
        )
        mean_abs_x, calib_act_sec, cached = _save_or_load_mean_abs_x(
            model, calib_records, device, cache_path, args.seed, args.smoke
        )
        print(
            f"  AWQ calibration activations ready (cached={cached}, time={calib_act_sec:.2f}s).",
            flush=True,
        )

    verify_sample_ids = torch.tensor(
        [calib_records[0]["token_ids"][: min(64, len(calib_records[0]["token_ids"]))]],
        dtype=torch.long,
        device=device,
    )

    all_results: dict[str, Any] = {}
    for b in args.bits:
        cond_id = f"{args.method}_int{b}"
        res_path = dirs["results"] / f"condition_{cond_id}.json"

        if args.resume and _condition_is_complete(res_path, args.seed, args.smoke):
            print(f"[RESUME] Skipping completed condition {cond_id}.", flush=True)
            all_results[cond_id] = read_json(res_path)
            continue

        update_condition_status(
            status_path,
            cond_id,
            "pending",
            {"seed": int(args.seed), "smoke": bool(args.smoke), "bits": int(b)},
        )
        try:
            t_cond0 = time.perf_counter()
            unwrap_scaled_linears(model)
            restore_eligible_weights(model, clean_weights)

            # PDF p. 3: full 2,000 HellaSwag IDs for all INT4 and INT3 variants; 512 diag for INT8
            run_full_hs = b in (4, 3)
            print(
                f"\n[Task 1 Quant] Running {cond_id} (b={b}, g={args.group_size}, full_hs={run_full_hs})...",
                flush=True,
            )

            awq_search_report: dict[str, Any] | None = None
            if args.method == "q1_uniform":
                q_apply = apply_q1_uniform_quantization(
                    model, bits=b, group_size=args.group_size
                )
                calib_err = measure_calibration_layer_output_error(
                    model, clean_weights, calib_records, device
                )
                search_time_sec = 0.0
                fp_equiv = None
                folded_count = 0
            else:
                assert mean_abs_x is not None
                awq_search_report = search_awq_scales_for_bits(
                    model=model,
                    clean_weights=clean_weights,
                    mean_abs_x=mean_abs_x,
                    calib_records=calib_records,
                    device=device,
                    bits=b,
                    group_size=args.group_size,
                    alpha_grid=alpha_grid,
                )
                search_time_sec = awq_search_report["search_time_sec"]
                atomic_write_json(
                    dirs["results"] / f"awq_scale_search_int{b}.json",
                    awq_search_report,
                )
                bits_map = {k: int(b) for k in awq_search_report["best_alpha_by_name"]}
                q_apply = apply_awq_quantization(
                    model=model,
                    clean_weights=clean_weights,
                    mean_abs_x=mean_abs_x,
                    bits_by_name=bits_map,
                    alpha_by_name=awq_search_report["best_alpha_by_name"],
                    group_size=args.group_size,
                    verify_sample_ids=verify_sample_ids,
                )
                fp_equiv = q_apply["fp_rescaling_equivalence"]
                folded_count = len(q_apply["folded_projections"])
                calib_err = {
                    "eval_time_sec": search_time_sec,
                    "mean_projection_rel_l2_error": awq_search_report["mean_rel_l2_error"],
                    "max_projection_rel_l2_error": max(
                        awq_search_report["best_rel_l2_by_name"].values()
                    ),
                    "pooled_rel_l2_error": float(
                        (
                            sum(awq_search_report["best_err_sq_by_name"].values())
                            / max(
                                sum(p["ref_sq"] for p in awq_search_report["per_projection"]),
                                1e-12,
                            )
                        )
                        ** 0.5
                    ),
                }

            storage_report = compute_quantized_model_storage(
                model=model,
                bits_by_name=q_apply["bits_by_name"],
                group_size=args.group_size,
                non_folded_comp_names=q_apply["non_folded_comp_names"],
            )

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

            hs_subset = hs_records_all if run_full_hs else hs_records_all[:hs_diag_limit]
            hs_summary, hs_per_ex = evaluate_hellaswag(
                model, tokenizer, hs_subset, device=device, diag_limit=512
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

            total_sec = time.perf_counter() - t_cond0
            payload: dict[str, Any] = {
                "status": "completed",
                "condition": cond_id,
                "family": "quantization",
                "method": args.method,
                "bits": int(b),
                "group_size": int(args.group_size),
                "smoke": bool(args.smoke),
                "seed": int(args.seed),
                "model_id": args.model_id,
                "model_revision": model_revision,
                "dtype": str(torch_dtype),
                "device": str(device),
                "folded_projection_count": int(folded_count),
                "non_folded_comp_count": len(q_apply["non_folded_comp_names"]),
                "fp_rescaling_equivalence": fp_equiv,
                "calibration_output_error": {
                    "eval_time_sec": float(calib_err["eval_time_sec"]),
                    "mean_projection_rel_l2_error": float(
                        calib_err["mean_projection_rel_l2_error"]
                    ),
                    "max_projection_rel_l2_error": float(
                        calib_err["max_projection_rel_l2_error"]
                    ),
                    "pooled_rel_l2_error": float(calib_err["pooled_rel_l2_error"]),
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
                    "awq_scale_search_sec": float(search_time_sec),
                    "quant_apply_sec": float(q_apply["quant_apply_time_sec"]),
                    "wikitext2_eval_sec": float(wt2_summary["eval_time_sec"]),
                    "c4_eval_sec": float(c4_summary["eval_time_sec"]),
                    "hellaswag_eval_sec": float(hs_summary["eval_time_sec"]),
                    "total_wall_sec": float(total_sec),
                },
                "hardware_acceleration_note": (
                    "Fake-quantized FP16 reconstruction for numerical evaluation; "
                    "packed bytes including metadata and compensation scales are "
                    "computed separately. Not a hardware-accelerated low-bit kernel."
                ),
                "awq_scope_note": (
                    "Reproduces AWQ activation-dependent per-channel scale search "
                    "(grid over alpha) and algebraic down_proj->up_proj folding with "
                    "explicit FP16 compensation on non-foldable layers; does not "
                    "implement fused INT4/INT3 CUDA GEMM kernels."
                    if args.method == "q2_awq"
                    else None
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
                    "bits": int(b),
                    "packed_bytes": storage_report["serialized_packed_total_bytes"],
                    "wikitext2_ppl": wt2_summary["perplexity"],
                    "c4_ppl": c4_summary["perplexity"],
                    "hellaswag_acc_norm": hs_summary["acc_norm"],
                    "summary_path": str(res_path),
                },
            )
            print(
                f"  [{cond_id}] packed={storage_report['serialized_packed_total_bytes']/1e6:.2f}MB "
                f"({storage_report['effective_bits_per_eligible_weight_incl_meta']:.2f} b/w) | "
                f"WT2 PPL={wt2_summary['perplexity']:.4f} | C4 PPL={c4_summary['perplexity']:.4f} | "
                f"HS(n={hs_summary['num_examples']}) acc_norm={hs_summary['acc_norm']:.4f} | {total_sec:.1f}s",
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

    unwrap_scaled_linears(model)
    restore_eligible_weights(model, clean_weights)
    return all_results


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_quantization(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
