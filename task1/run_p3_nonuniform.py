"""Task 1 Step 4: P3 Nonuniform Pruning (Sensitivity Probe + Constrained Allocation).

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task1.run_p3_nonuniform --sparsities 0.50 0.75 --data-dir <DATA_DIR> --out-dir <T1_OUT> --seed 6304
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
    enumerate_eligible_projections,
    load_base_model_and_tokenizer,
    verify_model_structure,
)
from task1.pruning import (
    P3_MAX_SPARSITY,
    P3_MIN_SPARSITY,
    allocate_nonuniform_sparsities,
    apply_nonuniform_wanda_pruning,
    apply_uniform_pruning,
    measure_calibration_layer_output_error,
    restore_eligible_weights,
    snapshot_eligible_weights,
    summarize_wanda_distributions,
)
from task1.run_pruning import _condition_is_complete, _save_or_load_calib_norms
from task1.storage import compute_fp16_and_sparse_storage

DEFAULT_LAMBDA_GRID: tuple[float, ...] = (0.04, 0.08, 0.12, 0.16)
DEFAULT_PROBES: tuple[str, ...] = ("owl_lod_m5", "calib_rel_l2_err")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 1 Step 4 (P3): Nonuniform Wanda pruning with dev-selected "
            "sensitivity measure & constrained allocation (0.10 <= s_l <= 0.85)."
        )
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.50, 0.75],
        help="Target global sparsities (default: 0.50 0.75).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to prepared atml-pa4-datasets directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output root directory (e.g. /content/drive/MyDrive/atml-pa4/task1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced smoke test and write outputs to a separate smoke directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed P3 conditions with verified result files.",
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dtype", type=str, default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup-reps", type=int, default=None)
    parser.add_argument("--timing-reps", type=int, default=None)
    parser.add_argument("--prefill-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--decode-tokens", type=int, default=None)
    parser.add_argument("--vocab-chunk-tokens", type=int, default=128)
    return parser


def _c4_dev_as_ppl_windows(dev_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert C4 development sequences into non-overlapping PPL window dicts."""
    windows: list[dict[str, Any]] = []
    for rec in dev_records:
        ids = rec["token_ids"]
        windows.append(
            {
                "token_ids": ids,
                "score_target_offsets": [1, len(ids)],
                "scored_tokens": len(ids) - 1,
                **({"source_row": rec["source_row"]} if "source_row" in rec else {}),
            }
        )
    return windows


def run_p3_nonuniform(
    args: argparse.Namespace,
    model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Select P3 sensitivity/allocation strictly on C4-dev, then evaluate held-out."""
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    status_path = dirs["results"] / "condition_status.json"

    if args.smoke:
        calib_limit: int | None = 4
        dev_limit: int | None = 4
        ppl_limit: int | None = 4
        hs_full_limit: int | None = 16
        hs_diag_limit = 8
        lambda_grid = (0.06, 0.12)
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
        hs_diag_limit = 512
        lambda_grid = DEFAULT_LAMBDA_GRID
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
        dirs["configs"] / "p3_nonuniform_env_metadata.json",
        collect_env_metadata(
            data_dir=args.data_dir,
            model_revisions={args.model_id: model_revision},
            seed=args.seed,
            extra={
                "stage": "task1_p3_nonuniform",
                "smoke": bool(args.smoke),
                "dtype": str(torch_dtype),
                "device": str(device),
            },
        ),
    )

    # 1. Calibration activation norms & OWL sensitivity (calibration set only)
    norm_cache = dirs["checkpoints"] / "c4_calib_act_norms.pt"
    act_norms, calib_time_sec, _ = _save_or_load_calib_norms(
        model, calib_records, device, norm_cache, args.seed, args.smoke
    )
    wanda_dists = summarize_wanda_distributions(model, act_norms)
    owl_sensitivities = {entry["full_name"]: float(entry["lod_m5"]) for entry in wanda_dists}

    allocation_summary_all: dict[str, Any] = {
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "model_id": args.model_id,
        "model_revision": model_revision,
        "unit_definition": "individual_eligible_projection",
        "bounds": {"s_min": P3_MIN_SPARSITY, "s_max": P3_MAX_SPARSITY},
        "selection_split": str(dev_path),
        "selection_criterion": "lowest C4-development perplexity (32-sequence locked dev manifest; no test data accessed)",
        "calibration_split": str(calib_path),
        "local_source_revisions": load_local_source_revisions(args.data_dir),
        "probes_evaluated": list(DEFAULT_PROBES),
        "lambda_grid": list(lambda_grid),
        "by_sparsity": {},
    }

    all_results: dict[str, Any] = {}
    for sparsity in args.sparsities:
        pct = int(round(sparsity * 100))
        cond_id = f"p3_nonuniform_s{pct}"
        res_path = dirs["results"] / f"condition_{cond_id}.json"

        if args.resume and _condition_is_complete(res_path, args.seed, args.smoke):
            print(f"[RESUME] Skipping completed condition {cond_id}.", flush=True)
            all_results[cond_id] = read_json(res_path)
            continue

        update_condition_status(
            status_path,
            cond_id,
            "pending",
            {"seed": int(args.seed), "smoke": bool(args.smoke), "sparsity": float(sparsity)},
        )

        try:
            t_sel0 = time.perf_counter()
            print(
                f"\n[Task 1 P3] Running calibration sensitivity & C4-dev selection for s={sparsity:.2f}...",
                flush=True,
            )

            # Uniform Wanda baseline on C4-dev + calibration relative output-error probe
            restore_eligible_weights(model, clean_weights)
            apply_uniform_pruning(model, "p2_wanda", sparsity=sparsity, act_norms=act_norms)
            uniform_dev_ppl, _ = evaluate_perplexity(
                model,
                dev_windows,
                device=device,
                suite_name="c4_dev_uniform",
                vocab_chunk_tokens=args.vocab_chunk_tokens,
            )
            calib_err_probe = measure_calibration_layer_output_error(
                model, clean_weights, calib_records, device
            )
            err_sensitivities = {
                entry["full_name"]: float(entry["rel_l2_error"])
                for entry in calib_err_probe["per_projection"]
            }

            probe_map = {
                "owl_lod_m5": owl_sensitivities,
                "calib_rel_l2_err": err_sensitivities,
            }

            candidates: list[dict[str, Any]] = []
            best_candidate: dict[str, Any] | None = None
            best_allocation: dict[str, Any] | None = None

            for probe_name, sens_dict in probe_map.items():
                for lam in lambda_grid:
                    alloc = allocate_nonuniform_sparsities(
                        projections=projections,
                        sensitivities=sens_dict,
                        target_sparsity=float(sparsity),
                        lam=float(lam),
                        s_min=P3_MIN_SPARSITY,
                        s_max=P3_MAX_SPARSITY,
                    )
                    restore_eligible_weights(model, clean_weights)
                    apply_nonuniform_wanda_pruning(
                        model, act_norms, alloc["zero_counts_by_name"]
                    )
                    dev_summary, _ = evaluate_perplexity(
                        model,
                        dev_windows,
                        device=device,
                        suite_name=f"c4_dev_{probe_name}_lam{lam}",
                        vocab_chunk_tokens=args.vocab_chunk_tokens,
                    )
                    cand_record = {
                        "probe": probe_name,
                        "lambda": float(lam),
                        "c4_dev_ppl": float(dev_summary["perplexity"]),
                        "c4_dev_mean_nll": float(dev_summary["mean_nll"]),
                        "achieved_global_sparsity": alloc["achieved_global_sparsity"],
                        "zero_count_diff": alloc["zero_count_diff"],
                        "min_unit_sparsity": alloc["min_unit_sparsity"],
                        "max_unit_sparsity": alloc["max_unit_sparsity"],
                    }
                    candidates.append(cand_record)
                    print(
                        f"  [C4-dev] probe={probe_name:16s} lam={lam:.2f} -> "
                        f"dev_PPL={dev_summary['perplexity']:.4f} "
                        f"(uniform_dev_PPL={uniform_dev_ppl['perplexity']:.4f})",
                        flush=True,
                    )
                    if (
                        best_candidate is None
                        or cand_record["c4_dev_mean_nll"] < best_candidate["c4_dev_mean_nll"]
                    ):
                        best_candidate = cand_record
                        best_allocation = alloc

            selection_time_sec = time.perf_counter() - t_sel0
            assert best_candidate is not None and best_allocation is not None

            print(
                f"  [FROZEN P3 s={sparsity:.2f}] Selected probe={best_candidate['probe']}, "
                f"lam={best_candidate['lambda']:.2f} (C4-dev PPL={best_candidate['c4_dev_ppl']:.4f}, "
                f"zero_diff={best_allocation['zero_count_diff']}). Evaluating held-out splits...",
                flush=True,
            )

            allocation_summary_all["by_sparsity"][f"s{pct}"] = {
                "target_sparsity": float(sparsity),
                "uniform_wanda_c4_dev_ppl": float(uniform_dev_ppl["perplexity"]),
                "selection_time_sec": float(selection_time_sec),
                "all_candidates_retained": candidates,
                "selected_candidate": best_candidate,
                "selected_allocation": {
                    k: v
                    for k, v in best_allocation.items()
                    if k != "zero_counts_by_name"
                },
            }

            # Apply frozen winning allocation to a clean copy of FP16 weights and evaluate held-out
            t_eval0 = time.perf_counter()
            restore_eligible_weights(model, clean_weights)
            apply_report = apply_nonuniform_wanda_pruning(
                model, act_norms, best_allocation["zero_counts_by_name"]
            )
            storage_report = compute_fp16_and_sparse_storage(model)
            calib_err = measure_calibration_layer_output_error(
                model, clean_weights, calib_records, device
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

            run_full_hs = abs(sparsity - 0.50) < 1e-6
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

            eval_total_sec = time.perf_counter() - t_eval0
            payload: dict[str, Any] = {
                "status": "completed",
                "condition": cond_id,
                "family": "pruning",
                "method": "p3_nonuniform",
                "structured_2to4": False,
                "target_sparsity": float(sparsity),
                "smoke": bool(args.smoke),
                "seed": int(args.seed),
                "model_id": args.model_id,
                "model_revision": model_revision,
                "dtype": str(torch_dtype),
                "device": str(device),
                "selected_probe": best_candidate["probe"],
                "selected_lambda": best_candidate["lambda"],
                "c4_dev_ppl": best_candidate["c4_dev_ppl"],
                "uniform_wanda_c4_dev_ppl": float(uniform_dev_ppl["perplexity"]),
                "matched_budget_check": {
                    "target_total_zeros": best_allocation["target_total_zeros"],
                    "achieved_total_zeros": best_allocation["achieved_total_zeros"],
                    "zero_count_diff": best_allocation["zero_count_diff"],
                    "min_unit_sparsity": best_allocation["min_unit_sparsity"],
                    "max_unit_sparsity": best_allocation["max_unit_sparsity"],
                },
                "pruning": apply_report,
                "calibration_output_error": {
                    "eval_time_sec": calib_err["eval_time_sec"],
                    "mean_projection_rel_l2_error": calib_err["mean_projection_rel_l2_error"],
                    "max_projection_rel_l2_error": calib_err["max_projection_rel_l2_error"],
                    "pooled_rel_l2_error": calib_err["pooled_rel_l2_error"],
                },
                "storage": storage_report,
                "quality": {
                    "wikitext2": wt2_summary,
                    "c4_validation": c4_summary,
                    "hellaswag": hs_summary,
                },
                "benchmark": bench_report,
                "timing_sec": {
                    "calibration_sec": float(calib_time_sec),
                    "dev_selection_sec": float(selection_time_sec),
                    "pruning_apply_sec": float(apply_report["pruning_apply_time_sec"]),
                    "calib_error_eval_sec": float(calib_err["eval_time_sec"]),
                    "wikitext2_eval_sec": float(wt2_summary["eval_time_sec"]),
                    "c4_eval_sec": float(c4_summary["eval_time_sec"]),
                    "hellaswag_eval_sec": float(hs_summary["eval_time_sec"]),
                    "total_wall_sec": float(selection_time_sec + eval_total_sec),
                },
                "hardware_acceleration_note": (
                    "Dense FP16 execution with zeroed weights; no sparse kernel "
                    "acceleration is claimed."
                ),
            }
            atomic_write_json(res_path, payload)
            atomic_write_json(
                dirs["results"] / f"calib_layer_error_{cond_id}.json", calib_err
            )
            update_condition_status(
                status_path,
                cond_id,
                "completed",
                {
                    "seed": int(args.seed),
                    "smoke": bool(args.smoke),
                    "selected_probe": best_candidate["probe"],
                    "selected_lambda": best_candidate["lambda"],
                    "achieved_sparsity": storage_report["achieved_sparsity"],
                    "wikitext2_ppl": wt2_summary["perplexity"],
                    "c4_ppl": c4_summary["perplexity"],
                    "hellaswag_acc_norm": hs_summary["acc_norm"],
                    "summary_path": str(res_path),
                },
            )
            print(
                f"  [{cond_id}] s={storage_report['achieved_sparsity']:.4f} | "
                f"WT2 PPL={wt2_summary['perplexity']:.4f} | C4 PPL={c4_summary['perplexity']:.4f} | "
                f"HS(n={hs_summary['num_examples']}) acc_norm={hs_summary['acc_norm']:.4f}",
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
        dirs["results"] / "p3_allocation_and_sensitivity.json",
        allocation_summary_all,
    )
    restore_eligible_weights(model, clean_weights)
    return all_results


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_p3_nonuniform(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
