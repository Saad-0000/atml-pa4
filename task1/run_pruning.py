"""Task 1 Steps 2, 3 & 8: Uniform Magnitude (P1), Wanda (P2), and 2:4 Diagnostic.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task1.run_pruning --method p1_magnitude --sparsities 0.25 0.50 0.75 --data-dir <DATA_DIR> --out-dir <T1_OUT> --seed 6304
  python -m task1.run_pruning --method p2_wanda --sparsities 0.25 0.50 0.75 --run-2to4-at-50 --data-dir <DATA_DIR> --out-dir <T1_OUT> --seed 6304
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
    apply_uniform_pruning,
    collect_calibration_input_norms,
    measure_calibration_layer_output_error,
    restore_eligible_weights,
    snapshot_eligible_weights,
    summarize_wanda_distributions,
)
from task1.storage import compute_fp16_and_sparse_storage


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 1 Steps 2, 3 & 8: Run uniform per-row magnitude pruning (P1), "
            "Wanda pruning (P2), and optional 2:4 structured sparsity diagnostic at 50%."
        )
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=("p1_magnitude", "p2_wanda"),
        help="Pruning method: p1_magnitude or p2_wanda.",
    )
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
        help="Target global sparsities (default: 0.25 0.50 0.75).",
    )
    parser.add_argument(
        "--run-2to4-at-50",
        action="store_true",
        help="Also run Step 8 2:4 structured sparsity diagnostic at 50%% Wanda sparsity.",
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
        help="Skip already-completed conditions with verified result files.",
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


def _condition_id(method: str, sparsity: float, structured_2to4: bool = False) -> str:
    pct = int(round(sparsity * 100))
    if structured_2to4:
        return f"{method}_2to4_s{pct}"
    return f"{method}_s{pct}"


def _condition_is_complete(result_path: Path, expected_seed: int, expected_smoke: bool) -> bool:
    if not result_path.exists() or result_path.stat().st_size == 0:
        return False
    try:
        data = read_json(result_path)
        return (
            data.get("status") == "completed"
            and data.get("seed") == expected_seed
            and bool(data.get("smoke")) == expected_smoke
        )
    except Exception:
        return False


def _save_or_load_calib_norms(
    model: torch.nn.Module,
    calib_records: list[dict[str, Any]],
    device: torch.device,
    cache_path: Path,
    seed: int,
    smoke: bool,
) -> tuple[dict[str, torch.Tensor], float, bool]:
    """Load cached calibration activation norms if valid, else collect and cache."""
    if cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                payload.get("seed") == seed
                and bool(payload.get("smoke")) == smoke
                and payload.get("num_sequences") == len(calib_records)
            ):
                return payload["act_norms"], float(payload.get("calib_time_sec", 0.0)), True
        except Exception:
            pass

    act_norms, calib_sec = collect_calibration_input_norms(model, calib_records, device)
    tmp_path = cache_path.parent / f".{cache_path.name}.tmp"
    torch.save(
        {
            "seed": seed,
            "smoke": smoke,
            "num_sequences": len(calib_records),
            "calib_time_sec": calib_sec,
            "act_norms": act_norms,
        },
        tmp_path,
    )
    tmp_path.replace(cache_path)
    return act_norms, calib_sec, False


def run_pruning(
    args: argparse.Namespace,
    model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Run P1/P2 uniform pruning and optional 2:4 diagnostic."""
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    status_path = dirs["results"] / "condition_status.json"

    if args.smoke:
        calib_limit: int | None = 4
        ppl_limit: int | None = 4
        hs_full_limit: int | None = 16
        hs_diag_limit = 8
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
        warmup = args.warmup_reps if args.warmup_reps is not None else 2
        reps = args.timing_reps if args.timing_reps is not None else 5
        prefill_lengths = list(args.prefill_lengths) if args.prefill_lengths else [512, 2048]
        decode_prompt_len = 512
        decode_tokens = args.decode_tokens if args.decode_tokens is not None else 64

    # Build list of conditions: (sparsity, structured_2to4, run_full_hellaswag)
    conditions: list[tuple[float, bool, bool]] = []
    for s in args.sparsities:
        is_50 = abs(s - 0.50) < 1e-6
        conditions.append((float(s), False, is_50))
    if args.run_2to4_at_50:
        if args.method != "p2_wanda":
            raise ValueError("--run-2to4-at-50 is only specified for p2_wanda (PA4 Step 8).")
        conditions.append((0.50, True, False))

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

    env_meta = collect_env_metadata(
        data_dir=args.data_dir,
        model_revisions={args.model_id: model_revision},
        seed=args.seed,
        extra={
            "stage": f"task1_{args.method}",
            "smoke": bool(args.smoke),
            "dtype": str(torch_dtype),
            "device": str(device),
        },
    )
    atomic_write_json(dirs["configs"] / f"{args.method}_env_metadata.json", env_meta)
    atomic_write_json(
        dirs["configs"] / f"{args.method}_config.json",
        {
            "method": args.method,
            "sparsities": [float(s) for s in args.sparsities],
            "run_2to4_at_50": bool(args.run_2to4_at_50),
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

    act_norms: dict[str, torch.Tensor] | None = None
    calib_time_sec = 0.0
    if args.method == "p2_wanda":
        norm_cache = dirs["checkpoints"] / "c4_calib_act_norms.pt"
        print(
            f"[Task 1 {args.method}] Collecting/loading Wanda calibration norms "
            f"({len(calib_records)} sequences)...",
            flush=True,
        )
        act_norms, calib_time_sec, loaded_from_cache = _save_or_load_calib_norms(
            model, calib_records, device, norm_cache, args.seed, args.smoke
        )
        print(
            f"  Calibration norms ready (cached={loaded_from_cache}, "
            f"calib_time_sec={calib_time_sec:.2f}s).",
            flush=True,
        )
        dist_path = dirs["results"] / "wanda_score_distributions.json"
        if not dist_path.exists():
            wanda_dists = summarize_wanda_distributions(model, act_norms)
            atomic_write_json(
                dist_path,
                {
                    "seed": int(args.seed),
                    "smoke": bool(args.smoke),
                    "num_calibration_sequences": len(calib_records),
                    "calib_time_sec": float(calib_time_sec),
                    "projections": wanda_dists,
                },
            )

    all_results: dict[str, Any] = {}
    for sparsity, is_2to4, run_full_hs in conditions:
        cond_id = _condition_id(args.method, sparsity, structured_2to4=is_2to4)
        res_path = dirs["results"] / f"condition_{cond_id}.json"

        if args.resume and _condition_is_complete(res_path, args.seed, args.smoke):
            print(f"[RESUME] Skipping completed condition {cond_id}.", flush=True)
            all_results[cond_id] = read_json(res_path)
            continue

        update_condition_status(
            status_path,
            cond_id,
            "pending",
            {"seed": int(args.seed), "smoke": bool(args.smoke), "sparsity": sparsity},
        )
        try:
            t_cond0 = time.perf_counter()
            restore_eligible_weights(model, clean_weights)

            print(
                f"\n[Task 1 Pruning] Running {cond_id} "
                f"(target_s={sparsity:.2f}, 2to4={is_2to4}, full_hs={run_full_hs})...",
                flush=True,
            )
            prune_report = apply_uniform_pruning(
                model=model,
                method=args.method,
                sparsity=sparsity,
                act_norms=act_norms,
                structured_2to4=is_2to4,
            )
            storage_report = compute_fp16_and_sparse_storage(model)

            # Local calibration output error
            calib_err = measure_calibration_layer_output_error(
                model, clean_weights, calib_records, device
            )

            # Quality evaluations
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
                "family": "pruning",
                "method": args.method,
                "structured_2to4": bool(is_2to4),
                "target_sparsity": float(sparsity),
                "smoke": bool(args.smoke),
                "seed": int(args.seed),
                "model_id": args.model_id,
                "model_revision": model_revision,
                "dtype": str(torch_dtype),
                "device": str(device),
                "pruning": {
                    k: v for k, v in prune_report.items() if k != "per_projection"
                },
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
                    "pruning_apply_sec": float(prune_report["pruning_apply_time_sec"]),
                    "calib_error_eval_sec": float(calib_err["eval_time_sec"]),
                    "wikitext2_eval_sec": float(wt2_summary["eval_time_sec"]),
                    "c4_eval_sec": float(c4_summary["eval_time_sec"]),
                    "hellaswag_eval_sec": float(hs_summary["eval_time_sec"]),
                    "total_wall_sec": float(total_sec),
                },
                "hardware_acceleration_note": (
                    "Dense FP16 execution with zeroed weights; no sparse or 2:4 "
                    "sparse-tensor-core kernel acceleration is claimed."
                ),
            }
            atomic_write_json(res_path, payload)
            atomic_write_json(
                dirs["results"] / f"calib_layer_error_{cond_id}.json",
                calib_err,
            )
            update_condition_status(
                status_path,
                cond_id,
                "completed",
                {
                    "seed": int(args.seed),
                    "smoke": bool(args.smoke),
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
                f"HS(n={hs_summary['num_examples']}) acc_norm={hs_summary['acc_norm']:.4f} | "
                f"calib_rel_err={calib_err['mean_projection_rel_l2_error']:.4f} | {total_sec:.1f}s",
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

    # Step 8 dedicated comparison file: 2:4 Wanda vs unstructured 50% Wanda
    if args.run_2to4_at_50 and "p2_wanda_2to4_s50" in all_results:
        unstruct = all_results.get("p2_wanda_s50")
        struct = all_results["p2_wanda_2to4_s50"]
        diag_payload = {
            "status": "completed",
            "seed": int(args.seed),
            "smoke": bool(args.smoke),
            "note": (
                "Quality and representation comparison at matched 50% eligible-weight "
                "sparsity. No T4/L4 sparse-tensor-core acceleration is claimed."
            ),
            "unstructured_50": unstruct,
            "structured_2to4_50": struct,
            "paired_comparison_summary": (
                {
                    "sparsity_unstructured": unstruct["storage"]["achieved_sparsity"],
                    "sparsity_2to4": struct["storage"]["achieved_sparsity"],
                    "wt2_ppl_delta_2to4_minus_unstruct": (
                        struct["quality"]["wikitext2"]["perplexity"]
                        - unstruct["quality"]["wikitext2"]["perplexity"]
                    ),
                    "c4_ppl_delta_2to4_minus_unstruct": (
                        struct["quality"]["c4_validation"]["perplexity"]
                        - unstruct["quality"]["c4_validation"]["perplexity"]
                    ),
                    "hellaswag_diag_acc_norm_delta_2to4_minus_unstruct": (
                        struct["quality"]["hellaswag"]["diag_acc_norm"]
                        - unstruct["quality"]["hellaswag"]["diag_acc_norm"]
                    ),
                }
                if unstruct is not None
                else None
            ),
        }
        atomic_write_json(
            dirs["results"] / "structured_2to4_diagnostic.json", diag_payload
        )

    restore_eligible_weights(model, clean_weights)
    return all_results


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pruning(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
