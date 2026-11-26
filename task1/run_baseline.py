"""Task 1 Step 1: Unchanged FP16 Baseline & Measurement Protocol.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  Smoke:
    python -m task1.run_baseline --smoke --data-dir <DATA_DIR> --out-dir <SMOKE_OUT> --seed 6304
  Full:
    python -m task1.run_baseline --data-dir <DATA_DIR> --out-dir <T1_OUT> --seed 6304
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
from task1.data import load_hellaswag_val, load_ppl_windows
from task1.eval_hellaswag import (
    HELLASWAG_PROMPT_FORMAT_SPEC,
    HELLASWAG_TASK_VERSION,
    evaluate_hellaswag,
)
from task1.eval_ppl import evaluate_perplexity
from task1.model_utils import (
    DEFAULT_MODEL_ID,
    load_base_model_and_tokenizer,
    verify_model_structure,
)
from task1.storage import compute_fp16_and_sparse_storage

CONDITION_NAME = "fp16_baseline"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 1 Step 1: Establish unchanged FP16 baseline (Qwen/Qwen2.5-1.5B), "
            "verify eligible/tied weights, evaluate WikiText-2 & C4 perplexity and "
            "HellaSwag zero-shot, and measure storage/memory/latency."
        )
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
        help=(
            "Run a reduced end-to-end smoke test (4 PPL windows per suite, 16 HellaSwag "
            "examples, shorter timing sweep) and write outputs to a smoke directory "
            "separate from full runs."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip baseline execution if verified artifacts already exist for this config.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Pretrained base model ID (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=("fp16", "fp32"),
        help="Model precision (default: fp16; full PA4 protocol requires fp16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. 'cuda' or 'cpu'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--warmup-reps",
        type=int,
        default=None,
        help="Warmup repetitions for latency benchmark (default: 2 full, 1 smoke).",
    )
    parser.add_argument(
        "--timing-reps",
        type=int,
        default=None,
        help="Timed repetitions for latency benchmark (default: 5 full, 2 smoke).",
    )
    parser.add_argument(
        "--prefill-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Prefill prompt lengths to benchmark (default: 512 2048 full, 256 512 smoke).",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=None,
        help="Cached decode token count to benchmark (default: 64 full, 16 smoke).",
    )
    parser.add_argument(
        "--vocab-chunk-tokens",
        type=int,
        default=128,
        help="Sequence chunk size for FP32 cross-entropy during PPL eval (default: 128).",
    )
    return parser


def _resolve_run_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        ppl_limit: int | None = 4
        hs_limit: int | None = 16
        warmup = args.warmup_reps if args.warmup_reps is not None else 1
        reps = args.timing_reps if args.timing_reps is not None else 2
        prefill_lengths = (
            list(args.prefill_lengths) if args.prefill_lengths is not None else [256, 512]
        )
        decode_prompt_len = 256
        decode_tokens = args.decode_tokens if args.decode_tokens is not None else 16
    else:
        ppl_limit = None
        hs_limit = None
        warmup = args.warmup_reps if args.warmup_reps is not None else 2
        reps = args.timing_reps if args.timing_reps is not None else 5
        prefill_lengths = (
            list(args.prefill_lengths) if args.prefill_lengths is not None else [512, 2048]
        )
        decode_prompt_len = 512
        decode_tokens = args.decode_tokens if args.decode_tokens is not None else 64

    return {
        "smoke": bool(args.smoke),
        "ppl_window_limit": ppl_limit,
        "hellaswag_limit": hs_limit,
        "warmup_reps": int(warmup),
        "timing_reps": int(reps),
        "prefill_lengths": [int(x) for x in prefill_lengths],
        "decode_prompt_len": int(decode_prompt_len),
        "decode_tokens": int(decode_tokens),
        "vocab_chunk_tokens": int(args.vocab_chunk_tokens),
    }


def _baseline_artifacts_valid(
    dirs: dict[str, Path],
    expected_seed: int,
    expected_smoke: bool,
    expected_model_id: str,
) -> bool:
    required = [
        dirs["results"] / "baseline_fp16.json",
        dirs["results"] / "ppl_windows_wikitext2_fp16_baseline.jsonl",
        dirs["results"] / "ppl_windows_c4_fp16_baseline.jsonl",
        dirs["results"] / "hellaswag_per_example_fp16_baseline.jsonl",
        dirs["configs"] / "baseline_config.json",
    ]
    if not all(p.exists() and p.stat().st_size > 0 for p in required):
        return False
    try:
        summary = read_json(dirs["results"] / "baseline_fp16.json")
        return (
            summary.get("status") == "completed"
            and summary.get("seed") == expected_seed
            and bool(summary.get("smoke")) == expected_smoke
            and summary.get("model_id") == expected_model_id
        )
    except Exception:
        return False


def run_baseline(
    args: argparse.Namespace,
    model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Execute Task 1 FP16 baseline (accepts preloaded model/tokenizer for tests)."""
    t_total0 = time.perf_counter()
    set_seed(args.seed)

    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    if args.smoke and resolved_out != Path(args.out_dir):
        print(
            f"[SMOKE] Redirected smoke outputs to separate directory: {resolved_out}",
            flush=True,
        )
    dirs = ensure_task_dirs(resolved_out)
    status_path = dirs["results"] / "condition_status.json"
    condition_key = f"{CONDITION_NAME}_smoke" if args.smoke else CONDITION_NAME

    if args.resume and _baseline_artifacts_valid(
        dirs, args.seed, args.smoke, args.model_id
    ):
        print(
            f"[RESUME] Verified baseline artifacts already exist in {dirs['results']}; skipping.",
            flush=True,
        )
        return read_json(dirs["results"] / "baseline_fp16.json")

    update_condition_status(
        status_path,
        condition_key,
        "pending",
        {"seed": args.seed, "smoke": bool(args.smoke), "model_id": args.model_id},
    )

    settings = _resolve_run_settings(args)
    try:
        t_load0 = time.perf_counter()
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
        load_time_sec = time.perf_counter() - t_load0

        model_revision = resolve_model_revision(model, args.model_id)
        local_revs = load_local_source_revisions(args.data_dir)
        env_meta = collect_env_metadata(
            data_dir=args.data_dir,
            model_revisions={args.model_id: model_revision},
            seed=args.seed,
            extra={
                "stage": "task1_step1_baseline",
                "smoke": bool(args.smoke),
                "dtype": str(torch_dtype),
                "device": str(device),
                "tokenizer_class": tokenizer.__class__.__name__,
                "vocab_size": int(getattr(model.config, "vocab_size", len(tokenizer))),
            },
        )
        atomic_write_json(dirs["configs"] / "env_metadata.json", env_meta)

        print(
            f"[Task 1 Baseline] Model={args.model_id} (rev={model_revision}) "
            f"dtype={torch_dtype} device={device} smoke={args.smoke}",
            flush=True,
        )

        # 1. Verify eligible projections and tied weights
        structure_report = verify_model_structure(model)
        print(
            f"[Task 1 Baseline] Verified {structure_report['eligible_matrix_count']} eligible "
            f"matrices ({structure_report['eligible_weight_params']:,} weights), "
            f"{structure_report['unique_total_params']:,} unique params "
            f"({len(structure_report['tied_aliases'])} tied alias(es)).",
            flush=True,
        )

        # 2. Storage accounting (dense FP16 + 16-bit unstructured sparse model)
        storage_report = compute_fp16_and_sparse_storage(model)

        # 3. Load locked datasets
        wt2_path, wt2_windows = load_ppl_windows(
            args.data_dir, "wikitext2", limit=settings["ppl_window_limit"]
        )
        c4_path, c4_windows = load_ppl_windows(
            args.data_dir, "c4_validation", limit=settings["ppl_window_limit"]
        )
        hs_path, hs_records = load_hellaswag_val(
            args.data_dir, limit=settings["hellaswag_limit"]
        )

        run_config = {
            "condition": CONDITION_NAME,
            "smoke": bool(args.smoke),
            "seed": int(args.seed),
            "model_id": args.model_id,
            "model_revision": model_revision,
            "dtype": str(torch_dtype),
            "device": str(device),
            "out_dir": str(dirs["root"]),
            "data_dir": str(args.data_dir),
            "manifests": {
                "wikitext2": str(wt2_path),
                "c4_validation": str(c4_path),
                "hellaswag": str(hs_path),
            },
            "local_source_revisions": local_revs,
            "hellaswag_task_version": HELLASWAG_TASK_VERSION,
            "hellaswag_prompt_format_spec": HELLASWAG_PROMPT_FORMAT_SPEC,
            "settings": settings,
        }
        atomic_write_json(dirs["configs"] / "baseline_config.json", run_config)

        # 4. Perplexity evaluations
        print(
            f"[Task 1 Baseline] Evaluating WikiText-2 ({len(wt2_windows)} windows)...",
            flush=True,
        )
        wt2_summary, wt2_per_win = evaluate_perplexity(
            model,
            wt2_windows,
            device=device,
            suite_name="wikitext2",
            vocab_chunk_tokens=settings["vocab_chunk_tokens"],
        )
        atomic_write_jsonl(
            dirs["results"] / "ppl_windows_wikitext2_fp16_baseline.jsonl",
            wt2_per_win,
        )
        print(
            f"  WikiText-2: N={wt2_summary['scored_tokens']}, "
            f"mean_nll={wt2_summary['mean_nll']:.6f}, PPL={wt2_summary['perplexity']:.4f}",
            flush=True,
        )

        print(
            f"[Task 1 Baseline] Evaluating C4 validation ({len(c4_windows)} windows)...",
            flush=True,
        )
        c4_summary, c4_per_win = evaluate_perplexity(
            model,
            c4_windows,
            device=device,
            suite_name="c4_validation",
            vocab_chunk_tokens=settings["vocab_chunk_tokens"],
        )
        atomic_write_jsonl(
            dirs["results"] / "ppl_windows_c4_fp16_baseline.jsonl",
            c4_per_win,
        )
        print(
            f"  C4 val:     N={c4_summary['scored_tokens']}, "
            f"mean_nll={c4_summary['mean_nll']:.6f}, PPL={c4_summary['perplexity']:.4f}",
            flush=True,
        )

        # 5. Zero-shot HellaSwag evaluation
        print(
            f"[Task 1 Baseline] Evaluating HellaSwag ({len(hs_records)} examples)...",
            flush=True,
        )
        hs_summary, hs_per_ex = evaluate_hellaswag(
            model,
            tokenizer,
            hs_records,
            device=device,
            diag_limit=512,
        )
        atomic_write_jsonl(
            dirs["results"] / "hellaswag_per_example_fp16_baseline.jsonl",
            hs_per_ex,
        )
        print(
            f"  HellaSwag (n={hs_summary['num_examples']}): "
            f"acc_raw={hs_summary['acc_raw']:.4f}, acc_norm={hs_summary['acc_norm']:.4f} | "
            f"diag_{hs_summary['diag_subset_size']}: "
            f"acc_raw={hs_summary['diag_acc_raw']:.4f}, acc_norm={hs_summary['diag_acc_norm']:.4f}",
            flush=True,
        )

        # 6. Prefill & cached decode latency + peak GPU memory
        print(
            f"[Task 1 Baseline] Benchmarking prefill {settings['prefill_lengths']} "
            f"and {settings['decode_tokens']}-token cached decode "
            f"(warmup={settings['warmup_reps']}, reps={settings['timing_reps']})...",
            flush=True,
        )
        bench_report = benchmark_prefill_and_decode(
            model=model,
            device=device,
            prefill_lengths=settings["prefill_lengths"],
            decode_prompt_len=settings["decode_prompt_len"],
            decode_tokens=settings["decode_tokens"],
            warmup=settings["warmup_reps"],
            reps=settings["timing_reps"],
            seed=args.seed,
        )

        total_time_sec = time.perf_counter() - t_total0
        result_payload: dict[str, Any] = {
            "status": "completed",
            "condition": CONDITION_NAME,
            "smoke": bool(args.smoke),
            "seed": int(args.seed),
            "model_id": args.model_id,
            "model_revision": model_revision,
            "dtype": str(torch_dtype),
            "device": str(device),
            "structure_verification": structure_report,
            "storage": storage_report,
            "quality": {
                "wikitext2": wt2_summary,
                "c4_validation": c4_summary,
                "hellaswag": hs_summary,
            },
            "benchmark": bench_report,
            "timing_sec": {
                "model_load_sec": float(load_time_sec),
                "wikitext2_eval_sec": float(wt2_summary["eval_time_sec"]),
                "c4_eval_sec": float(c4_summary["eval_time_sec"]),
                "hellaswag_eval_sec": float(hs_summary["eval_time_sec"]),
                "total_wall_sec": float(total_time_sec),
            },
            "artifacts": {
                "env_metadata": str(dirs["configs"] / "env_metadata.json"),
                "baseline_config": str(dirs["configs"] / "baseline_config.json"),
                "baseline_summary": str(dirs["results"] / "baseline_fp16.json"),
                "wikitext2_windows": str(
                    dirs["results"] / "ppl_windows_wikitext2_fp16_baseline.jsonl"
                ),
                "c4_windows": str(
                    dirs["results"] / "ppl_windows_c4_fp16_baseline.jsonl"
                ),
                "hellaswag_per_example": str(
                    dirs["results"] / "hellaswag_per_example_fp16_baseline.jsonl"
                ),
            },
        }
        atomic_write_json(dirs["results"] / "baseline_fp16.json", result_payload)
        update_condition_status(
            status_path,
            condition_key,
            "completed",
            {
                "seed": int(args.seed),
                "smoke": bool(args.smoke),
                "wikitext2_ppl": wt2_summary["perplexity"],
                "c4_ppl": c4_summary["perplexity"],
                "hellaswag_acc_norm": hs_summary["acc_norm"],
                "summary_path": str(dirs["results"] / "baseline_fp16.json"),
            },
        )
        print(
            f"[Task 1 Baseline] Completed in {total_time_sec:.2f}s. "
            f"Saved to {dirs['results'] / 'baseline_fp16.json'}",
            flush=True,
        )
        return result_payload

    except Exception as exc:
        update_condition_status(
            status_path,
            condition_key,
            "failed",
            {
                "seed": int(args.seed),
                "smoke": bool(args.smoke),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_baseline(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
