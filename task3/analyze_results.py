"""Task 3 Step 7 Analysis: Paired Bootstrap CIs, Amortized Break-Even, Plots & Final Bundle.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task3.analyze_results --data-dir <DATA_DIR> --out-dir <T3_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common.io import (
    atomic_write_json,
    ensure_task_dirs,
    export_final_report_bundle,
    read_json,
    read_jsonl,
    resolve_smoke_out_dir,
)
from common.seeding import DEFAULT_SEED


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 3 Analysis: Prompt-paired bootstrap CIs, amortized cost break-even, plots, and bundle."
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _prompt_means(recs: list[dict[str, Any]], key: str) -> dict[str, float]:
    by_pid: dict[str, list[float]] = {}
    for r in recs:
        by_pid.setdefault(str(r["prompt_id"]), []).append(float(r[key]))
    return {pid: float(np.mean(vals)) for pid, vals in by_pid.items()}


def _paired_bootstrap_prompt_diff(
    recs_a: list[dict[str, Any]],
    recs_b: list[dict[str, Any]],
    key: str,
    n_resamples: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    ma = _prompt_means(recs_a, key)
    mb = _prompt_means(recs_b, key)
    pids = sorted(set(ma.keys()) & set(mb.keys()))
    if not pids:
        return {"status": "empty"}
    a = np.array([ma[p] for p in pids], dtype=np.float64)
    b = np.array([mb[p] for p in pids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pids), size=(n_resamples, len(pids)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5]).tolist()
    return {
        "n_prompts": len(pids),
        "n_resamples": int(n_resamples),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff_a_minus_b": float(a.mean() - b.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def analyze_task3(args: argparse.Namespace) -> dict[str, Any]:
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    res_dir = dirs["results"]
    plots_dir = dirs["plots"]

    # 1. Prompt-paired 95% bootstrap CIs (averaging reps within prompt, seeds kept separate)
    boot_out: dict[str, Any] = {
        "seed": int(args.seed),
        "n_resamples": int(args.bootstrap_resamples),
        "comparisons": {},
    }
    compare_pairs = [
        ("off_policy_s6304", "original_0.5b"),
        ("off_policy_s6305", "original_0.5b"),
        ("on_policy_s6304", "original_0.5b"),
        ("on_policy_s6305", "original_0.5b"),
        ("on_policy_s6304", "off_policy_s6304"),
        ("on_policy_s6305", "off_policy_s6305"),
    ]
    for g in (2, 4):
        for da, db in compare_pairs:
            fa = res_dir / f"spec_per_prompt_{da}_g{g}_eos64.jsonl"
            fb = res_dir / f"spec_per_prompt_{db}_g{g}_eos64.jsonl"
            if not (fa.exists() and fb.exists()):
                continue
            ra, rb = read_jsonl(fa), read_jsonl(fb)
            key = f"{da}_minus_{db}_g{g}"
            boot_out["comparisons"][key] = {
                "draft_a": da,
                "draft_b": db,
                "gamma": g,
                "accepted_over_examined": _paired_bootstrap_prompt_diff(
                    ra, rb, "accepted_over_examined", args.bootstrap_resamples, args.seed
                ),
                "throughput_ratio": _paired_bootstrap_prompt_diff(
                    ra, rb, "throughput_ratio_spec_over_target", args.bootstrap_resamples, args.seed + 1
                ),
            }
    atomic_write_json(res_dir / "speculative_bootstrap_ci.json", boot_out)

    # 2. Amortized deployment break-even calculation
    cost_doc = read_json(res_dir / "distillation_cost_accounting.json") if (res_dir / "distillation_cost_accounting.json").exists() else {}
    breakeven_out: dict[str, Any] = {"by_run_and_gamma": {}}
    for run_id, cinfo in cost_doc.items():
        upfront_sec = float(cinfo.get("total_end_to_end_cost_sec", 0.0))
        for g in (2, 4):
            f_spec = res_dir / f"spec_per_prompt_{run_id}_g{g}_eos64.jsonl"
            if not f_spec.exists():
                continue
            recs = read_jsonl(f_spec)
            # Measured wall-clock savings per request vs adjacent target-only decode
            mean_target_sec = float(np.mean([r["target_decode_sec"] for r in recs]))
            mean_spec_sec = float(np.mean([r["spec_decode_sec"] for r in recs]))
            savings_per_req_sec = mean_target_sec - mean_spec_sec
            if savings_per_req_sec > 0:
                be_requests: float | str = float(math.ceil(upfront_sec / savings_per_req_sec))
                finite = True
            else:
                be_requests = "non_finite_no_speedup"
                finite = False
            breakeven_out["by_run_and_gamma"][f"{run_id}_g{g}"] = {
                "run_id": run_id,
                "gamma": g,
                "upfront_distill_and_merge_sec": upfront_sec,
                "mean_target_decode_sec": mean_target_sec,
                "mean_spec_decode_sec": mean_spec_sec,
                "measured_savings_per_request_sec": savings_per_req_sec,
                "breakeven_requests": be_requests,
                "breakeven_is_finite": finite,
            }
    atomic_write_json(res_dir / "amortized_cost_breakeven.json", breakeven_out)

    # 3. 300-dpi plots
    saved_plots: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 10, "figure.dpi": 300})
    except Exception:
        plt = None

    if plt is not None:
        # Plot 1: Distillation KL vs scored tokens
        log_rows = _read_csv(res_dir / "distillation_training_logs.csv")
        if log_rows:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            by_run: dict[str, list[tuple[int, float]]] = {}
            for r in log_rows:
                by_run.setdefault(r["run_id"], []).append((int(r["total_tokens_cum"]), float(r["forward_kl"])))
            for rid, pts in sorted(by_run.items()):
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, label=rid)
            ax.set_xlabel("Teacher-Scored Response Tokens")
            ax.set_ylabel("Forward KL Loss (nats/token)")
            ax.set_title("Task 3: Off-Policy vs. On-Policy Distillation Curves")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = plots_dir / "distillation_kl_curves.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            saved_plots.append(str(p))

        # Plot 2: Speculative acceptance & measured vs analytical speedup
        spec_rows = [r for r in _read_csv(res_dir / "speculative_decoding_summary.csv") if r.get("eval_tag") == "eos_aware_64"]
        if spec_rows:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            labels = [f"{r['draft']}\ng={r['gamma']}" for r in spec_rows]
            x = np.arange(len(labels))
            acc = [float(r["accepted_over_examined"]) * 100.0 for r in spec_rows]
            meas = [float(r["measured_speedup_S_gamma"]) for r in spec_rows]
            anal = [float(r["analytical_speedup_S_hat"]) for r in spec_rows]
            axes[0].bar(x, acc, color="#4C72B0")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            axes[0].set_ylabel("Accepted / Examined (%)")
            axes[0].set_title("Empirical Proposal Acceptance")
            axes[0].grid(True, axis="y", alpha=0.3)

            axes[1].bar(x - 0.18, meas, width=0.35, label="Measured S_gamma")
            axes[1].bar(x + 0.18, anal, width=0.35, label="Analytical S_hat")
            axes[1].axhline(1.0, color="gray", linestyle="--")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            axes[1].set_ylabel("Speedup vs. Target-Only (x)")
            axes[1].set_title("Measured vs. Analytical Speedup")
            axes[1].grid(True, axis="y", alpha=0.3)
            axes[1].legend(fontsize=8)
            fig.tight_layout()
            p = plots_dir / "speculative_acceptance_and_speedup.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            saved_plots.append(str(p))

        # Plot 3: 4-part per-round timing decomposition
        t_rows = [r for r in _read_csv(res_dir / "speculative_round_timing_decomposition.csv") if r.get("eval_tag") == "eos_aware_64"]
        if t_rows:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            labels = [f"{r['draft']}\ng={r['gamma']}" for r in t_rows]
            x = np.arange(len(labels))
            parts = [
                ("Draft proposals", "mean_draft_proposals_ms"),
                ("Target verification", "mean_target_verification_ms"),
                ("Accept/residual", "mean_accept_residual_ms"),
                ("Cache rollback", "mean_cache_rollback_ms"),
                ("Host overhead", "mean_host_overhead_ms"),
            ]
            bottom = np.zeros(len(labels))
            for pname, col in parts:
                vals = np.array([float(r[col]) for r in t_rows])
                ax.bar(x, vals, bottom=bottom, label=pname)
                bottom += vals
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("Mean Decode Time per Prompt (ms)")
            ax.set_title("Task 3: Per-Round Speculative Timing Decomposition")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = plots_dir / "speculative_timing_decomposition.png"
            fig.savefig(p, dpi=300)
            plt.close(fig)
            saved_plots.append(str(p))

    bundle = export_final_report_bundle(
        dirs["root"],
        rel_items=[
            "configs/env_metadata.json",
            "results/step1_sampler_validation.json",
            "results/distillation_training_logs.csv",
            "results/distillation_cost_accounting.json",
            "results/prefix_bank_alignment.csv",
            "results/speculative_decoding_summary.csv",
            "results/speculative_round_timing_decomposition.csv",
            "results/speculative_bootstrap_ci.json",
            "results/standalone_student_accuracy.csv",
            "results/merged_vs_unmerged_check.json",
            "results/amortized_cost_breakeven.json",
            "results/condition_status.json",
            "evidence/qualitative_disagreements_and_continuations.json",
            "plots/",
        ],
    )
    print(
        f"[Task 3 Analysis] Generated {len(saved_plots)} plot(s) and exported "
        f"{len(bundle['copied_files'])} final file(s) to {bundle['bundle_root']}.",
        flush=True,
    )
    return {"plots": saved_plots, "final_report_bundle": bundle}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    analyze_task3(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
