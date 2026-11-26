"""Task 2 Analysis & Plots: Generate 300-dpi figures and export final_report_bundle/.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task2.analyze_results --data-dir <DATA_DIR> --out-dir <T2_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common.io import (
    ensure_task_dirs,
    export_final_report_bundle,
    resolve_smoke_out_dir,
)
from common.seeding import DEFAULT_SEED


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 2 Analysis: Generate 300-dpi plots and export final_report_bundle/."
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def analyze_task2(args: argparse.Namespace) -> dict[str, Any]:
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    res_dir = dirs["results"]
    plots_dir = dirs["plots"]
    saved_plots: list[str] = []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 10, "figure.dpi": 300})
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable ({exc}); skipping plots.", flush=True)
        plt = None

    # 1. Part A: Latency & Peak Memory vs Sequence Length
    pa_rows = _read_csv(res_dir / "partA_fidelity_and_timing.csv")
    if plt is not None and pa_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for op, style in (("dense", "o-"), ("tiled", "s-"), ("sdpa", "^-")):
            sub = [
                r for r in pa_rows
                if r["operator"] == op and r["dtype"] == ("fp16" if any(x["dtype"] == "fp16" for x in pa_rows) else "fp32")
                and r["fwd_mean_ms"] != ""
            ]
            if not sub:
                continue
            xs = [int(r["seq_len"]) for r in sub]
            ys_ms = [float(r["fwd_mean_ms"]) for r in sub]
            ys_mb = [float(r["fwd_peak_alloc_bytes"] or 0) / 1e6 for r in sub]
            axes[0].plot(xs, ys_ms, style, label=f"{op} (fwd)")
            axes[1].plot(xs, ys_mb, style, label=f"{op} (fwd)")
        axes[0].set_xscale("log", base=2)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Sequence Length L")
        axes[0].set_ylabel("Forward Latency (ms)")
        axes[0].set_title("Part A: Exact Attention Latency vs. L")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("Sequence Length L")
        axes[1].set_ylabel("Peak Allocated GPU Memory (MB)")
        axes[1].set_title("Part A: Peak Allocated Memory vs. L")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        p = plots_dir / "partA_latency_and_memory.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        saved_plots.append(str(p))

    # 2. Part B: Learning curves & accuracy summary
    pb_curves = _read_csv(res_dir / "partB_learning_curves.csv")
    if plt is not None and pb_curves:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        grouped: dict[tuple[str, str], dict[int, list[float]]] = {}
        for r in pb_curves:
            key = (r["attn_mode"], f"K={r['K']}_{r['train_regime']}")
            grouped.setdefault(key, {}).setdefault(int(r["step"]), []).append(float(r["dev_mixed_acc"]))
        for (mode, tag), step_map in sorted(grouped.items()):
            steps = sorted(step_map.keys())
            means = [float(np.mean(step_map[s])) * 100.0 for s in steps]
            ax.plot(steps, means, marker="o", markersize=3, label=f"{mode} ({tag})")
        ax.set_xlabel("Optimizer Updates")
        ax.set_ylabel("Dev Mixed Accuracy (%)")
        ax.set_title("Part B: MQAR Learning Curves")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = plots_dir / "partB_learning_curves.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        saved_plots.append(str(p))

    # 3. Part C: Matched-budget KV fidelity & packed encode/decode benchmark
    pc_rows = _read_csv(res_dir / "partC_kv_fidelity_matched_budget.csv")
    if plt is not None and pc_rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        pols = [r["policy"] for r in pc_rows]
        uncond = [float(r["unconditional_exact_match"]) * 100.0 for r in pc_rows]
        cond = [float(r["conditional_on_fp16_correct"]) * 100.0 for r in pc_rows]
        nlls = [float(r["mean_ref_answer_nll"]) for r in pc_rows]
        x = np.arange(len(pols))
        axes[0].bar(x - 0.18, uncond, width=0.35, label="Unconditional EM")
        axes[0].bar(x + 0.18, cond, width=0.35, label="Cond. on FP16 Correct")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(pols, rotation=28, ha="right", fontsize=8)
        axes[0].set_ylabel("Exact-Match Accuracy (%)")
        axes[0].set_title("Part C: Matched INT4-Budget Retrieval Accuracy")
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[0].legend(fontsize=8)

        axes[1].bar(x, nlls, width=0.5, color="#4C72B0")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(pols, rotation=28, ha="right", fontsize=8)
        axes[1].set_ylabel("Mean Reference-Answer NLL (lower is better)")
        axes[1].set_title("Part C: Reference-Answer Likelihood")
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p = plots_dir / "partC_kv_matched_budget_comparison.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        saved_plots.append(str(p))

    pc_bench = _read_csv(res_dir / "partC_packed_encode_decode_bench.csv")
    if plt is not None and pc_bench:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        labels = [f"{r['config']}\nB={r['batch_size']},L={r['seq_len']}" for r in pc_bench]
        vals = [float(r["decode_step_all_layers_ms"]) for r in pc_bench]
        ax.bar(range(len(labels)), vals)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Decode-Step Latency w/ Unpack (ms, 28 layers)")
        ax.set_title("Part C5: Packed KV Decode-Step Latency")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p = plots_dir / "partC_packed_encode_decode_bench.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        saved_plots.append(str(p))

    bundle = export_final_report_bundle(
        dirs["root"],
        rel_items=[
            "configs/env_metadata.json",
            "results/partA_fidelity_and_timing.csv",
            "results/partA_backends.json",
            "results/partB_learning_curves.csv",
            "results/partB_mqar_accuracy_summary.csv",
            "results/partB_distance_bins.csv",
            "results/partB_inference_sweep.csv",
            "results/partC_kv_fidelity_matched_budget.csv",
            "results/partC_kv_breakdown_factorial.csv",
            "results/partC_bootstrap_ci.json",
            "results/partC_packed_encode_decode_bench.csv",
            "results/partC_c1_validation.json",
            "results/condition_status.json",
            "evidence/partC_qualitative_generations.json",
            "plots/",
        ],
    )
    print(
        f"[Task 2 Analysis] Generated {len(saved_plots)} plot(s) and exported "
        f"{len(bundle['copied_files'])} final file(s) to {bundle['bundle_root']}.",
        flush=True,
    )
    return {"plots": saved_plots, "final_report_bundle": bundle}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    analyze_task2(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
