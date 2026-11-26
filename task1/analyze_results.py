"""Task 1 Step 9: Summary Tables, Paired 95% Bootstrap CIs, Qualitative Cases & Plots.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task1.analyze_results --data-dir <DATA_DIR> --out-dir <T1_OUT> --seed 6304
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
    atomic_write_text,
    ensure_task_dirs,
    export_final_report_bundle,
    read_json,
    read_jsonl,
    resolve_smoke_out_dir,
)
from common.seeding import DEFAULT_SEED
from task1.data import load_hellaswag_val
from task1.eval_hellaswag import format_hellaswag_example

CANONICAL_CONDITIONS: tuple[str, ...] = (
    "fp16_baseline",
    "p1_magnitude_s25",
    "p1_magnitude_s50",
    "p1_magnitude_s75",
    "p2_wanda_s25",
    "p2_wanda_s50",
    "p2_wanda_s75",
    "p2_wanda_2to4_s50",
    "p3_nonuniform_s50",
    "p3_nonuniform_s75",
    "q1_uniform_int8",
    "q1_uniform_int4",
    "q1_uniform_int3",
    "q2_awq_int8",
    "q2_awq_int4",
    "q2_awq_int3",
    "q3_mixed_int4",
    "q3_mixed_int3",
)

BOOTSTRAP_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("p3_minus_p2_s50", "p3_nonuniform_s50", "p2_wanda_s50"),
    ("p3_minus_p2_s75", "p3_nonuniform_s75", "p2_wanda_s75"),
    ("q3_minus_q2_int4", "q3_mixed_int4", "q2_awq_int4"),
    ("q3_minus_q2_int3", "q3_mixed_int3", "q2_awq_int3"),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 1 Step 9: Aggregate all conditions, compute paired 95% bootstrap CIs "
            "for P3-P2 and Q3-Q2, export qualitative cases, and generate 300-dpi plots."
        )
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    return parser


def paired_bootstrap_ppl_diff(
    windows_a: list[dict[str, Any]],
    windows_b: list[dict[str, Any]],
    n_resamples: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Paired window-level bootstrap for PPL(A) - PPL(B).

    Resamples whole fixed evaluation windows with replacement and recomputes
    pooled token NLL before exponentiating (ATML PA4 PDF p. 5).
    """
    if len(windows_a) != len(windows_b):
        raise ValueError(
            f"Mismatched window counts: {len(windows_a)} vs {len(windows_b)}."
        )
    n_win = len(windows_a)
    nll_a = np.array([float(w["nll_sum"]) for w in windows_a], dtype=np.float64)
    nll_b = np.array([float(w["nll_sum"]) for w in windows_b], dtype=np.float64)
    tok_a = np.array([float(w["scored_tokens"]) for w in windows_a], dtype=np.float64)
    tok_b = np.array([float(w["scored_tokens"]) for w in windows_b], dtype=np.float64)
    if not np.array_equal(tok_a, tok_b):
        raise ValueError("Scored token counts differ between paired PPL windows.")

    point_ppl_a = float(math.exp(nll_a.sum() / tok_a.sum()))
    point_ppl_b = float(math.exp(nll_b.sum() / tok_b.sum()))
    point_diff = point_ppl_a - point_ppl_b

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_win, size=(n_resamples, n_win))
    tok_sum = tok_a[idx].sum(axis=1)
    ppl_a_boot = np.exp(nll_a[idx].sum(axis=1) / tok_sum)
    ppl_b_boot = np.exp(nll_b[idx].sum(axis=1) / tok_sum)
    diffs = ppl_a_boot - ppl_b_boot

    lo, hi = np.percentile(diffs, [2.5, 97.5]).tolist()
    return {
        "n_windows": int(n_win),
        "n_resamples": int(n_resamples),
        "point_a": point_ppl_a,
        "point_b": point_ppl_b,
        "point_diff_a_minus_b": float(point_diff),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def paired_bootstrap_acc_diff(
    ex_a: list[dict[str, Any]],
    ex_b: list[dict[str, Any]],
    metric_key: str = "correct_norm",
    n_resamples: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Paired example-level bootstrap for HellaSwag accuracy(A) - accuracy(B)."""
    n = min(len(ex_a), len(ex_b))
    if n == 0:
        raise ValueError("Empty HellaSwag per-example list.")
    a = np.array([float(r[metric_key]) for r in ex_a[:n]], dtype=np.float64)
    b = np.array([float(r[metric_key]) for r in ex_b[:n]], dtype=np.float64)

    point_a = float(a.mean())
    point_b = float(b.mean())
    point_diff = point_a - point_b

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5]).tolist()
    return {
        "n_examples": int(n),
        "metric": metric_key,
        "n_resamples": int(n_resamples),
        "point_a": point_a,
        "point_b": point_b,
        "point_diff_a_minus_b": float(point_diff),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def _load_condition_json(results_dir: Path, cond: str) -> dict[str, Any] | None:
    path = (
        results_dir / "baseline_fp16.json"
        if cond == "fp16_baseline"
        else results_dir / f"condition_{cond}.json"
    )
    if not path.exists():
        return None
    return read_json(path)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    atomic_write_text(path, buf.getvalue())


def _generate_plots(
    dirs: dict[str, Path],
    cond_docs: dict[str, dict[str, Any]],
) -> list[str]:
    """Generate clean 300-dpi PNG plots under plots/."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable ({exc}); skipping plots.", flush=True)
        return []

    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})
    saved_plots: list[str] = []
    plots_dir = dirs["plots"]
    base = cond_docs.get("fp16_baseline")

    # 1. Pruning resource-quality plot
    pruning_groups = [
        ("P1 Magnitude", ["p1_magnitude_s25", "p1_magnitude_s50", "p1_magnitude_s75"], "o-"),
        ("P2 Wanda", ["p2_wanda_s25", "p2_wanda_s50", "p2_wanda_s75"], "s-"),
        ("P3 Nonuniform", ["p3_nonuniform_s50", "p3_nonuniform_s75"], "^-"),
        ("P2 Wanda 2:4", ["p2_wanda_2to4_s50"], "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, cids, style in pruning_groups:
        xs, y_wt2, y_hs = [], [], []
        for cid in cids:
            doc = cond_docs.get(cid)
            if doc is None:
                continue
            xs.append(doc["storage"]["achieved_sparsity"] * 100.0)
            y_wt2.append(doc["quality"]["wikitext2"]["perplexity"])
            y_hs.append(doc["quality"]["hellaswag"]["diag_acc_norm"] * 100.0)
        if xs:
            axes[0].plot(xs, y_wt2, style, label=label)
            axes[1].plot(xs, y_hs, style, label=label)
    if base is not None:
        axes[0].axhline(
            base["quality"]["wikitext2"]["perplexity"],
            color="gray",
            linestyle="--",
            label="FP16 Baseline",
        )
        axes[1].axhline(
            base["quality"]["hellaswag"]["diag_acc_norm"] * 100.0,
            color="gray",
            linestyle="--",
            label="FP16 Baseline",
        )
    axes[0].set_xlabel("Eligible-Weight Sparsity (%)")
    axes[0].set_ylabel("WikiText-2 Perplexity (lower is better)")
    axes[0].set_title("Pruning: Perplexity vs. Sparsity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Eligible-Weight Sparsity (%)")
    axes[1].set_ylabel("HellaSwag acc_norm (%, paired 512 diag)")
    axes[1].set_title("Pruning: HellaSwag vs. Sparsity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p_path = plots_dir / "pruning_resource_quality.png"
    fig.savefig(p_path, dpi=300)
    plt.close(fig)
    saved_plots.append(str(p_path))

    # 2. Quantization resource-quality plot
    quant_groups = [
        ("Q1 Uniform PTQ", ["q1_uniform_int8", "q1_uniform_int4", "q1_uniform_int3"], "o-"),
        ("Q2 AWQ", ["q2_awq_int8", "q2_awq_int4", "q2_awq_int3"], "s-"),
        ("Q3 Mixed AWQ", ["q3_mixed_int4", "q3_mixed_int3"], "^-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, cids, style in quant_groups:
        xs, y_wt2, y_hs = [], [], []
        for cid in cids:
            doc = cond_docs.get(cid)
            if doc is None:
                continue
            xs.append(doc["storage"]["serialized_packed_total_bytes"] / 1e9)
            y_wt2.append(doc["quality"]["wikitext2"]["perplexity"])
            y_hs.append(doc["quality"]["hellaswag"]["diag_acc_norm"] * 100.0)
        if xs:
            axes[0].plot(xs, y_wt2, style, label=label)
            axes[1].plot(xs, y_hs, style, label=label)
    if base is not None:
        base_gb = base["storage"]["dense_fp16_total_bytes"] / 1e9
        axes[0].scatter(
            [base_gb],
            [base["quality"]["wikitext2"]["perplexity"]],
            color="black",
            marker="*",
            s=80,
            label="FP16 Baseline",
        )
        axes[1].scatter(
            [base_gb],
            [base["quality"]["hellaswag"]["diag_acc_norm"] * 100.0],
            color="black",
            marker="*",
            s=80,
            label="FP16 Baseline",
        )
    axes[0].set_xlabel("Actual Packed Model Storage (GB, incl. metadata)")
    axes[0].set_ylabel("WikiText-2 Perplexity (lower is better)")
    axes[0].set_title("Quantization: Perplexity vs. Packed Storage")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Actual Packed Model Storage (GB, incl. metadata)")
    axes[1].set_ylabel("HellaSwag acc_norm (%, paired 512 diag)")
    axes[1].set_title("Quantization: HellaSwag vs. Packed Storage")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    q_path = plots_dir / "quantization_resource_quality.png"
    fig.savefig(q_path, dpi=300)
    plt.close(fig)
    saved_plots.append(str(q_path))

    # 3. P3 sensitivity & allocation heatmap
    p3_file = dirs["results"] / "p3_allocation_and_sensitivity.json"
    if p3_file.exists():
        p3_doc = read_json(p3_file)
        first_key = next(iter(p3_doc.get("by_sparsity", {})), None)
        if first_key is not None:
            units = p3_doc["by_sparsity"][first_key]["selected_allocation"]["per_unit"]
            proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            n_layers = max(u["layer_idx"] for u in units) + 1
            sens_grid = np.zeros((n_layers, len(proj_names)))
            spar_grid = np.zeros((n_layers, len(proj_names)))
            for u in units:
                j = proj_names.index(u["proj_name"])
                sens_grid[u["layer_idx"], j] = u["sensitivity"]
                spar_grid[u["layer_idx"], j] = u["sparsity"]
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            im0 = axes[0].imshow(sens_grid, aspect="auto", cmap="viridis")
            axes[0].set_xticks(range(len(proj_names)))
            axes[0].set_xticklabels(proj_names, rotation=35, ha="right")
            axes[0].set_ylabel("Transformer Layer")
            axes[0].set_title(f"P3 Sensitivity ({first_key})")
            fig.colorbar(im0, ax=axes[0])
            im1 = axes[1].imshow(spar_grid, aspect="auto", cmap="magma_r", vmin=0.10, vmax=0.85)
            axes[1].set_xticks(range(len(proj_names)))
            axes[1].set_xticklabels(proj_names, rotation=35, ha="right")
            axes[1].set_ylabel("Transformer Layer")
            axes[1].set_title(f"P3 Allocated Sparsity ({first_key})")
            fig.colorbar(im1, ax=axes[1])
            fig.tight_layout()
            p3_plot = plots_dir / "p3_sensitivity_and_allocation.png"
            fig.savefig(p3_plot, dpi=300)
            plt.close(fig)
            saved_plots.append(str(p3_plot))

    # 4. Q3 sensitivity & bit-width allocation heatmap
    q3_file = dirs["results"] / "q3_allocation_and_sensitivity.json"
    if q3_file.exists():
        q3_doc = read_json(q3_file)
        first_key = next(iter(q3_doc.get("by_budget", {})), None)
        if first_key is not None:
            units = q3_doc["by_budget"][first_key]["per_unit"]
            proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            n_layers = max(u["layer_idx"] for u in units) + 1
            err_grid = np.zeros((n_layers, len(proj_names)))
            bit_grid = np.zeros((n_layers, len(proj_names)))
            for u in units:
                j = proj_names.index(u["proj_name"])
                err_grid[u["layer_idx"], j] = u["calib_rel_l2_err_at_assigned_bits"]
                bit_grid[u["layer_idx"], j] = u["assigned_bits"]
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            im0 = axes[0].imshow(err_grid, aspect="auto", cmap="viridis")
            axes[0].set_xticks(range(len(proj_names)))
            axes[0].set_xticklabels(proj_names, rotation=35, ha="right")
            axes[0].set_ylabel("Transformer Layer")
            axes[0].set_title(f"Q3 Calib Rel L2 Error ({first_key})")
            fig.colorbar(im0, ax=axes[0])
            im1 = axes[1].imshow(bit_grid, aspect="auto", cmap="coolwarm", vmin=2, vmax=8)
            axes[1].set_xticks(range(len(proj_names)))
            axes[1].set_xticklabels(proj_names, rotation=35, ha="right")
            axes[1].set_ylabel("Transformer Layer")
            axes[1].set_title(f"Q3 Assigned Bit Width ({first_key})")
            fig.colorbar(im1, ax=axes[1])
            fig.tight_layout()
            q3_plot = plots_dir / "q3_sensitivity_and_allocation.png"
            fig.savefig(q3_plot, dpi=300)
            plt.close(fig)
            saved_plots.append(str(q3_plot))

    # 5. Calibration output error vs held-out WikiText-2 PPL diagnostic
    xs_err, ys_ppl, labels_err = [], [], []
    for cid, doc in cond_docs.items():
        if cid == "fp16_baseline":
            continue
        ce = doc.get("calibration_output_error")
        if ce and "mean_projection_rel_l2_error" in ce:
            xs_err.append(float(ce["mean_projection_rel_l2_error"]))
            ys_ppl.append(float(doc["quality"]["wikitext2"]["perplexity"]))
            labels_err.append(cid)
    if xs_err:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(xs_err, ys_ppl, s=45)
        for x, y, lbl in zip(xs_err, ys_ppl, labels_err):
            ax.annotate(lbl, (x, y), fontsize=7, alpha=0.85, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Mean Projection Relative L2 Calibration Error")
        ax.set_ylabel("Held-Out WikiText-2 Perplexity")
        ax.set_title("Calibration Output Error vs. Held-Out Perplexity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        c_plot = plots_dir / "calib_error_vs_heldout.png"
        fig.savefig(c_plot, dpi=300)
        plt.close(fig)
        saved_plots.append(str(c_plot))

    return saved_plots


def analyze_task1(args: argparse.Namespace) -> dict[str, Any]:
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    results_dir = dirs["results"]

    cond_docs: dict[str, dict[str, Any]] = {}
    for cid in CANONICAL_CONDITIONS:
        doc = _load_condition_json(results_dir, cid)
        if doc is not None:
            cond_docs[cid] = doc

    base = cond_docs.get("fp16_baseline")
    base_wt2 = base["quality"]["wikitext2"]["perplexity"] if base else None
    base_c4 = base["quality"]["c4_validation"]["perplexity"] if base else None
    base_hs_norm = base["quality"]["hellaswag"]["acc_norm"] if base else None
    base_hs_diag = base["quality"]["hellaswag"]["diag_acc_norm"] if base else None

    summary_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []

    for cid in CANONICAL_CONDITIONS:
        doc = cond_docs.get(cid)
        if doc is None:
            summary_rows.append({"condition": cid, "status": "pending"})
            resource_rows.append({"condition": cid, "status": "pending"})
            continue

        wt2 = doc["quality"]["wikitext2"]
        c4 = doc["quality"]["c4_validation"]
        hs = doc["quality"]["hellaswag"]
        st = doc["storage"]
        bm = doc["benchmark"]
        ce = doc.get("calibration_output_error") or {}
        tm = doc.get("timing_sec") or {}

        summary_rows.append(
            {
                "condition": cid,
                "status": doc.get("status", "completed"),
                "family": doc.get("family", "baseline"),
                "achieved_sparsity": st.get("achieved_sparsity", 0.0),
                "nominal_bits": st.get("nominal_bits_per_eligible_weight", 16.0),
                "packed_bytes": st.get("serialized_packed_total_bytes", ""),
                "sparse_16bit_bytes": st.get("sparse_16bit_total_bytes", ""),
                "wt2_scored_tokens": wt2["scored_tokens"],
                "wt2_mean_nll": wt2["mean_nll"],
                "wt2_ppl": wt2["perplexity"],
                "wt2_ppl_delta": (wt2["perplexity"] - base_wt2) if base_wt2 is not None else "",
                "c4_scored_tokens": c4["scored_tokens"],
                "c4_mean_nll": c4["mean_nll"],
                "c4_ppl": c4["perplexity"],
                "c4_ppl_delta": (c4["perplexity"] - base_c4) if base_c4 is not None else "",
                "hs_n": hs["num_examples"],
                "hs_acc_raw": hs["acc_raw"],
                "hs_acc_norm": hs["acc_norm"],
                "hs_acc_norm_delta": (hs["acc_norm"] - base_hs_norm)
                if (base_hs_norm is not None and hs["num_examples"] == base["quality"]["hellaswag"]["num_examples"])
                else "",
                "hs_diag_n": hs["diag_subset_size"],
                "hs_diag_acc_norm": hs["diag_acc_norm"],
                "hs_diag_acc_norm_delta": (hs["diag_acc_norm"] - base_hs_diag)
                if base_hs_diag is not None
                else "",
                "calib_mean_rel_l2_err": ce.get("mean_projection_rel_l2_error", ""),
            }
        )

        prefill_512 = bm["prefill"].get("512", {})
        prefill_2048 = bm["prefill"].get("2048", {})
        decode = bm.get("decode", {})
        resource_rows.append(
            {
                "condition": cid,
                "status": doc.get("status", "completed"),
                "achieved_sparsity": st.get("achieved_sparsity", 0.0),
                "nominal_bits": st.get("nominal_bits_per_eligible_weight", 16.0),
                "dense_fp16_total_bytes": st.get("dense_fp16_total_bytes", ""),
                "sparse_16bit_total_bytes": st.get("sparse_16bit_total_bytes", ""),
                "serialized_packed_total_bytes": st.get("serialized_packed_total_bytes", ""),
                "eligible_metadata_bytes": st.get("eligible_metadata_bytes", 0),
                "compensation_scale_bytes": st.get("compensation_scale_bytes", 0),
                "prefill_512_mean_ms": prefill_512.get("mean_ms", ""),
                "prefill_2048_mean_ms": prefill_2048.get("mean_ms", ""),
                "decode_mean_ms": decode.get("mean_ms", ""),
                "decode_tokens": decode.get("decode_tokens", ""),
                "peak_allocated_gpu_bytes": bm.get("overall_peak_allocated_bytes", 0),
                "peak_reserved_gpu_bytes": bm.get("overall_peak_reserved_bytes", 0),
                "total_wall_sec": tm.get("total_wall_sec", ""),
            }
        )

    _write_csv(
        results_dir / "summary_all_conditions.csv",
        [
            "condition", "status", "family", "achieved_sparsity", "nominal_bits",
            "packed_bytes", "sparse_16bit_bytes",
            "wt2_scored_tokens", "wt2_mean_nll", "wt2_ppl", "wt2_ppl_delta",
            "c4_scored_tokens", "c4_mean_nll", "c4_ppl", "c4_ppl_delta",
            "hs_n", "hs_acc_raw", "hs_acc_norm", "hs_acc_norm_delta",
            "hs_diag_n", "hs_diag_acc_norm", "hs_diag_acc_norm_delta",
            "calib_mean_rel_l2_err",
        ],
        summary_rows,
    )
    _write_csv(
        results_dir / "resource_accounting.csv",
        [
            "condition", "status", "achieved_sparsity", "nominal_bits",
            "dense_fp16_total_bytes", "sparse_16bit_total_bytes",
            "serialized_packed_total_bytes", "eligible_metadata_bytes",
            "compensation_scale_bytes", "prefill_512_mean_ms",
            "prefill_2048_mean_ms", "decode_mean_ms", "decode_tokens",
            "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes", "total_wall_sec",
        ],
        resource_rows,
    )

    # Paired 95% bootstrap CIs for P3 - P2 and Q3 - Q2
    bootstrap_results: dict[str, Any] = {
        "seed": int(args.seed),
        "n_resamples": int(args.bootstrap_resamples),
        "comparisons": {},
    }
    for pair_key, cond_a, cond_b in BOOTSTRAP_PAIRS:
        wt2_a = results_dir / f"ppl_windows_wikitext2_{cond_a}.jsonl"
        wt2_b = results_dir / f"ppl_windows_wikitext2_{cond_b}.jsonl"
        c4_a = results_dir / f"ppl_windows_c4_{cond_a}.jsonl"
        c4_b = results_dir / f"ppl_windows_c4_{cond_b}.jsonl"
        hs_a = results_dir / f"hellaswag_per_example_{cond_a}.jsonl"
        hs_b = results_dir / f"hellaswag_per_example_{cond_b}.jsonl"

        if not (wt2_a.exists() and wt2_b.exists() and c4_a.exists() and c4_b.exists()):
            bootstrap_results["comparisons"][pair_key] = {
                "status": "pending",
                "cond_a": cond_a,
                "cond_b": cond_b,
            }
            continue

        entry: dict[str, Any] = {
            "status": "completed",
            "cond_a": cond_a,
            "cond_b": cond_b,
            "wikitext2_ppl": paired_bootstrap_ppl_diff(
                read_jsonl(wt2_a),
                read_jsonl(wt2_b),
                n_resamples=args.bootstrap_resamples,
                seed=args.seed,
            ),
            "c4_validation_ppl": paired_bootstrap_ppl_diff(
                read_jsonl(c4_a),
                read_jsonl(c4_b),
                n_resamples=args.bootstrap_resamples,
                seed=args.seed + 1,
            ),
        }
        if hs_a.exists() and hs_b.exists():
            rec_a = read_jsonl(hs_a)
            rec_b = read_jsonl(hs_b)
            entry["hellaswag_acc_norm"] = paired_bootstrap_acc_diff(
                rec_a,
                rec_b,
                metric_key="correct_norm",
                n_resamples=args.bootstrap_resamples,
                seed=args.seed + 2,
            )
            entry["hellaswag_acc_raw"] = paired_bootstrap_acc_diff(
                rec_a,
                rec_b,
                metric_key="correct_raw",
                n_resamples=args.bootstrap_resamples,
                seed=args.seed + 3,
            )
        bootstrap_results["comparisons"][pair_key] = entry

    atomic_write_json(results_dir / "bootstrap_ci_p3_q3.json", bootstrap_results)

    # Qualitative cases: fixed selection of HellaSwag examples across key conditions
    qual_conds = [
        c
        for c in (
            "fp16_baseline",
            "p1_magnitude_s50",
            "p2_wanda_s50",
            "p3_nonuniform_s50",
            "q1_uniform_int4",
            "q2_awq_int4",
            "q3_mixed_int4",
            "q2_awq_int3",
            "q3_mixed_int3",
        )
        if (results_dir / f"hellaswag_per_example_{c}.jsonl").exists()
    ]
    qual_cases: list[dict[str, Any]] = []
    if qual_conds:
        per_cond_hs = {
            c: read_jsonl(results_dir / f"hellaswag_per_example_{c}.jsonl")
            for c in qual_conds
        }
        min_len = min(len(v) for v in per_cond_hs.values())
        _, raw_hs = load_hellaswag_val(args.data_dir, limit=min_len)

        # Deterministic selection: up to 5 cases where methods disagree + up to 3 where all agree
        disagree_idx: list[int] = []
        agree_idx: list[int] = []
        for idx in range(min_len):
            preds = {per_cond_hs[c][idx]["pred_norm"] for c in qual_conds}
            if len(preds) > 1 and len(disagree_idx) < 5:
                disagree_idx.append(idx)
            elif len(preds) == 1 and len(agree_idx) < 3:
                agree_idx.append(idx)
            if len(disagree_idx) >= 5 and len(agree_idx) >= 3:
                break

        for idx in disagree_idx + agree_idx:
            ctx, endings, label, meta = format_hellaswag_example(raw_hs[idx])
            qual_cases.append(
                {
                    "eval_index": idx,
                    "order": meta["order"],
                    "source_row_index": meta["source_row_index"],
                    "context": ctx,
                    "endings": endings,
                    "gold_label": label,
                    "selection_reason": "method_disagreement" if idx in disagree_idx else "consensus",
                    "per_condition_scores": {
                        c: {
                            "log_likelihoods": per_cond_hs[c][idx]["log_likelihoods"],
                            "norm_log_likelihoods": per_cond_hs[c][idx]["norm_log_likelihoods"],
                            "pred_norm": per_cond_hs[c][idx]["pred_norm"],
                            "correct_norm": per_cond_hs[c][idx]["correct_norm"],
                        }
                        for c in qual_conds
                    },
                }
            )

    atomic_write_json(
        dirs["evidence"] / "qualitative_cases.json",
        {
            "selection_rule": (
                "Deterministic first up to 5 HellaSwag examples where compressed "
                "conditions disagree on pred_norm plus up to 3 consensus examples."
            ),
            "conditions_included": qual_conds,
            "cases": qual_cases,
        },
    )

    saved_plots = _generate_plots(dirs, cond_docs)
    bundle = export_final_report_bundle(
        dirs["root"],
        rel_items=[
            "configs/env_metadata.json",
            "results/summary_all_conditions.csv",
            "results/resource_accounting.csv",
            "results/bootstrap_ci_p3_q3.json",
            "results/structured_2to4_diagnostic.json",
            "results/p3_allocation_and_sensitivity.json",
            "results/q3_allocation_and_sensitivity.json",
            "results/condition_status.json",
            "evidence/qualitative_cases.json",
            "plots/",
        ],
    )
    print(
        f"[Task 1 Analysis] Wrote summary_all_conditions.csv, resource_accounting.csv, "
        f"bootstrap_ci_p3_q3.json, qualitative_cases.json, {len(saved_plots)} plot(s), "
        f"and exported {len(bundle['copied_files'])} final file(s) to {bundle['bundle_root']}.",
        flush=True,
    )
    return {
        "conditions_found": sorted(cond_docs.keys()),
        "plots": saved_plots,
        "final_report_bundle": bundle,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    analyze_task1(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
