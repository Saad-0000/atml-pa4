"""Task 2 Part A: Exact Attention Operators (A1 Dense, A2 Tiled, A3 SDPA).

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task2.run_part_a --out-dir <T2_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import torch

from common.env import collect_env_metadata
from common.io import (
    atomic_write_json,
    atomic_write_text,
    ensure_task_dirs,
    resolve_smoke_out_dir,
    update_condition_status,
)
from common.seeding import DEFAULT_SEED, set_seed
from common.timing import measure_latency
from task2.attention_ops import (
    dense_causal_attention,
    probe_sdpa_backends,
    sdpa_causal_attention,
    tiled_online_softmax_attention,
    verify_causality,
    verify_extreme_score_stability,
)

DEFAULT_LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
SMOKE_LENGTHS = (128, 256)
DEFAULT_DTYPES = ("fp16", "fp32")
OPERATORS = ("dense", "tiled", "sdpa")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 2 Part A: Fidelity, gradient error, forward/backward latency, and "
            "memory benchmark for A1 Dense, A2 Tiled Online-Softmax, and A3 PyTorch SDPA."
        )
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--lengths", type=int, nargs="+", default=None)
    parser.add_argument("--warmup-reps", type=int, default=None)
    parser.add_argument("--timing-reps", type=int, default=None)
    return parser


def _make_qkv(
    bsz: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed + seq_len)
    q = torch.randn((bsz, n_heads, seq_len, head_dim), generator=g).to(device=device, dtype=dtype)
    k = torch.randn((bsz, n_heads, seq_len, head_dim), generator=g).to(device=device, dtype=dtype)
    v = torch.randn((bsz, n_heads, seq_len, head_dim), generator=g).to(device=device, dtype=dtype)
    grad_out = torch.randn((bsz, n_heads, seq_len, head_dim), generator=g).to(device=device, dtype=dtype)
    return q, k, v, grad_out


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _run_op(op_name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tile_size: int) -> torch.Tensor:
    if op_name == "dense":
        return dense_causal_attention(q, k, v)
    if op_name == "tiled":
        return tiled_online_softmax_attention(q, k, v, tile_q=tile_size, tile_kv=tile_size)
    if op_name == "sdpa":
        return sdpa_causal_attention(q, k, v)
    raise ValueError(f"Unknown operator {op_name}")


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    status_path = dirs["results"] / "condition_status.json"

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    lengths = (
        list(args.lengths)
        if args.lengths is not None
        else list(SMOKE_LENGTHS if args.smoke else DEFAULT_LENGTHS)
    )
    warmup = args.warmup_reps if args.warmup_reps is not None else (1 if args.smoke else 3)
    reps = args.timing_reps if args.timing_reps is not None else (2 if args.smoke else 10)
    dtypes = ["fp32"] if device.type == "cpu" else list(DEFAULT_DTYPES)

    atomic_write_json(
        dirs["configs"] / "partA_env_metadata.json",
        collect_env_metadata(
            seed=args.seed,
            extra={
                "stage": "task2_partA",
                "smoke": bool(args.smoke),
                "device": str(device),
                "lengths": lengths,
                "dtypes": dtypes,
                "tile_size": int(args.tile_size),
                "warmup_reps": warmup,
                "timing_reps": reps,
            },
        ),
    )

    # 1. Causality, edge-case stability, multi-tile agreement, and SDPA backend probes
    causality_checks = {
        op: verify_causality(
            lambda q, k, v, o=op: _run_op(o, q, k, v, args.tile_size),
            device=device,
            dtype=torch.float32,
        )
        for op in OPERATORS
    }
    extreme_checks = {
        op: verify_extreme_score_stability(
            lambda q, k, v, o=op: _run_op(o, q, k, v, args.tile_size),
            device=device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        for op in OPERATORS
    }

    # Verify tiled attention across tile sizes 64 and 128 against dense FP32
    q_t, k_t, v_t, _ = _make_qkv(1, 2, 192, 32, device, torch.float32, args.seed + 99)
    ref_t = dense_causal_attention(q_t, k_t, v_t)
    tile_size_checks = {}
    for ts in (32, 64, 128):
        out_ts = tiled_online_softmax_attention(q_t, k_t, v_t, tile_q=ts, tile_kv=ts)
        tile_size_checks[str(ts)] = {"max_abs_err_vs_dense_fp32": _max_abs_err(out_ts, ref_t)}

    backend_probes: dict[str, Any] = {}
    for dt_name in dtypes:
        dt = torch.float16 if dt_name == "fp16" else torch.float32
        backend_probes[dt_name] = probe_sdpa_backends(
            device=device,
            dtype=dt,
            bsz=args.batch_size,
            n_heads=args.num_heads,
            seq_len=min(256, lengths[0]),
            head_dim=args.head_dim,
        )

    backends_doc = {
        "device": str(device),
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "causality_checks": causality_checks,
        "extreme_score_checks": extreme_checks,
        "tiled_multi_tile_checks": tile_size_checks,
        "sdpa_backend_probes": backend_probes,
        "tiled_backward_note": (
            "Python-loop tiled online-softmax backward relies on PyTorch autograd, "
            "retaining per-tile intermediates rather than recomputing them in SRAM "
            "like a fused FlashAttention CUDA kernel."
        ),
    }
    atomic_write_json(dirs["results"] / "partA_backends.json", backends_doc)

    # 2. Sweep over all (L, dtype, operator)
    rows: list[dict[str, Any]] = []
    for seq_len in lengths:
        # Compute dense FP32 reference outputs & gradients where feasible
        ref_out = ref_dq = ref_dk = ref_dv = None
        ref_feasible = seq_len <= 4096
        if ref_feasible:
            try:
                _cleanup_cuda()
                qr, kr, vr, gr = _make_qkv(
                    args.batch_size, args.num_heads, seq_len, args.head_dim,
                    device, torch.float32, args.seed,
                )
                qr.requires_grad_(True)
                kr.requires_grad_(True)
                vr.requires_grad_(True)
                ref_out = dense_causal_attention(qr, kr, vr)
                ref_out.backward(gr)
                ref_out = ref_out.detach().cpu()
                ref_dq = qr.grad.detach().cpu()
                ref_dk = kr.grad.detach().cpu()
                ref_dv = vr.grad.detach().cpu()
                del qr, kr, vr, gr
                _cleanup_cuda()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    ref_feasible = False
                    _cleanup_cuda()
                else:
                    raise

        for dt_name in dtypes:
            dt = torch.float16 if dt_name == "fp16" else torch.float32
            dispatched = backend_probes[dt_name].get("dispatched_default_guess", "unknown")

            for op_name in OPERATORS:
                cond_key = f"partA_{op_name}_{dt_name}_L{seq_len}"
                row: dict[str, Any] = {
                    "operator": op_name,
                    "dtype": dt_name,
                    "seq_len": int(seq_len),
                    "batch_size": int(args.batch_size),
                    "num_heads": int(args.num_heads),
                    "head_dim": int(args.head_dim),
                    "tile_size": int(args.tile_size) if op_name == "tiled" else "",
                    "sdpa_backend": dispatched if op_name == "sdpa" else "n/a",
                    "timed_reps": int(reps),
                    "status": "pending",
                    "out_max_abs_err": "",
                    "dq_max_abs_err": "",
                    "dk_max_abs_err": "",
                    "dv_max_abs_err": "",
                    "fwd_mean_ms": "",
                    "fwd_std_ms": "",
                    "fwd_peak_alloc_bytes": "",
                    "fwd_peak_res_bytes": "",
                    "fwdbwd_mean_ms": "",
                    "fwdbwd_std_ms": "",
                    "fwdbwd_peak_alloc_bytes": "",
                    "fwdbwd_peak_res_bytes": "",
                    "error_note": "",
                }

                # Skip obviously infeasible Python-loop tiled backward OOM only if it actually OOMs
                try:
                    _cleanup_cuda()
                    q, k, v, grad_out = _make_qkv(
                        args.batch_size, args.num_heads, seq_len, args.head_dim,
                        device, dt, args.seed,
                    )

                    # Forward timing
                    def _fwd() -> torch.Tensor:
                        with torch.no_grad():
                            return _run_op(op_name, q, k, v, args.tile_size)

                    fwd_stats = measure_latency(_fwd, warmup=warmup, reps=reps, device=device)
                    row["fwd_mean_ms"] = round(fwd_stats["mean_ms"], 4)
                    row["fwd_std_ms"] = round(fwd_stats["std_ms"], 4)
                    row["fwd_peak_alloc_bytes"] = fwd_stats["peak_allocated_bytes"]
                    row["fwd_peak_res_bytes"] = fwd_stats["peak_reserved_bytes"]

                    if ref_out is not None:
                        with torch.no_grad():
                            out_check = _run_op(op_name, q, k, v, args.tile_size).detach().cpu()
                        row["out_max_abs_err"] = _max_abs_err(out_check, ref_out)
                        del out_check

                    # Forward + backward timing & gradient fidelity
                    try:
                        def _fwdbwd() -> None:
                            q_g = q.detach().requires_grad_(True)
                            k_g = k.detach().requires_grad_(True)
                            v_g = v.detach().requires_grad_(True)
                            o = _run_op(op_name, q_g, k_g, v_g, args.tile_size)
                            o.backward(grad_out)

                        fb_stats = measure_latency(_fwdbwd, warmup=warmup, reps=reps, device=device)
                        row["fwdbwd_mean_ms"] = round(fb_stats["mean_ms"], 4)
                        row["fwdbwd_std_ms"] = round(fb_stats["std_ms"], 4)
                        row["fwdbwd_peak_alloc_bytes"] = fb_stats["peak_allocated_bytes"]
                        row["fwdbwd_peak_res_bytes"] = fb_stats["peak_reserved_bytes"]

                        if ref_dq is not None:
                            q_g = q.detach().requires_grad_(True)
                            k_g = k.detach().requires_grad_(True)
                            v_g = v.detach().requires_grad_(True)
                            o = _run_op(op_name, q_g, k_g, v_g, args.tile_size)
                            o.backward(grad_out)
                            row["dq_max_abs_err"] = _max_abs_err(q_g.grad.detach().cpu(), ref_dq)
                            row["dk_max_abs_err"] = _max_abs_err(k_g.grad.detach().cpu(), ref_dk)
                            row["dv_max_abs_err"] = _max_abs_err(v_g.grad.detach().cpu(), ref_dv)
                            del q_g, k_g, v_g, o

                        row["status"] = "completed"
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            row["status"] = "completed_fwd_only_bwd_oom"
                            row["error_note"] = f"Backward OOM: {type(exc).__name__}"
                            _cleanup_cuda()
                        else:
                            raise

                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        row["status"] = "oom"
                        row["error_note"] = f"OOM: {type(exc).__name__}"
                        _cleanup_cuda()
                    else:
                        row["status"] = "failed"
                        row["error_note"] = f"{type(exc).__name__}: {exc}"
                        _cleanup_cuda()

                rows.append(row)
                print(
                    f"  [Part A] {op_name:5s} {dt_name:4s} L={seq_len:4d} -> "
                    f"status={row['status']} fwd={row['fwd_mean_ms']}ms "
                    f"fwdbwd={row['fwdbwd_mean_ms']}ms out_err={row['out_max_abs_err']}",
                    flush=True,
                )

    fieldnames = list(rows[0].keys()) if rows else []
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    atomic_write_text(dirs["results"] / "partA_fidelity_and_timing.csv", buf.getvalue())

    update_condition_status(
        status_path,
        "partA_attention_ops",
        "completed",
        {"seed": int(args.seed), "smoke": bool(args.smoke), "rows": len(rows)},
    )
    return {"rows": rows, "backends": backends_doc}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_part_a(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
