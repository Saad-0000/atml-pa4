"""Task 2 Part B: MQAR Training (B1, B2, B3) with Sparse Rolling Checkpoints & Sweep.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task2.run_part_b --stage b1 --seeds 6304 6305 6306 --ckpt-every 1500 --resume --data-dir <DATA_DIR> --out-dir <T2_OUT>
  python -m task2.run_part_b --stage b2 --seeds 6304 6305 6306 --ckpt-every 1500 --resume --data-dir <DATA_DIR> --out-dir <T2_OUT>
  python -m task2.run_part_b --stage b3 --seeds 6304 6305 6306 --ckpt-every 1500 --resume --data-dir <DATA_DIR> --out-dir <T2_OUT>
  python -m task2.run_part_b --stage sweep --out-dir <T2_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import random
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from common.env import collect_env_metadata
from common.io import (
    atomic_write_json,
    atomic_write_text,
    ensure_task_dirs,
    read_json,
    resolve_smoke_out_dir,
    update_condition_status,
)
from common.seeding import DEFAULT_SEED, set_seed
from common.timing import measure_latency
from task2.mqar_data import (
    DISTANCE_BINS,
    VALUE_OFFSET,
    distance_bin_label,
    load_mqar_generator,
    load_mqar_split,
    verify_mqar_invariants,
)
from task2.mqar_model import ATTN_MODES, MQARTransformer, theoretical_attention_cost

FIXED_BATCH_SIZE = 32
FIXED_TOTAL_UPDATES = 4500
FIXED_WARMUP_UPDATES = 200
FIXED_PEAK_LR = 8e-4
FIXED_MIN_LR_FRAC = 0.15
FIXED_WEIGHT_DECAY = 0.01
FIXED_GRAD_CLIP = 1.0
FIXED_EVAL_STEPS = (500, 1000, 1500, 2000, 3000, 4500)
SWEEP_LENGTHS = (256, 1024, 4096, 8192)

B3_INTERVENTION_SPEC = {
    "name": "b3_local_prefix",
    "description": (
        "Hybrid Local + Prefix-Sink Sparse Attention: 2 heads attend to the preceding "
        "64 positions (W=64); 2 heads attend to 32 global prefix anchors [0, 32) plus "
        "32 recent positions [i-31, i], respecting causality."
    ),
    "matched_constraints": {
        "extra_trainable_params_pct": 0.0,
        "max_kv_positions_per_query_per_head": 64,
        "extra_persistent_state_bytes": 0,
    },
    "pre_eval_prediction": (
        "Because MQAR places K<=16 key-value pairs in [0, 32) and far queries span "
        "[128, 250] including non-stride-4 offsets, b3_local_prefix should substantially "
        "outperform window64_mixed (receptive-field limit 2*(W-1)=126) and "
        "local_dilated_mixed (stride-4 alignment blind spots) on far and mixed regimes "
        "at matched 64-position budget and identical parameter count."
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 2 Part B: MQAR training (b1, b2, b3) and forward inference sweep."
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=("b1", "b2", "b3", "sweep"),
        help="Stage to run: b1, b2, b3, or sweep.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[6304, 6305, 6306])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ckpt-every", type=int, default=1500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


def _lr_for_step(step: int, total_updates: int, warmup: int) -> float:
    if step <= warmup:
        return FIXED_PEAK_LR * (step / max(warmup, 1))
    progress = (step - warmup) / max(total_updates - warmup, 1)
    progress = min(1.0, max(0.0, progress))
    return FIXED_PEAK_LR * (1.0 - progress * (1.0 - FIXED_MIN_LR_FRAC))


def _collate_examples(
    examples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack batch into (tokens [B,T], query_pos [B,K], targets [B,K] in 0..63)."""
    tokens = torch.tensor([e["tokens"] for e in examples], dtype=torch.long, device=device)
    q_pos = torch.tensor([e["query_positions"] for e in examples], dtype=torch.long, device=device)
    targets = torch.tensor(
        [[a - VALUE_OFFSET for a in e["answer_ids"]] for e in examples],
        dtype=torch.long,
        device=device,
    )
    return tokens, q_pos, targets


def evaluate_mqar_split(
    model: MQARTransformer,
    examples: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 64,
    override_attn_mode: str | None = None,
) -> dict[str, Any]:
    """Evaluate accuracy and cross-entropy loss across supervised query positions + distance bins."""
    model.eval()
    total_loss = 0.0
    total_queries = 0
    total_correct = 0

    bin_correct = {lbl: 0 for lbl, _, _ in DISTANCE_BINS}
    bin_total = {lbl: 0 for lbl, _, _ in DISTANCE_BINS}

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            tokens, q_pos, targets = _collate_examples(batch, device)
            logits = model(tokens, override_attn_mode=override_attn_mode)  # [B, T, 64]
            # Gather logits at marked query positions [B, K, 64]
            idx = q_pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
            q_logits = torch.gather(logits, 1, idx)
            loss = F.cross_entropy(
                q_logits.reshape(-1, q_logits.shape[-1]).float(),
                targets.reshape(-1),
                reduction="sum",
            )
            preds = q_logits.argmax(dim=-1)
            correct_mat = (preds == targets).cpu()

            total_loss += float(loss.item())
            total_queries += int(targets.numel())
            total_correct += int(correct_mat.sum().item())

            for bi, ex in enumerate(batch):
                for qi, d in enumerate(ex["distances"]):
                    lbl = distance_bin_label(int(d))
                    if lbl in bin_total:
                        bin_total[lbl] += 1
                        bin_correct[lbl] += int(correct_mat[bi, qi].item())

    return {
        "num_examples": len(examples),
        "num_queries": int(total_queries),
        "loss": float(total_loss / max(total_queries, 1)),
        "accuracy": float(total_correct / max(total_queries, 1)),
        "distance_bins": {
            lbl: {
                "correct": int(bin_correct[lbl]),
                "total": int(bin_total[lbl]),
                "accuracy": float(bin_correct[lbl] / bin_total[lbl]) if bin_total[lbl] > 0 else None,
            }
            for lbl, _, _ in DISTANCE_BINS
        },
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    torch.save(payload, tmp)
    tmp.replace(path)


def _train_single_run(
    run_id: str,
    attn_mode: str,
    k_pairs: int,
    train_regime: str,
    seed: int,
    make_example,
    dev_by_k_regime: dict[tuple[int, str], list[dict[str, Any]]],
    test_by_k_regime: dict[tuple[int, str], list[dict[str, Any]]],
    dirs: dict[str, Path],
    device: torch.device,
    total_updates: int,
    warmup_updates: int,
    eval_steps: tuple[int, ...],
    ckpt_every: int,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    res_path = dirs["results"] / f"partB_run_{run_id}.json"
    ckpt_dir = dirs["checkpoints"] / run_id
    latest_ckpt = ckpt_dir / "latest.pt"
    final_ckpt = ckpt_dir / "final.pt"

    if resume and res_path.exists() and final_ckpt.exists():
        try:
            existing = read_json(res_path)
            if existing.get("status") == "completed" and existing.get("seed") == seed:
                print(f"[RESUME] Skipping completed MQAR run {run_id}.", flush=True)
                return existing
        except Exception:
            pass

    set_seed(seed)
    model = MQARTransformer(attn_mode=attn_mode).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FIXED_PEAK_LR, weight_decay=FIXED_WEIGHT_DECAY
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    stream_idx = 0
    skipped_overflows = 0
    learning_curve: list[dict[str, Any]] = []
    train_elapsed_sec = 0.0

    if resume and latest_ckpt.exists():
        try:
            state = torch.load(latest_ckpt, map_location=device, weights_only=False)
            if state.get("run_id") == run_id and state.get("seed") == seed:
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                if use_amp and "scaler" in state:
                    scaler.load_state_dict(state["scaler"])
                step = int(state["step"])
                stream_idx = int(state["stream_idx"])
                skipped_overflows = int(state.get("skipped_overflows", 0))
                learning_curve = list(state.get("learning_curve", []))
                train_elapsed_sec = float(state.get("train_elapsed_sec", 0.0))
                random.setstate(state["py_rng"])
                np.random.set_state(state["np_rng"])
                torch.set_rng_state(state["torch_rng"].cpu())
                if torch.cuda.is_available() and state.get("cuda_rng") is not None:
                    torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
                print(
                    f"[RESUME] Restored {run_id} from step {step}/{total_updates} (stream_idx={stream_idx}).",
                    flush=True,
                )
        except Exception as exc:
            print(f"[WARN] Could not load {latest_ckpt} ({exc}); starting fresh.", flush=True)

    # Deterministic per-(run, seed) stream offset so seeds see distinct training streams
    seed_stream_offset = (seed - 6304) * 500_000 + k_pairs * 10_000
    eval_set = set(eval_steps)

    print(
        f"[Task 2B] Training {run_id} (mode={attn_mode}, K={k_pairs}, regime={train_regime}, "
        f"seed={seed}, updates={step}->{total_updates})...",
        flush=True,
    )
    t_run0 = time.perf_counter()

    while step < total_updates:
        batch_examples = [
            make_example("train", k_pairs, train_regime, seed_stream_offset + stream_idx + i)
            for i in range(FIXED_BATCH_SIZE)
        ]
        tokens, q_pos, targets = _collate_examples(batch_examples, device)

        # Retry the same batch if GradScaler overflows, without consuming the update budget
        updated = False
        while not updated:
            lr = _lr_for_step(step + 1, total_updates, warmup_updates)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(tokens)
                idx = q_pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
                q_logits = torch.gather(logits, 1, idx)
                loss = F.cross_entropy(
                    q_logits.reshape(-1, q_logits.shape[-1]).float(),
                    targets.reshape(-1),
                )

            scale_before = scaler.get_scale() if use_amp else 1.0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), FIXED_GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            if use_amp and scaler.get_scale() < scale_before:
                skipped_overflows += 1
            else:
                updated = True

        step += 1
        stream_idx += FIXED_BATCH_SIZE

        if step in eval_set:
            dev_metrics = {}
            for reg in ("near", "far", "mixed"):
                ex_list = dev_by_k_regime[(k_pairs, reg)]
                m = evaluate_mqar_split(model, ex_list, device)
                dev_metrics[reg] = {"accuracy": m["accuracy"], "loss": m["loss"]}
            curve_pt = {
                "run_id": run_id,
                "attn_mode": attn_mode,
                "K": int(k_pairs),
                "train_regime": train_regime,
                "seed": int(seed),
                "step": int(step),
                "train_loss": float(loss.item()),
                "lr": float(lr),
                "dev_near_acc": dev_metrics["near"]["accuracy"],
                "dev_far_acc": dev_metrics["far"]["accuracy"],
                "dev_mixed_acc": dev_metrics["mixed"]["accuracy"],
                "dev_mixed_loss": dev_metrics["mixed"]["loss"],
            }
            learning_curve.append(curve_pt)
            print(
                f"  [{run_id}] step={step:4d} loss={loss.item():.4f} | "
                f"dev near={curve_pt['dev_near_acc']:.4f} far={curve_pt['dev_far_acc']:.4f} "
                f"mixed={curve_pt['dev_mixed_acc']:.4f}",
                flush=True,
            )

        if ckpt_every > 0 and step < total_updates and (step % ckpt_every == 0):
            _atomic_torch_save(
                {
                    "run_id": run_id,
                    "seed": int(seed),
                    "step": int(step),
                    "stream_idx": int(stream_idx),
                    "skipped_overflows": int(skipped_overflows),
                    "train_elapsed_sec": float(train_elapsed_sec + (time.perf_counter() - t_run0)),
                    "learning_curve": learning_curve,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "py_rng": random.getstate(),
                    "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
                },
                latest_ckpt,
            )

    train_elapsed_sec += time.perf_counter() - t_run0

    # Save final checkpoint and remove rolling latest.pt after verification
    _atomic_torch_save(
        {
            "run_id": run_id,
            "attn_mode": attn_mode,
            "K": int(k_pairs),
            "train_regime": train_regime,
            "seed": int(seed),
            "step": int(step),
            "model": model.state_dict(),
        },
        final_ckpt,
    )
    if latest_ckpt.exists() and final_ckpt.exists() and final_ckpt.stat().st_size > 0:
        latest_ckpt.unlink()

    # Final-test evaluation across all three regimes (near, far, mixed)
    test_results = {}
    for reg in ("near", "far", "mixed"):
        test_results[reg] = evaluate_mqar_split(model, test_by_k_regime[(k_pairs, reg)], device)

    # B1 test-time isolation control for dense_mixed: impose W=64 mask at test time only
    test_window64_control = None
    if attn_mode == "dense" and train_regime == "mixed":
        test_window64_control = {}
        for reg in ("near", "far", "mixed"):
            test_window64_control[reg] = evaluate_mqar_split(
                model,
                test_by_k_regime[(k_pairs, reg)],
                device,
                override_attn_mode="window64",
            )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    payload = {
        "status": "completed",
        "run_id": run_id,
        "attn_mode": attn_mode,
        "K": int(k_pairs),
        "train_regime": train_regime,
        "seed": int(seed),
        "smoke": bool(smoke),
        "trainable_params": int(trainable_params),
        "total_updates": int(step),
        "skipped_overflows": int(skipped_overflows),
        "train_elapsed_sec": float(train_elapsed_sec),
        "chance_label_space": 1.0 / 64.0,
        "chance_in_context": 1.0 / float(k_pairs),
        "theoretical_cost_T256": theoretical_attention_cost(attn_mode, seq_len=256),
        "learning_curve": learning_curve,
        "test_results": test_results,
        "test_time_window64_control": test_window64_control,
    }
    atomic_write_json(res_path, payload)
    return payload


def rebuild_part_b_csvs(results_dir: Path) -> None:
    """Aggregate all saved partB_run_*.json into the required CSV tables."""
    run_files = sorted(results_dir.glob("partB_run_*.json"))
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []

    for rf in run_files:
        doc = read_json(rf)
        for pt in doc.get("learning_curve", []):
            curve_rows.append(pt)

        def _append_eval(eval_label: str, attn_eval: str, res_map: dict[str, Any]) -> None:
            summary_rows.append(
                {
                    "run_id": eval_label,
                    "attn_mode": attn_eval,
                    "K": doc["K"],
                    "train_regime": doc["train_regime"],
                    "seed": doc["seed"],
                    "trainable_params": doc["trainable_params"],
                    "train_elapsed_sec": round(doc["train_elapsed_sec"], 2),
                    "chance_1_over_64": doc["chance_label_space"],
                    "chance_1_over_K": doc["chance_in_context"],
                    "test_near_acc": res_map["near"]["accuracy"],
                    "test_far_acc": res_map["far"]["accuracy"],
                    "test_mixed_acc": res_map["mixed"]["accuracy"],
                    "test_mixed_loss": res_map["mixed"]["loss"],
                }
            )
            for reg in ("near", "far", "mixed"):
                for bin_lbl, stats in res_map[reg]["distance_bins"].items():
                    bin_rows.append(
                        {
                            "run_id": eval_label,
                            "attn_mode": attn_eval,
                            "K": doc["K"],
                            "train_regime": doc["train_regime"],
                            "seed": doc["seed"],
                            "test_regime": reg,
                            "distance_bin": bin_lbl,
                            "correct": stats["correct"],
                            "total": stats["total"],
                            "accuracy": "" if stats["accuracy"] is None else stats["accuracy"],
                        }
                    )

        _append_eval(doc["run_id"], doc["attn_mode"], doc["test_results"])
        if doc.get("test_time_window64_control") is not None:
            _append_eval(
                f"{doc['run_id']}__test_window64",
                "dense_test_window64",
                doc["test_time_window64_control"],
            )

    def _dump(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        atomic_write_text(results_dir / name, buf.getvalue())

    _dump("partB_learning_curves.csv", curve_rows)
    _dump("partB_mqar_accuracy_summary.csv", summary_rows)
    _dump("partB_distance_bins.csv", bin_rows)


def run_inference_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Forward-only inference sweep at T in {256, 1024, 4096, 8192}."""
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    lengths = (256, 512) if args.smoke else SWEEP_LENGTHS
    warmup = 1 if args.smoke else 3
    reps = 2 if args.smoke else 10
    bsz = 2

    rows: list[dict[str, Any]] = []
    for mode in ATTN_MODES:
        model = MQARTransformer(attn_mode=mode, max_len=max(lengths)).to(device)
        if device.type == "cuda":
            model = model.half()
        model.eval()
        dtype = next(model.parameters()).dtype

        for seq_len in lengths:
            theo = theoretical_attention_cost(mode, seq_len=seq_len)
            row: dict[str, Any] = {
                "attn_mode": mode,
                "seq_len": int(seq_len),
                "batch_size": bsz,
                "dtype": str(dtype),
                "status": "pending",
                "mean_ms": "",
                "std_ms": "",
                "peak_allocated_bytes": "",
                "peak_reserved_bytes": "",
                **theo,
                "implementation_note": (
                    "Materializes dense boolean attention mask [1, H, T, T]"
                    if mode in ("window64", "local_dilated", "b3_local_prefix")
                    else ("Chunked recurrent ELU+1 linear scan (chunk=256)" if mode == "linear_elu" else "PyTorch causal SDPA")
                ),
            }
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                x = torch.randn((bsz, seq_len, 128), device=device, dtype=dtype)

                def _fwd() -> torch.Tensor:
                    with torch.no_grad():
                        return model.forward_from_embeddings(x)

                stats = measure_latency(_fwd, warmup=warmup, reps=reps, device=device)
                row["status"] = "completed"
                row["mean_ms"] = round(stats["mean_ms"], 4)
                row["std_ms"] = round(stats["std_ms"], 4)
                row["peak_allocated_bytes"] = stats["peak_allocated_bytes"]
                row["peak_reserved_bytes"] = stats["peak_reserved_bytes"]
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    row["status"] = "oom"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
            rows.append(row)
            print(
                f"  [Part B Sweep] {mode:16s} T={seq_len:4d} -> "
                f"{row['status']} {row['mean_ms']}ms peak_alloc={row['peak_allocated_bytes']}",
                flush=True,
            )

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    atomic_write_text(dirs["results"] / "partB_inference_sweep.csv", buf.getvalue())
    update_condition_status(
        dirs["results"] / "condition_status.json",
        "partB_inference_sweep",
        "completed",
        {"seed": int(args.seed), "smoke": bool(args.smoke), "rows": len(rows)},
    )
    return rows


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage == "sweep":
        return {"sweep": run_inference_sweep(args)}

    if not args.data_dir:
        raise ValueError("--data-dir is required for Part B training stages (b1, b2, b3).")

    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    status_path = dirs["results"] / "condition_status.json"
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    dev_path, dev_all = load_mqar_split(args.data_dir, "development")
    test_path, test_all = load_mqar_split(args.data_dir, "test")
    make_example, gen_source = load_mqar_generator(args.data_dir)

    # Verify generator & frozen split invariants before training
    gen_samples = [
        make_example("train", k, reg, idx)
        for k in (4, 8, 16)
        for reg in ("near", "far", "mixed")
        for idx in range(4)
    ]
    inv_gen = verify_mqar_invariants(gen_samples)
    inv_dev = verify_mqar_invariants(dev_all[:64])

    atomic_write_json(
        dirs["configs"] / f"partB_{args.stage}_config.json",
        {
            "stage": args.stage,
            "seeds": [int(s) for s in args.seeds],
            "smoke": bool(args.smoke),
            "device": str(device),
            "generator_source": gen_source,
            "dev_path": str(dev_path),
            "test_path": str(test_path),
            "invariant_checks": {"generator": inv_gen, "dev_sample": inv_dev},
            "b3_intervention": B3_INTERVENTION_SPEC if args.stage == "b3" else None,
        },
    )

    dev_by_k_reg: dict[tuple[int, str], list[dict[str, Any]]] = {}
    test_by_k_reg: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for ex in dev_all:
        dev_by_k_reg.setdefault((int(ex["K"]), ex["regime"]), []).append(ex)
    for ex in test_all:
        test_by_k_reg.setdefault((int(ex["K"]), ex["regime"]), []).append(ex)

    if args.smoke:
        seeds = [int(args.seeds[0])]
        total_updates = 40
        warmup_updates = 10
        eval_steps = (20, 40)
        ckpt_every = 20
        for key in dev_by_k_reg:
            dev_by_k_reg[key] = dev_by_k_reg[key][:32]
        for key in test_by_k_reg:
            test_by_k_reg[key] = test_by_k_reg[key][:32]
    else:
        seeds = [int(s) for s in args.seeds]
        total_updates = FIXED_TOTAL_UPDATES
        warmup_updates = FIXED_WARMUP_UPDATES
        eval_steps = FIXED_EVAL_STEPS
        ckpt_every = int(args.ckpt_every)

    # Exact grid from ATML PA4 PDF p. 9
    if args.stage == "b1":
        specs = [
            ("dense", 4, "mixed"),
            ("dense", 8, "mixed"),
            ("dense", 16, "mixed"),
            ("dense", 16, "near"),
            ("window64", 4, "near"),
            ("window64", 8, "near"),
            ("window64", 16, "near"),
            ("window64", 16, "mixed"),
        ]
        if args.smoke:
            specs = [("dense", 4, "mixed"), ("window64", 4, "near")]
    elif args.stage == "b2":
        specs = [
            ("linear_elu", 4, "mixed"),
            ("linear_elu", 8, "mixed"),
            ("linear_elu", 16, "mixed"),
            ("local_dilated", 4, "mixed"),
            ("local_dilated", 16, "mixed"),
        ]
        if args.smoke:
            specs = [("linear_elu", 4, "mixed"), ("local_dilated", 4, "mixed")]
    else:  # b3
        specs = [("b3_local_prefix", 16, "mixed")]

    outputs: dict[str, Any] = {}
    for attn_mode, k_pairs, train_regime in specs:
        for seed in seeds:
            run_id = f"{attn_mode}_K{k_pairs}_{train_regime}_s{seed}"
            res = _train_single_run(
                run_id=run_id,
                attn_mode=attn_mode,
                k_pairs=k_pairs,
                train_regime=train_regime,
                seed=seed,
                make_example=make_example,
                dev_by_k_regime=dev_by_k_reg,
                test_by_k_regime=test_by_k_reg,
                dirs=dirs,
                device=device,
                total_updates=total_updates,
                warmup_updates=warmup_updates,
                eval_steps=eval_steps,
                ckpt_every=ckpt_every,
                resume=args.resume,
                smoke=args.smoke,
            )
            outputs[run_id] = res
            rebuild_part_b_csvs(dirs["results"])

    update_condition_status(
        status_path,
        f"partB_{args.stage}",
        "completed",
        {"seeds": seeds, "smoke": bool(args.smoke), "runs": sorted(outputs.keys())},
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_part_b(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
