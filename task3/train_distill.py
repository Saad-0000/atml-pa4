"""Task 3 Steps 3 & 4: Off-Policy and On-Policy Forward-KL LoRA Distillation.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task3.train_distill --mode off_policy --seed 6304 --ckpt-every-tokens 10000 --resume --data-dir <DATA_DIR> --out-dir <T3_OUT>
  python -m task3.train_distill --mode on_policy --seed 6304 --ckpt-every-tokens 10000 --resume --data-dir <DATA_DIR> --out-dir <T3_OUT>
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import (
    atomic_write_json,
    atomic_write_text,
    ensure_task_dirs,
    read_json,
    read_jsonl,
    resolve_smoke_out_dir,
    update_condition_status,
)
from common.seeding import DEFAULT_SEED, set_seed
from common.timing import get_memory_stats, reset_peak_memory, sync_device
from task3.data import (
    DRAFT_MODEL_ID,
    TARGET_MODEL_ID,
    RoundRobinPromptSchedule,
    format_chat_prompt,
    get_eos_token_ids,
    load_task3_train_pool,
)
from task3.sampler import target_only_generate

FIXED_MATH_QUOTA = 30_000
FIXED_SHORT_QUOTA = 10_000
FIXED_ACCUM_TOKENS = 256
FIXED_LR = 3e-4
FIXED_WD = 0.01
FIXED_GRAD_CLIP = 1.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 3 Steps 3 & 4: Off-policy / On-policy forward-KL LoRA distillation."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=("off_policy", "on_policy"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ckpt-every-tokens", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


class LoRALinear(nn.Module):
    """Exact Hu et al. (2022) LoRA wrapper: frozen FP16 base + FP32 low-rank adapters.

    Avoids peft/torchao version conflicts on Colab while preserving identical
    r=8, alpha=16, dropout=0.0 math, parameter counts, and FP16 merging.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty((self.r, base.in_features), dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.r), dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)

    def merge_into_base(self) -> nn.Linear:
        with torch.no_grad():
            delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
            self.base.weight.data.copy_(
                (self.base.weight.data.float() + delta).to(dtype=self.base.weight.dtype)
            )
        return self.base


def _wrap_student_with_lora(student_base: torch.nn.Module) -> tuple[torch.nn.Module, int, int]:
    """Attach FP32 LoRA (r=8, alpha=16, dropout=0) on q/k/v/o_proj over frozen backbone."""
    for p in student_base.parameters():
        p.requires_grad_(False)

    target_set = set(LORA_TARGET_MODULES)
    for parent_name, parent in list(student_base.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in target_set and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, r=8, alpha=16.0))

    trainable = sum(int(p.numel()) for p in student_base.parameters() if p.requires_grad)
    total = sum(int(p.numel()) for p in student_base.parameters())
    return student_base, trainable, total


def _unwrap_and_merge_lora(model: torch.nn.Module) -> torch.nn.Module:
    """Merge every LoRALinear into its underlying nn.Linear and replace in-place."""
    for _, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, child_name, child.merge_into_base())
    return model


def _get_lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if "lora_" in k
    }


def _load_lora_state_dict(model: torch.nn.Module, lora_sd: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    del missing, unexpected


def _atomic_save_pt(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    torch.save(payload, tmp)
    tmp.replace(path)


def _merge_and_verify_adapter(
    peft_student: torch.nn.Module,
    fresh_backbone_fn,
    lora_state: dict[str, torch.Tensor],
    verify_ids: torch.Tensor,
    merged_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Merge LoRA adapter into a fresh FP16 0.5B backbone and verify logit equivalence."""
    peft_student.eval()
    with torch.no_grad():
        unmerged_logits = peft_student(input_ids=verify_ids, use_cache=False).logits.detach().float()

    t_merge0 = time.perf_counter()
    fresh_base = fresh_backbone_fn().to(device)
    fresh_lora, _, _ = _wrap_student_with_lora(fresh_base)
    _load_lora_state_dict(fresh_lora, lora_state)
    fresh_lora.eval()
    merged_model = _unwrap_and_merge_lora(fresh_lora)
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    merged_model = merged_model.to(dtype=target_dtype).eval()
    merge_sec = time.perf_counter() - t_merge0

    with torch.no_grad():
        merged_logits = merged_model(input_ids=verify_ids, use_cache=False).logits.detach().float()

    diff = (merged_logits - unmerged_logits).abs()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_dir)

    return {
        "merge_sec": float(merge_sec),
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "merged_dir": str(merged_dir),
    }


def _rebuild_distill_logs_csv(results_dir: Path) -> None:
    log_files = sorted(results_dir.glob("distill_log_*.json"))
    rows: list[dict[str, Any]] = []
    for lf in log_files:
        doc = read_json(lf)
        rows.extend(doc.get("steps", []))
    if not rows:
        return
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    atomic_write_text(results_dir / "distillation_training_logs.csv", buf.getvalue())


def train_distill(
    args: argparse.Namespace,
    teacher_model: torch.nn.Module | None = None,
    student_base_fn=None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    run_id = f"{args.mode}_s{args.seed}"
    ckpt_dir = dirs["checkpoints"] / run_id
    latest_ckpt = ckpt_dir / "latest_lora.pt"
    final_lora_path = ckpt_dir / "lora_final.pt"
    merged_dir = ckpt_dir / "merged_fp16"
    cost_path = dirs["results"] / "distillation_cost_accounting.json"
    merge_check_path = dirs["results"] / "merged_vs_unmerged_check.json"

    if args.resume and final_lora_path.exists() and merged_dir.exists() and (dirs["results"] / f"distill_log_{run_id}.json").exists():
        print(f"[RESUME] Distillation run {run_id} already completed; skipping.", flush=True)
        return read_json(dirs["results"] / f"distill_log_{run_id}.json")

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dt = torch.float16 if device.type == "cuda" else torch.float32

    math_pool, short_pool = load_task3_train_pool(args.data_dir)
    if args.smoke:
        math_pool = math_pool[:4]
        short_pool = short_pool[:4]
        math_quota = 48
        short_quota = 16
        accum_target = 32
        ckpt_every = 32
    else:
        math_quota = FIXED_MATH_QUOTA
        short_quota = FIXED_SHORT_QUOTA
        accum_target = FIXED_ACCUM_TOKENS
        ckpt_every = int(args.ckpt_every_tokens)

    # Load frozen teacher pool for off-policy mode
    teacher_pool_map: dict[str, dict[str, Any]] = {}
    if args.mode == "off_policy":
        pool_path = dirs["results"] / "teacher_rollout_pool.jsonl"
        if not pool_path.exists():
            raise FileNotFoundError(
                f"Off-policy requires teacher_rollout_pool.jsonl ({pool_path}). Run task3.prepare_teacher_pool first."
            )
        for rec in read_jsonl(pool_path):
            teacher_pool_map[str(rec["source_id"])] = rec

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, use_fast=True)
    if teacher_model is None:
        teacher_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, dtype=dt).to(device).eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
    if student_base_fn is None:
        def student_base_fn():
            return AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_ID, dtype=dt)

    eos_ids = get_eos_token_ids(tokenizer, teacher_model)
    student_base = student_base_fn().to(device)
    student, trainable_params, total_student_params = _wrap_student_with_lora(student_base)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=FIXED_LR,
        weight_decay=FIXED_WD,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    schedule = RoundRobinPromptSchedule(math_pool, short_pool, seed=args.seed)
    math_scored = 0
    short_scored = 0
    opt_updates = 0
    opt_attempts = 0
    skipped_overflows = 0
    rollout_count = 0
    next_ckpt_milestone = ckpt_every
    step_logs: list[dict[str, Any]] = []

    t_student_rollout_sec = 0.0
    t_teacher_score_sec = 0.0
    t_backward_sec = 0.0

    if args.resume and latest_ckpt.exists():
        try:
            st = torch.load(latest_ckpt, map_location=device, weights_only=False)
            if st.get("run_id") == run_id and st.get("seed") == args.seed:
                _load_lora_state_dict(student, st["lora_state"])
                optimizer.load_state_dict(st["optimizer"])
                if use_amp and st.get("scaler") is not None:
                    scaler.load_state_dict(st["scaler"])
                schedule.load_state_dict(st["schedule"])
                math_scored = int(st["math_scored"])
                short_scored = int(st["short_scored"])
                opt_updates = int(st["opt_updates"])
                opt_attempts = int(st["opt_attempts"])
                skipped_overflows = int(st["skipped_overflows"])
                rollout_count = int(st["rollout_count"])
                next_ckpt_milestone = int(st["next_ckpt_milestone"])
                step_logs = list(st["step_logs"])
                t_student_rollout_sec = float(st["t_student_rollout_sec"])
                t_teacher_score_sec = float(st["t_teacher_score_sec"])
                t_backward_sec = float(st["t_backward_sec"])
                random.setstate(st["py_rng"])
                np.random.set_state(st["np_rng"])
                torch.set_rng_state(st["torch_rng"].cpu())
                if torch.cuda.is_available() and st.get("cuda_rng") is not None:
                    torch.cuda.set_rng_state(st["cuda_rng"].cpu(), device)
                print(
                    f"[RESUME] Restored {run_id} at {math_scored + short_scored} tokens "
                    f"(math={math_scored}, short={short_scored}, updates={opt_updates}).",
                    flush=True,
                )
        except Exception as exc:
            print(f"[WARN] Could not resume from {latest_ckpt} ({exc}); starting fresh.", flush=True)

    reset_peak_memory(device)
    t_wall0 = time.perf_counter()
    print(
        f"[Task 3 Distill] Starting {run_id} (quota={math_quota}+{short_quota}={math_quota+short_quota} tokens, "
        f"LoRA trainable={trainable_params:,}/{total_student_params:,})...",
        flush=True,
    )

    while (math_scored < math_quota) or (short_scored < short_quota):
        # Gather rollouts for one accumulation batch (>= accum_target tokens, or final short batch)
        batch_items: list[dict[str, Any]] = []
        batch_tokens = 0
        batch_math_add = 0
        batch_short_add = 0

        while batch_tokens < accum_target and (
            (math_scored + batch_math_add < math_quota)
            or (short_scored + batch_short_add < short_quota)
        ):
            math_done = (math_scored + batch_math_add) >= math_quota
            short_done = (short_scored + batch_short_add) >= short_quota
            item = schedule.next_prompt(math_done=math_done, short_done=short_done)
            cat = item["category"]
            rem_cat = (math_quota - math_scored - batch_math_add) if cat == "math" else (short_quota - short_scored - batch_short_add)

            rollout_count += 1
            if args.mode == "off_policy":
                rec = teacher_pool_map[item["source_id"]]
                prompt_ids = list(rec["prompt_token_ids"])
                resp_ids = list(rec["response_token_ids"])
            else:
                chat_text = format_chat_prompt(tokenizer, item["prompt"])
                prompt_ids = tokenizer.encode(chat_text, add_special_tokens=False)
                cap = 16 if args.smoke else int(item["max_new_tokens"])
                student.eval()
                sync_device(device)
                t_r0 = time.perf_counter()
                gen_out = target_only_generate(
                    target_model=student,
                    prompt_ids=prompt_ids,
                    max_new_tokens=cap,
                    eos_ids=eos_ids,
                    device=device,
                    seed=int(args.seed) + rollout_count * 31,
                    temperature=1.0,
                )
                sync_device(device)
                t_student_rollout_sec += time.perf_counter() - t_r0
                resp_ids = gen_out["generated_ids"]

            if not resp_ids:
                continue
            # Truncate mid-response if quota is reached
            truncated_for_quota = len(resp_ids) > rem_cat
            used_resp = resp_ids[:rem_cat]
            n_sup = len(used_resp)
            if n_sup == 0:
                continue

            # Precompute teacher log-probs once per rollout (not redone on GradScaler overflow retry)
            full_ids = prompt_ids + used_resp
            inp_t = torch.tensor([full_ids], dtype=torch.long, device=device)
            p_len = len(prompt_ids)
            sync_device(device)
            t_ts0 = time.perf_counter()
            with torch.no_grad():
                t_logits = teacher_model(input_ids=inp_t, use_cache=False).logits[0, p_len - 1 : p_len - 1 + n_sup, :].float()
                t_log_probs = F.log_softmax(t_logits, dim=-1).detach()
            sync_device(device)
            t_teacher_score_sec += time.perf_counter() - t_ts0

            batch_items.append(
                {
                    "inp_t": inp_t,
                    "p_len": p_len,
                    "n_sup": n_sup,
                    "t_log_probs": t_log_probs,
                    "category": cat,
                    "truncated_for_quota": truncated_for_quota,
                }
            )
            batch_tokens += n_sup
            if cat == "math":
                batch_math_add += n_sup
            else:
                batch_short_add += n_sup

        if not batch_items:
            break

        # Perform forward-KL accumulation & optimizer step; retry exact same batch on GradScaler overflow
        updated = False
        batch_kl_val = 0.0
        while not updated:
            opt_attempts += 1
            student.train()
            optimizer.zero_grad(set_to_none=True)
            sync_device(device)
            t_b0 = time.perf_counter()
            total_kl_loss = 0.0

            for bi in batch_items:
                p_len = bi["p_len"]
                n_sup = bi["n_sup"]
                t_lp = bi["t_log_probs"]
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    s_logits = student(input_ids=bi["inp_t"], use_cache=False).logits[0, p_len - 1 : p_len - 1 + n_sup, :]
                s_lp = F.log_softmax(s_logits.float(), dim=-1)
                t_p = t_lp.exp()
                # Exact forward KL normalized by total supervised tokens in this optimizer update
                kl_sum = (t_p * (t_lp - s_lp)).sum()
                loss_part = kl_sum / float(batch_tokens)
                scaler.scale(loss_part).backward()
                total_kl_loss += float(loss_part.item())

            scale_before = scaler.get_scale() if use_amp else 1.0
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], FIXED_GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()
            sync_device(device)
            t_backward_sec += time.perf_counter() - t_b0

            if use_amp and scaler.get_scale() < scale_before:
                skipped_overflows += 1
            else:
                updated = True
                batch_kl_val = total_kl_loss

        opt_updates += 1
        math_scored += batch_math_add
        short_scored += batch_short_add
        total_scored = math_scored + short_scored

        step_logs.append(
            {
                "run_id": run_id,
                "mode": args.mode,
                "seed": int(args.seed),
                "update": int(opt_updates),
                "batch_tokens": int(batch_tokens),
                "math_tokens_cum": int(math_scored),
                "short_tokens_cum": int(short_scored),
                "total_tokens_cum": int(total_scored),
                "forward_kl": float(batch_kl_val),
                "lr": float(FIXED_LR),
                "skipped_overflows_cum": int(skipped_overflows),
            }
        )
        if opt_updates % 10 == 0 or total_scored >= (math_quota + short_quota):
            print(
                f"  [{run_id}] upd={opt_updates:3d} tokens={total_scored:5d}/{math_quota+short_quota} "
                f"(math={math_scored}, short={short_scored}) kl={batch_kl_val:.4f} skips={skipped_overflows}",
                flush=True,
            )

        if ckpt_every > 0 and total_scored >= next_ckpt_milestone and total_scored < (math_quota + short_quota):
            _atomic_save_pt(
                {
                    "run_id": run_id,
                    "seed": int(args.seed),
                    "lora_state": _get_lora_state_dict(student),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "schedule": schedule.state_dict(),
                    "math_scored": int(math_scored),
                    "short_scored": int(short_scored),
                    "opt_updates": int(opt_updates),
                    "opt_attempts": int(opt_attempts),
                    "skipped_overflows": int(skipped_overflows),
                    "rollout_count": int(rollout_count),
                    "next_ckpt_milestone": int(next_ckpt_milestone + ckpt_every),
                    "step_logs": step_logs,
                    "t_student_rollout_sec": float(t_student_rollout_sec),
                    "t_teacher_score_sec": float(t_teacher_score_sec),
                    "t_backward_sec": float(t_backward_sec),
                    "py_rng": random.getstate(),
                    "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
                },
                latest_ckpt,
            )
            next_ckpt_milestone += ckpt_every

    peak_mem = get_memory_stats(device)
    wall_sec = time.perf_counter() - t_wall0

    # Save final LoRA adapter, merge into fresh FP16 0.5B backbone, verify logits, clean latest_lora.pt
    lora_final = _get_lora_state_dict(student)
    _atomic_save_pt({"run_id": run_id, "mode": args.mode, "seed": int(args.seed), "lora_state": lora_final}, final_lora_path)
    verify_ids = torch.tensor(
        [tokenizer.encode(format_chat_prompt(tokenizer, math_pool[0]["prompt"]), add_special_tokens=False)[:64]],
        dtype=torch.long,
        device=device,
    )
    merge_rep = _merge_and_verify_adapter(
        student, student_base_fn, lora_final, verify_ids, merged_dir, device
    )
    if latest_ckpt.exists() and final_lora_path.exists() and merged_dir.exists():
        latest_ckpt.unlink()

    # Update merged_vs_unmerged_check.json & distillation_cost_accounting.json
    merge_doc = read_json(merge_check_path) if merge_check_path.exists() else {}
    merge_doc[run_id] = merge_rep
    atomic_write_json(merge_check_path, merge_doc)

    teacher_pool_sec = 0.0
    if args.mode == "off_policy" and (dirs["results"] / "teacher_pool_cost.json").exists():
        teacher_pool_sec = float(read_json(dirs["results"] / "teacher_pool_cost.json").get("total_teacher_gen_sec", 0.0))

    cost_doc = read_json(cost_path) if cost_path.exists() else {}
    cost_doc[run_id] = {
        "mode": args.mode,
        "seed": int(args.seed),
        "trainable_lora_params": int(trainable_params),
        "total_student_params": int(total_student_params),
        "math_tokens_scored": int(math_scored),
        "short_tokens_scored": int(short_scored),
        "total_tokens_scored": int(math_scored + short_scored),
        "optimizer_updates": int(opt_updates),
        "optimizer_attempts": int(opt_attempts),
        "skipped_overflows": int(skipped_overflows),
        "teacher_rollout_prep_sec": float(teacher_pool_sec),
        "student_rollout_sec": float(t_student_rollout_sec),
        "teacher_scoring_sec": float(t_teacher_score_sec),
        "backward_update_sec": float(t_backward_sec),
        "adapter_merge_sec": float(merge_rep["merge_sec"]),
        "total_train_wall_sec": float(wall_sec),
        "total_end_to_end_cost_sec": float(teacher_pool_sec + wall_sec + merge_rep["merge_sec"]),
        "peak_allocated_gpu_bytes": peak_mem["peak_allocated_bytes"],
        "peak_reserved_gpu_bytes": peak_mem["peak_reserved_bytes"],
    }
    atomic_write_json(cost_path, cost_doc)

    log_doc = {
        "status": "completed",
        "run_id": run_id,
        "mode": args.mode,
        "seed": int(args.seed),
        "cost": cost_doc[run_id],
        "merge_verification": merge_rep,
        "steps": step_logs,
    }
    atomic_write_json(dirs["results"] / f"distill_log_{run_id}.json", log_doc)
    _rebuild_distill_logs_csv(dirs["results"])
    update_condition_status(
        dirs["results"] / "condition_status.json",
        f"distill_{run_id}",
        "completed",
        cost_doc[run_id],
    )
    return log_doc


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train_distill(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
