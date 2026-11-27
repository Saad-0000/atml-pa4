"""Task 3 Step 3 prep: Generate frozen teacher-response pool for off-policy distillation.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task3.prepare_teacher_pool --resume --data-dir <DATA_DIR> --out-dir <T3_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import (
    atomic_write_json,
    atomic_write_jsonl,
    ensure_task_dirs,
    read_jsonl,
    resolve_smoke_out_dir,
    update_condition_status,
)
from common.seeding import DEFAULT_SEED, set_seed
from task3.data import (
    TARGET_MODEL_ID,
    format_chat_prompt,
    get_eos_token_ids,
    load_task3_train_pool,
)
from task3.sampler import target_only_generate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare frozen teacher rollout pool (500 GSM8K + 300 short-form) for off-policy KD."
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


def prepare_teacher_pool(
    args: argparse.Namespace,
    teacher_model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    math_pool, short_pool = load_task3_train_pool(args.data_dir)
    if args.smoke:
        math_pool = math_pool[:4]
        short_pool = short_pool[:4]
    full_pool = math_pool + short_pool

    pool_path = dirs["results"] / "teacher_rollout_pool.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and pool_path.exists():
        for rec in read_jsonl(pool_path):
            existing[str(rec["source_id"])] = rec
        print(f"[RESUME] Loaded {len(existing)}/{len(full_pool)} existing teacher rollouts.", flush=True)

    if len(existing) < len(full_pool):
        if teacher_model is None or tokenizer is None:
            dt = torch.float16 if device.type == "cuda" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, use_fast=True)
            teacher_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, dtype=dt).to(device).eval()
        eos_ids = get_eos_token_ids(tokenizer, teacher_model)
    else:
        eos_ids = set()

    t0 = time.perf_counter()
    ordered: list[dict[str, Any]] = []
    newly_generated = 0
    for idx, item in enumerate(full_pool):
        sid = item["source_id"]
        if sid in existing:
            ordered.append(existing[sid])
            continue

        chat_text = format_chat_prompt(tokenizer, item["prompt"])
        prompt_ids = tokenizer.encode(chat_text, add_special_tokens=False)
        cap = 16 if args.smoke else int(item["max_new_tokens"])
        rollout_seed = int(args.seed) + idx * 17
        gen_out = target_only_generate(
            target_model=teacher_model,
            prompt_ids=prompt_ids,
            max_new_tokens=cap,
            eos_ids=eos_ids,
            device=device,
            seed=rollout_seed,
            temperature=1.0,
        )
        rec = {
            "source_id": sid,
            "category": item["category"],
            "rollout_seed": rollout_seed,
            "prompt_token_ids": prompt_ids,
            "response_token_ids": gen_out["generated_ids"],
            "termination_reason": "eos" if gen_out["hit_eos"] else "max_tokens",
            "num_response_tokens": gen_out["num_tokens"],
            "gen_time_sec": gen_out["total_sec"],
        }
        ordered.append(rec)
        newly_generated += 1
        if newly_generated % 25 == 0 or (idx + 1) == len(full_pool):
            atomic_write_jsonl(pool_path, ordered)
            print(f"  [Teacher Pool] {len(ordered)}/{len(full_pool)} completed.", flush=True)

    atomic_write_jsonl(pool_path, ordered)
    wall_sec = time.perf_counter() - t0
    total_gen_sec = sum(float(r.get("gen_time_sec", 0.0)) for r in ordered)
    total_tokens = sum(int(r["num_response_tokens"]) for r in ordered)
    cost_report = {
        "status": "completed",
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "num_prompts": len(ordered),
        "total_response_tokens": int(total_tokens),
        "total_teacher_gen_sec": float(total_gen_sec),
        "wall_sec_this_run": float(wall_sec),
        "pool_path": str(pool_path),
    }
    atomic_write_json(dirs["results"] / "teacher_pool_cost.json", cost_report)
    update_condition_status(
        dirs["results"] / "condition_status.json",
        "step3_prepare_teacher_pool",
        "completed",
        cost_report,
    )
    return cost_report


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    prepare_teacher_pool(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
