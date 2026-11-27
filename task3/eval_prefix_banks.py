"""Task 3 Step 5: Common-Prefix Distribution Alignment Across All 7 Prefix Banks.

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task3.eval_prefix_banks --data-dir <DATA_DIR> --out-dir <T3_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
import csv
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    ensure_task_dirs,
    read_jsonl,
    resolve_smoke_out_dir,
    update_condition_status,
)
from common.seeding import DEFAULT_SEED, set_seed
from task3.data import (
    DRAFT_MODEL_ID,
    TARGET_MODEL_ID,
    format_chat_prompt,
    get_eos_token_ids,
    load_task3_test_prompts,
)
from task3.sampler import exact_speculative_generate, target_only_generate

POS_BINS = (
    ("[1,16]", 1, 16),
    ("[17,32]", 17, 32),
    ("[33,64]", 33, 64),
    ("[65,192]", 65, 192),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 3 Step 5: Forward KL, reverse KL, TV, greedy agreement, and 1-TV on prefix banks."
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


def _pos_bin(pos: int) -> str:
    for lbl, lo, hi in POS_BINS:
        if lo <= pos <= hi:
            return lbl
    return ">192"


def _sample_positions_from_response(resp_ids: list[int], is_math: bool) -> list[int]:
    """Select up to 32 positions in 1..64 (8+8+16) + up to 16 in 65..192 (1-indexed)."""
    n = len(resp_ids)
    chosen: list[int] = []
    for lo, hi, cap in ((1, 16, 8), (17, 32, 8), (33, 64, 16)):
        avail = [p for p in range(lo, min(hi, n) + 1)]
        if len(avail) <= cap:
            chosen.extend(avail)
        else:
            idxs = np.linspace(0, len(avail) - 1, cap, dtype=int).tolist()
            chosen.extend(avail[i] for i in idxs)
    if is_math and n >= 65:
        avail_long = list(range(65, min(192, n) + 1))
        if len(avail_long) <= 16:
            chosen.extend(avail_long)
        else:
            idxs = np.linspace(0, len(avail_long) - 1, 16, dtype=int).tolist()
            chosen.extend(avail_long[i] for i in idxs)
    return sorted(set(chosen))


def _build_bank_entries_from_rollouts(
    bank_name: str,
    test_prompts: list[dict[str, Any]],
    rollout_fn,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for idx, p in enumerate(test_prompts):
        chat_text = format_chat_prompt(tokenizer, p["prompt"])
        prompt_ids = tokenizer.encode(chat_text, add_special_tokens=False)
        cap = 192 if p["category"] == "math" else 32
        resp_ids = rollout_fn(prompt_ids, cap, idx)
        for pos in _sample_positions_from_response(resp_ids, is_math=(p["category"] == "math")):
            entries.append(
                {
                    "prompt_id": p["prompt_id"],
                    "workload": p["workload"],
                    "bank": bank_name,
                    "position": int(pos),
                    "prompt_token_ids": prompt_ids,
                    "response_prefix_token_ids": resp_ids[: pos - 1],
                }
            )
    return entries


def _compute_alignment_metrics_for_entry(
    teacher_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    prompt_ids: list[int],
    prefix_ids: list[int],
    device: torch.device,
) -> dict[str, Any]:
    """Compute forward KL, reverse KL, TV, greedy agreement, and 1-TV at one prefix."""
    full_ids = prompt_ids + prefix_ids
    inp = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        t_logits = teacher_model(input_ids=inp, use_cache=False).logits[0, -1, :].float()
        d_logits = draft_model(input_ids=inp, use_cache=False).logits[0, -1, :].float()
    log_p = F.log_softmax(t_logits, dim=-1)
    log_q = F.log_softmax(d_logits, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    fwd_kl = float((p * (log_p - log_q)).sum().item())
    rev_kl = float((q * (log_q - log_p)).sum().item())
    tv = float(0.5 * (p - q).abs().sum().item())
    t_top = int(t_logits.argmax().item())
    d_top = int(d_logits.argmax().item())
    return {
        "forward_kl": fwd_kl,
        "reverse_kl": rev_kl,
        "tv": tv,
        "expected_acceptance": 1.0 - tv,
        "greedy_agree": int(t_top == d_top),
        "teacher_top_id": t_top,
        "draft_top_id": d_top,
    }


def eval_prefix_banks(
    args: argparse.Namespace,
    teacher_model: torch.nn.Module | None = None,
    draft_models: dict[str, torch.nn.Module] | None = None,
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
    dt = torch.float16 if device.type == "cuda" else torch.float32

    test_prompts = load_task3_test_prompts(args.data_dir)
    if args.smoke:
        test_prompts = test_prompts[:4]
    workload_by_pid = {p["prompt_id"]: p["workload"] for p in test_prompts}

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, use_fast=True)
    if teacher_model is None:
        teacher_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, dtype=dt).to(device).eval()
    eos_ids = get_eos_token_ids(tokenizer, teacher_model)

    if draft_models is None:
        draft_models = {
            "original_0.5b": AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_ID, dtype=dt).to(device).eval()
        }
        for run_id in ("off_policy_s6304", "off_policy_s6305", "on_policy_s6304", "on_policy_s6305"):
            mdir = dirs["checkpoints"] / run_id / "merged_fp16"
            if mdir.exists():
                draft_models[run_id] = AutoModelForCausalLM.from_pretrained(str(mdir), dtype=dt).to(device).eval()

    # 1. Load staff prefix_banks_locked.jsonl.gz if present; else generate teacher & original_draft cleanly
    bank_entries: list[dict[str, Any]] = []
    staff_bank_path = None
    for cand in (
        Path(args.data_dir) / "prefix_banks" / "prefix_banks_locked.jsonl.gz",
        Path(args.data_dir) / "data" / "prefix_banks" / "prefix_banks_locked.jsonl.gz",
    ):
        if cand.exists():
            staff_bank_path = cand
            break

    if staff_bank_path is not None:
        print(f"[Task 3 Step 5] Loading staff prefix banks from {staff_bank_path}...", flush=True)
        for rec in read_jsonl(staff_bank_path):
            pid = str(rec["prompt_id"])
            if args.smoke and pid not in workload_by_pid:
                continue
            p_ids = (
                rec["prompt_token_ids"]
                if "prompt_token_ids" in rec
                else tokenizer.encode(format_chat_prompt(tokenizer, str(rec["prompt"])), add_special_tokens=False)
            )
            bank_entries.append(
                {
                    "prompt_id": pid,
                    "workload": workload_by_pid.get(pid, "math"),
                    "bank": str(rec["bank"]),
                    "position": int(rec["position"]),
                    "prompt_token_ids": p_ids,
                    "response_prefix_token_ids": list(rec["response_prefix_token_ids"]),
                }
            )
    else:
        print(
            "[Task 3 Step 5] Staff prefix_banks_locked.jsonl.gz not found; generating teacher & original_draft banks cleanly...",
            flush=True,
        )
        bank_entries.extend(
            _build_bank_entries_from_rollouts(
                "teacher",
                test_prompts,
                lambda p_ids, cap, i: target_only_generate(
                    teacher_model, p_ids, min(cap, 16 if args.smoke else cap), eos_ids, device, seed=args.seed + i
                )["generated_ids"],
                tokenizer,
            )
        )
        bank_entries.extend(
            _build_bank_entries_from_rollouts(
                "original_draft",
                test_prompts,
                lambda p_ids, cap, i: target_only_generate(
                    draft_models["original_0.5b"], p_ids, min(cap, 16 if args.smoke else cap), eos_ids, device, seed=args.seed + 1000 + i
                )["generated_ids"],
                tokenizer,
            )
        )

    # 2. Always generate exact_spec_original_gamma4 using our exact speculative sampler
    print("[Task 3 Step 5] Generating exact_spec_original_gamma4 distributional-consistency bank...", flush=True)
    bank_entries.extend(
        _build_bank_entries_from_rollouts(
            "exact_spec_original_gamma4",
            test_prompts,
            lambda p_ids, cap, i: exact_speculative_generate(
                teacher_model, draft_models["original_0.5b"], p_ids,
                min(cap, 16 if args.smoke else cap), gamma=4, eos_ids=eos_ids, device=device, seed=args.seed + 2000 + i,
            )["generated_ids"],
            tokenizer,
        )
    )

    # 3. Generate frozen own-student rollout banks for every trained student
    for d_name, d_mod in draft_models.items():
        if d_name == "original_0.5b":
            continue
        print(f"[Task 3 Step 5] Generating frozen rollout bank for {d_name}...", flush=True)
        bank_entries.extend(
            _build_bank_entries_from_rollouts(
                f"student_{d_name}",
                test_prompts,
                lambda p_ids, cap, i, m=d_mod: target_only_generate(
                    m, p_ids, min(cap, 16 if args.smoke else cap), eos_ids, device, seed=args.seed + 3000 + i
                )["generated_ids"],
                tokenizer,
            )
        )

    # 4. Evaluate every draft on every bank
    grouped_stats: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    disagreements: list[dict[str, Any]] = []

    for d_name, d_mod in draft_models.items():
        for entry in bank_entries:
            m = _compute_alignment_metrics_for_entry(
                teacher_model, d_mod, entry["prompt_token_ids"], entry["response_prefix_token_ids"], device
            )
            key = (d_name, entry["bank"], entry["workload"], _pos_bin(entry["position"]))
            grouped_stats.setdefault(key, []).append(m)
            if (
                entry["bank"] == "teacher"
                and m["greedy_agree"] == 0
                and len(disagreements) < 20
            ):
                disagreements.append(
                    {
                        "draft": d_name,
                        "prompt_id": entry["prompt_id"],
                        "workload": entry["workload"],
                        "position": entry["position"],
                        "forward_kl": m["forward_kl"],
                        "tv": m["tv"],
                        "teacher_top_id": m["teacher_top_id"],
                        "draft_top_id": m["draft_top_id"],
                    }
                )

    rows: list[dict[str, Any]] = []
    for (d_name, bank_name, workload, pbin), mlist in sorted(grouped_stats.items()):
        rows.append(
            {
                "draft": d_name,
                "bank": bank_name,
                "workload": workload,
                "position_bin": pbin,
                "num_prefixes": len(mlist),
                "mean_forward_kl": float(np.mean([x["forward_kl"] for x in mlist])),
                "mean_reverse_kl": float(np.mean([x["reverse_kl"] for x in mlist])),
                "mean_tv": float(np.mean([x["tv"] for x in mlist])),
                "mean_expected_acceptance": float(np.mean([x["expected_acceptance"] for x in mlist])),
                "greedy_agreement": float(np.mean([x["greedy_agree"] for x in mlist])),
            }
        )

    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    atomic_write_text(dirs["results"] / "prefix_bank_alignment.csv", buf.getvalue())
    atomic_write_json(
        dirs["evidence"] / "qualitative_disagreements_and_continuations.json",
        {"prefix_level_disagreements": disagreements, "matched_prompt_continuations": []},
    )
    update_condition_status(
        dirs["results"] / "condition_status.json",
        "step5_prefix_alignment",
        "completed",
        {"num_rows": len(rows), "num_drafts": len(draft_models)},
    )
    print(f"[Task 3 Step 5] Wrote prefix_bank_alignment.csv ({len(rows)} rows).", flush=True)
    return {"rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    eval_prefix_banks(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
