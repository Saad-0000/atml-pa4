"""Task 3 Step 1: Validate Exact Speculative Sampler (100k Markov chi^2 + Cache Tests).

CLI compatible with ATML_PA4_Colab_Runner.ipynb:
  python -m task3.validate_sampler --out-dir <T3_OUT> --seed 6304
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.env import collect_env_metadata, resolve_model_revision
from common.io import (
    atomic_write_json,
    ensure_task_dirs,
    resolve_smoke_out_dir,
    update_condition_status,
)
from common.seeding import DEFAULT_SEED, set_seed
from task3.data import DRAFT_MODEL_ID, TARGET_MODEL_ID, get_eos_token_ids
from task3.sampler import exact_speculative_generate, run_markov_chi2_validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task 3 Step 1: 100k Markov chi^2 test, negative control, and real-model cache consistency."
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-real-model", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


def validate_real_model_cache_consistency(
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    eos_ids: set[int],
    device: torch.device,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Verify split-prefill equivalence and exact cache length invariant across gammas 1, 2, 4."""
    prompt_ids = [10, 20, 30, 40, 50, 60, 70, 80]
    full_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        full_logits = target_model(input_ids=full_t, use_cache=False).logits[0, -1, :].float()
        pre_out = target_model(input_ids=full_t[:, :-1], use_cache=True)
        step_out = target_model(
            input_ids=full_t[:, -1:],
            past_key_values=pre_out.past_key_values,
            use_cache=True,
        )
        split_logits = step_out.logits[0, -1, :].float()
    split_max_diff = float((full_logits - split_logits).abs().max().item())

    gamma_checks = {}
    for g in (1, 2, 4):
        res = exact_speculative_generate(
            target_model=target_model,
            draft_model=draft_model,
            prompt_ids=prompt_ids,
            max_new_tokens=8,
            gamma=g,
            eos_ids=eos_ids,
            device=device,
            seed=seed + g,
            ignore_eos=True,
        )
        expected_cache_len = len(prompt_ids) + res["num_tokens"] - 1
        gamma_checks[f"gamma_{g}"] = {
            "num_tokens": res["num_tokens"],
            "expected_cache_len": expected_cache_len,
            "target_cache_len": res["final_target_cache_len"],
            "draft_cache_len": res["final_draft_cache_len"],
            "invariant_ok": (
                res["final_target_cache_len"] == expected_cache_len
                and res["final_draft_cache_len"] == expected_cache_len
            ),
        }

    return {
        "split_prefill_max_abs_logit_diff": split_max_diff,
        "split_prefill_ok": split_max_diff < 5e-2,
        "gamma_cache_checks": gamma_checks,
        "passed": split_max_diff < 5e-2 and all(v["invariant_ok"] for v in gamma_checks.values()),
    }


def run_validate_sampler(
    args: argparse.Namespace,
    target_model: torch.nn.Module | None = None,
    draft_model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    set_seed(args.seed)
    resolved_out = resolve_smoke_out_dir(args.out_dir, smoke=args.smoke)
    dirs = ensure_task_dirs(resolved_out)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    n_trials = 20_000 if args.smoke else 100_000
    print(f"[Task 3 Step 1] Running {n_trials:,}-trial Markov chi^2 validation & negative control...", flush=True)
    markov_rep = run_markov_chi2_validation(n_trials=n_trials, seq_len=3, gamma=2, seed=args.seed)
    print(
        f"  Exact sampler: chi2={markov_rep['exact_sampler']['chi2_stat']:.2f}, "
        f"p={markov_rep['exact_sampler']['p_value']:.4f} (pass={markov_rep['exact_sampler']['passed_not_rejected_at_0p01']})",
        flush=True,
    )
    print(
        f"  Neg control:   chi2={markov_rep['negative_control_p_instead_of_residual']['chi2_stat']:.2f}, "
        f"p={markov_rep['negative_control_p_instead_of_residual']['p_value']:.2e} "
        f"(pass={markov_rep['negative_control_p_instead_of_residual']['passed_rejected_at_0p01']})",
        flush=True,
    )

    real_check: dict[str, Any] = {"skipped": True, "passed": True}
    model_revs: dict[str, str] = {}
    if not args.skip_real_model:
        if target_model is None or draft_model is None:
            dt = torch.float16 if device.type == "cuda" else torch.float32
            tok = AutoTokenizer.from_pretrained(TARGET_MODEL_ID, use_fast=True)
            target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, dtype=dt).to(device).eval()
            draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_ID, dtype=dt).to(device).eval()
            eos_ids = get_eos_token_ids(tok, target_model)
            model_revs = {
                TARGET_MODEL_ID: resolve_model_revision(target_model, TARGET_MODEL_ID),
                DRAFT_MODEL_ID: resolve_model_revision(draft_model, DRAFT_MODEL_ID),
            }
        else:
            eos_ids = {1}
        real_check = validate_real_model_cache_consistency(
            target_model, draft_model, eos_ids, device, seed=args.seed
        )
        print(
            f"  Real-model cache check: split_diff={real_check['split_prefill_max_abs_logit_diff']:.6f}, "
            f"passed={real_check['passed']}",
            flush=True,
        )

    atomic_write_json(
        dirs["configs"] / "step1_env_metadata.json",
        collect_env_metadata(model_revisions=model_revs, seed=args.seed, extra={"stage": "task3_step1"}),
    )

    report = {
        "status": "completed" if (markov_rep["passed"] and real_check["passed"]) else "failed",
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "markov_validation": markov_rep,
        "real_model_cache_validation": real_check,
        "one_pending_token_invariant": (
            "Both target and draft DynamicCaches hold emitted context[:-1]; pending_token = context[-1] "
            "is fed jointly with gamma proposals in one target verification pass, then both caches are "
            "cropped/advanced to new_len - 1."
        ),
    }
    atomic_write_json(dirs["results"] / "step1_sampler_validation.json", report)
    update_condition_status(
        dirs["results"] / "condition_status.json",
        "step1_validate_sampler",
        report["status"],
        {"markov_passed": markov_rep["passed"], "cache_passed": real_check["passed"]},
    )
    if report["status"] != "completed":
        raise AssertionError(f"Step 1 sampler validation failed: {report}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_validate_sampler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
