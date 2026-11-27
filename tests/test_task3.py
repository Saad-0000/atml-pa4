"""Unit and local smoke tests for Task 3 (exact sampler, Markov chi^2, distillation, eval)."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from common.seeding import DEFAULT_SEED, set_seed
from task3.analyze_results import analyze_task3, build_arg_parser as build_t3_analyze_parser
from task3.eval_prefix_banks import build_arg_parser as build_pb_parser, eval_prefix_banks
from task3.prepare_teacher_pool import build_arg_parser as build_tp_parser, prepare_teacher_pool
from task3.run_spec_eval import build_arg_parser as build_se_parser, run_spec_eval
from task3.sampler import (
    analytical_speedup,
    compute_breakeven_alpha,
    exact_speculative_generate,
    run_markov_chi2_validation,
)
from task3.train_distill import build_arg_parser as build_td_parser, train_distill
from task3.validate_sampler import build_arg_parser as build_vs_parser, run_validate_sampler


class _SimpleTok:
    eos_token_id = 1
    pad_token_id = 0

    def __len__(self) -> int:
        return 128

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [min(127, max(2, ord(c) % 120 + 2)) for c in text[:32]] or [10, 11]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "#### 42"


def _make_tiny_lm(seed: int = 6304, hidden: int = 32) -> Qwen2ForCausalLM:
    set_seed(seed)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(cfg).to(dtype=torch.float32).eval()


def _write_gz_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


class TestTask3(unittest.TestCase):
    def test_cli_help_flags(self) -> None:
        for mod, flags in [
            ("task3.validate_sampler", ("--out-dir", "--seed", "--smoke")),
            ("task3.prepare_teacher_pool", ("--resume", "--data-dir", "--out-dir", "--seed")),
            ("task3.train_distill", ("--mode", "--seed", "--ckpt-every-tokens", "--resume", "--data-dir", "--out-dir")),
            ("task3.eval_prefix_banks", ("--data-dir", "--out-dir", "--seed")),
            ("task3.run_spec_eval", ("--draft", "--gammas", "--include-unmerged-diag", "--include-fixed64-diag", "--resume")),
            ("task3.analyze_results", ("--data-dir", "--out-dir", "--seed")),
        ]:
            proc = subprocess.run(
                [sys.executable, "-m", mod, "--help"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"{mod}: {proc.stderr}")
            for flag in flags:
                self.assertIn(flag, proc.stdout)

    def test_markov_chi2_exact_passes_and_negative_control_rejects(self) -> None:
        rep = run_markov_chi2_validation(n_trials=30_000, seq_len=3, gamma=2, seed=6304)
        self.assertTrue(rep["exact_sampler"]["passed_not_rejected_at_0p01"], msg=rep)
        self.assertTrue(rep["negative_control_p_instead_of_residual"]["passed_rejected_at_0p01"], msg=rep)
        self.assertTrue(rep["passed"])

    def test_analytical_cost_model_and_breakeven(self) -> None:
        s = analytical_speedup(alpha=0.8, c=0.3, v_gamma=1.1, gamma=4)
        self.assertGreater(s, 1.0)
        be = compute_breakeven_alpha(c=0.3, v_gamma=1.1, gamma=4)
        self.assertIsNotNone(be)
        self.assertGreater(be, 0.0)
        self.assertLess(be, 1.0)
        # Impossible break-even when draft is too slow
        self.assertIsNone(compute_breakeven_alpha(c=1.5, v_gamma=1.2, gamma=4))

    def test_exact_speculative_cache_rollback_invariant(self) -> None:
        target = _make_tiny_lm(6304, hidden=32)
        draft = _make_tiny_lm(6305, hidden=32)
        prompt = [5, 10, 15, 20, 25]
        for g in (1, 2, 4):
            out = exact_speculative_generate(
                target, draft, prompt, max_new_tokens=9, gamma=g, eos_ids={1}, device=torch.device("cpu"), seed=6304 + g, ignore_eos=True
            )
            self.assertEqual(out["num_tokens"], 9)
            expected_len = len(prompt) + 9 - 1
            self.assertEqual(out["final_target_cache_len"], expected_len)
            self.assertEqual(out["final_draft_cache_len"], expected_len)

    def test_end_to_end_task3_pipeline_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            out_dir = root / "task3"

            gsm_train = [{"id": f"m_tr_{i}", "question": f"What is {i}+{i}?", "answer": f"#### {2*i}"} for i in range(6)]
            gsm_test = [{"id": f"m_te_{i}", "question": f"What is {i}+1?", "answer": f"#### {i+1}"} for i in range(4)]
            sf_train = [{"id": f"s_tr_{i}", "prompt": f"Extract code {i}", "answer": f"{i}", "family": i % 2} for i in range(6)]
            sf_test = [
                {"id": f"s_te_{i}", "prompt": f"Extract code {i}", "answer": f"{i}", "split": "familiar" if i < 2 else "shifted", "family": i % 2}
                for i in range(4)
            ]
            _write_gz_jsonl(data_dir / "public_locked" / "gsm8k_train_locked.jsonl.gz", gsm_train)
            _write_gz_jsonl(data_dir / "public_locked" / "gsm8k_test_locked.jsonl.gz", gsm_test)
            _write_jsonl(data_dir / "short_form" / "short_train.jsonl", sf_train)
            _write_jsonl(data_dir / "short_form" / "short_test.jsonl", sf_test)

            target = _make_tiny_lm(6304, hidden=32)
            draft_orig = _make_tiny_lm(6305, hidden=32)
            tok = _SimpleTok()

            # Step 1
            run_validate_sampler(
                build_vs_parser().parse_args(["--smoke", "--out-dir", str(out_dir), "--seed", "6304", "--device", "cpu"]),
                target_model=target,
                draft_model=draft_orig,
            )
            # Step 3 prep
            prepare_teacher_pool(
                build_tp_parser().parse_args(["--smoke", "--resume", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--device", "cpu"]),
                teacher_model=target,
                tokenizer=tok,
            )
            # Step 3 off-policy & Step 4 on-policy distillation (if peft is available locally)
            has_peft = True
            try:
                import peft  # noqa: F401
            except ImportError:
                has_peft = False

            draft_map = {"original_0.5b": draft_orig}
            if has_peft:
                for mode in ("off_policy", "on_policy"):
                    train_distill(
                        build_td_parser().parse_args(
                            ["--mode", mode, "--seed", "6304", "--ckpt-every-tokens", "32", "--resume", "--smoke", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--device", "cpu"]
                        ),
                        teacher_model=target,
                        student_base_fn=lambda: _make_tiny_lm(6305, hidden=32),
                        tokenizer=tok,
                    )
                    mdir = out_dir / "smoke" / "checkpoints" / f"{mode}_s6304" / "merged_fp16"
                    draft_map[f"{mode}_s6304"] = Qwen2ForCausalLM.from_pretrained(str(mdir)).eval()

            # Step 5 prefix alignment
            eval_prefix_banks(
                build_pb_parser().parse_args(["--smoke", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--device", "cpu"]),
                teacher_model=target,
                draft_models=draft_map,
                tokenizer=tok,
            )
            # Step 6 spec eval
            run_spec_eval(
                build_se_parser().parse_args(
                    ["--draft", "all_distilled", "--gammas", "2", "4", "--include-fixed64-diag", "--smoke", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--device", "cpu"]
                ),
                target_model=target,
                draft_models=draft_map,
                tokenizer=tok,
            )
            # Step 7 analysis & bundle
            analyze_task3(
                build_t3_analyze_parser().parse_args(["--smoke", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--seed", "6304", "--bootstrap-resamples", "100"])
            )

            smoke_root = out_dir / "smoke"
            for rel in (
                "results/step1_sampler_validation.json",
                "results/prefix_bank_alignment.csv",
                "results/speculative_decoding_summary.csv",
                "results/speculative_round_timing_decomposition.csv",
                "results/speculative_bootstrap_ci.json",
                "results/standalone_student_accuracy.csv",
                "results/amortized_cost_breakeven.json",
                "evidence/qualitative_disagreements_and_continuations.json",
                "final_report_bundle/MANIFEST.json",
            ):
                p = smoke_root / rel
                self.assertTrue(p.exists() and p.stat().st_size > 0, msg=f"Missing {rel}")


if __name__ == "__main__":
    unittest.main()
