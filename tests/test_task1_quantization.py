"""Unit and local smoke tests for Task 1 quantization (Q1, Q2, Q3) and Step 9 analysis."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from common.io import read_json
from common.seeding import DEFAULT_SEED, set_seed
from task1.analyze_results import (
    analyze_task1,
    build_arg_parser as build_analyze_parser,
    paired_bootstrap_acc_diff,
    paired_bootstrap_ppl_diff,
)
from task1.model_utils import enumerate_eligible_projections
from task1.pruning import restore_eligible_weights, snapshot_eligible_weights
from task1.quantization import (
    allocate_q3_mixed_precision,
    apply_awq_quantization,
    compute_matrix_packed_bytes,
    compute_quantized_model_storage,
    quantize_dequantize_weight,
    unwrap_scaled_linears,
)
from task1.run_baseline import build_arg_parser as build_baseline_parser
from task1.run_baseline import run_baseline
from task1.run_q3_mixed import build_arg_parser as build_q3_parser
from task1.run_q3_mixed import run_q3_mixed
from task1.run_quantization import build_arg_parser as build_q_parser
from task1.run_quantization import run_quantization
from tests.test_task1_pruning import _ByteTokenizer, _make_synthetic_data_dir


def _make_tiny_qwen2(num_layers: int = 2) -> Qwen2ForCausalLM:
    set_seed(DEFAULT_SEED)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg).to(dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class TestTask1Quantization(unittest.TestCase):
    def test_cli_help_flags(self) -> None:
        for mod, flags in [
            ("task1.run_quantization", ("--method", "--bits", "--group-size", "--data-dir", "--out-dir", "--seed", "--smoke", "--resume")),
            ("task1.run_q3_mixed", ("--budgets", "--group-size", "--data-dir", "--out-dir", "--seed", "--smoke", "--resume")),
            ("task1.analyze_results", ("--data-dir", "--out-dir", "--seed", "--smoke")),
        ]:
            proc = subprocess.run(
                [sys.executable, "-m", mod, "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            for flag in flags:
                self.assertIn(flag, proc.stdout)

    def test_quantize_dequantize_accuracy_and_zero_group_safety(self) -> None:
        set_seed(DEFAULT_SEED)
        w = torch.randn(8, 30, dtype=torch.float32)  # 30 not divisible by 16 -> tests padding
        w[0].zero_()  # all-zero row / group
        errs = {}
        for b in (8, 4, 3, 2):
            w_hat = quantize_dequantize_weight(w, bits=b, group_size=16)
            self.assertEqual(w_hat.shape, w.shape)
            self.assertTrue(torch.isfinite(w_hat).all().item())
            self.assertTrue(torch.all(w_hat[0] == 0.0).item())
            errs[b] = float((w_hat - w).pow(2).mean().item())
        self.assertLess(errs[8], errs[4])
        self.assertLess(errs[4], errs[3])
        self.assertLess(errs[3], errs[2])

    def test_packed_byte_accounting_and_awq_folding_equivalence(self) -> None:
        model = _make_tiny_qwen2(num_layers=2)
        projs = enumerate_eligible_projections(model)
        clean = snapshot_eligible_weights(model)

        # Packed bytes check for 1 matrix: 32 x 32, b=4, g=16 -> 64 groups
        # payload = 1024 * 4 / 8 = 512; meta = 64 * 3 = 192; comp = 32 * 2 = 64
        m = compute_matrix_packed_bytes(32, 32, bits=4, group_size=16, has_compensation_vector=True)
        self.assertEqual(m["payload_bytes"], 512)
        self.assertEqual(m["metadata_bytes"], 192)
        self.assertEqual(m["compensation_bytes"], 64)
        self.assertEqual(m["total_bytes"], 768)

        # Verify AWQ full-precision rescaling + down_proj->up_proj folding preserves outputs BEFORE rounding
        mean_abs_x = {
            p.full_name: torch.linspace(0.5, 2.5, p.in_features) for p in projs
        }
        bits_map = {p.full_name: 8 for p in projs}
        alpha_map = {p.full_name: 0.5 for p in projs}
        sample_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)

        res = apply_awq_quantization(
            model=model,
            clean_weights=clean,
            mean_abs_x=mean_abs_x,
            bits_by_name=bits_map,
            alpha_by_name=alpha_map,
            group_size=16,
            verify_sample_ids=sample_ids,
        )
        # 2 layers -> 2 down_proj folded into up_proj; 12 non-folded wrappers
        self.assertEqual(len(res["folded_projections"]), 2)
        self.assertEqual(len(res["non_folded_comp_names"]), 12)
        self.assertLess(res["fp_rescaling_equivalence"]["max_abs_logit_diff"], 1e-4)

        unwrapped = unwrap_scaled_linears(model)
        self.assertEqual(unwrapped, 12)
        restore_eligible_weights(model, clean)

    def test_q3_mixed_precision_respects_budget_and_prioritizes_sensitive(self) -> None:
        model = _make_tiny_qwen2(num_layers=2)
        projs = enumerate_eligible_projections(model)
        non_folded = {p.full_name for p in projs if p.proj_name != "down_proj"}
        q2_int4 = compute_quantized_model_storage(
            model, {p.full_name: 4 for p in projs}, group_size=16, non_folded_comp_names=non_folded
        )
        budget = q2_int4["serialized_packed_total_bytes"]
        uncomp = q2_int4["uncompressed_fp16_bytes"]

        # Make projection 0 100x more sensitive than all others
        errs_by_b = {
            b: {
                p.full_name: (100.0 if i == 0 else 1.0) / (2.0 ** b)
                for i, p in enumerate(projs)
            }
            for b in (2, 3, 4, 8)
        }
        alloc = allocate_q3_mixed_precision(
            projections=projs,
            errors_by_bits=errs_by_b,
            budget_packed_bytes=budget,
            uncompressed_fp16_bytes=uncomp,
            non_folded_comp_names=non_folded,
            group_size=16,
            min_bits=2,
        )
        self.assertLessEqual(alloc["achieved_packed_bytes"], budget)
        self.assertEqual(alloc["bits_by_name"][projs[0].full_name], 8)

    def test_bootstrap_functions(self) -> None:
        win_a = [{"nll_sum": 10.0, "scored_tokens": 10} for _ in range(8)]
        win_b = [{"nll_sum": 12.0, "scored_tokens": 10} for _ in range(8)]
        ppl_ci = paired_bootstrap_ppl_diff(win_a, win_b, n_resamples=200, seed=6304)
        self.assertLess(ppl_ci["point_diff_a_minus_b"], 0.0)
        self.assertLess(ppl_ci["ci95_high"], 0.0)

        ex_a = [{"correct_norm": 1} for _ in range(10)]
        ex_b = [{"correct_norm": 0} for _ in range(10)]
        acc_ci = paired_bootstrap_acc_diff(ex_a, ex_b, n_resamples=200, seed=6304)
        self.assertAlmostEqual(acc_ci["point_diff_a_minus_b"], 1.0, places=6)

    def test_end_to_end_q1_q2_q3_and_analyze_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = _make_synthetic_data_dir(root)
            out_dir = root / "task1"
            model = _make_tiny_qwen2(num_layers=2)
            tok = _ByteTokenizer()

            common_flags = [
                "--smoke",
                "--data-dir", str(data_dir),
                "--out-dir", str(out_dir),
                "--seed", "6304",
                "--dtype", "fp32",
                "--device", "cpu",
                "--prefill-lengths", "16",
                "--decode-tokens", "4",
                "--warmup-reps", "1",
                "--timing-reps", "2",
            ]

            run_baseline(build_baseline_parser().parse_args(common_flags), model=model, tokenizer=tok)
            run_quantization(
                build_q_parser().parse_args(["--method", "q1_uniform", "--bits", "4", "--group-size", "16", *common_flags]),
                model=model,
                tokenizer=tok,
            )
            run_quantization(
                build_q_parser().parse_args(["--method", "q2_awq", "--bits", "4", "3", "--group-size", "16", *common_flags]),
                model=model,
                tokenizer=tok,
            )
            q3_res = run_q3_mixed(
                build_q3_parser().parse_args(["--budgets", "int4", "int3", "--group-size", "16", *common_flags]),
                model=model,
                tokenizer=tok,
            )
            self.assertGreaterEqual(q3_res["q3_mixed_int4"]["matched_budget_check"]["bytes_under_budget"], 0)
            self.assertGreaterEqual(q3_res["q3_mixed_int3"]["matched_budget_check"]["bytes_under_budget"], 0)

            analyze_task1(
                build_analyze_parser().parse_args(
                    ["--smoke", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--seed", "6304", "--bootstrap-resamples", "200"]
                )
            )
            smoke_root = out_dir / "smoke"
            for rel in (
                "results/summary_all_conditions.csv",
                "results/resource_accounting.csv",
                "results/bootstrap_ci_p3_q3.json",
                "results/q3_allocation_and_sensitivity.json",
                "evidence/qualitative_cases.json",
            ):
                p = smoke_root / rel
                self.assertTrue(p.exists() and p.stat().st_size > 0, msg=f"Missing {rel}")

            ci_doc = read_json(smoke_root / "results" / "bootstrap_ci_p3_q3.json")
            self.assertEqual(ci_doc["comparisons"]["q3_minus_q2_int4"]["status"], "completed")
            self.assertEqual(ci_doc["comparisons"]["q3_minus_q2_int3"]["status"], "completed")
            bundle_manifest = smoke_root / "final_report_bundle" / "MANIFEST.json"
            self.assertTrue(bundle_manifest.exists())
            self.assertTrue(
                (smoke_root / "final_report_bundle" / "results" / "summary_all_conditions.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
