"""Unit and local smoke tests for Task 2 Part C (KV cache compression) & Analysis."""

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
from task2.analyze_results import analyze_task2
from task2.analyze_results import build_arg_parser as build_t2_analyze_parser
from task2.kv_cache import (
    extract_five_digit_code,
    kv_quant_reference_bytes,
    max_eviction_positions,
    pack_kv_tensor,
    solve_hybrid_allocation,
    unpack_kv_tensor,
)
from task2.run_part_c import (
    build_arg_parser as build_pc_parser,
    run_stage_bench_packed,
    run_stage_c1,
    run_stage_eval_all,
)


class _CharOffsetTokenizer:
    """Offline character tokenizer with return_offsets_mapping support for unit tests."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        del add_special_tokens
        ids = [min(255, max(2, ord(ch) % 250 + 2)) for ch in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "Code is 54321"


def _make_tiny_qwen2_kv() -> Qwen2ForCausalLM:
    """Tiny Qwen2 with head_dim=64 so group_size=64 token-wise KV quant works identically."""
    set_seed(DEFAULT_SEED)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg).to(dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class TestTask2PartC(unittest.TestCase):
    def test_cli_help_flags(self) -> None:
        pc = subprocess.run(
            [sys.executable, "-m", "task2.run_part_c", "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(pc.returncode, 0, msg=pc.stderr)
        for flag in ("--stage", "--batches", "--lengths", "--resume", "--data-dir", "--out-dir", "--seed", "--smoke"):
            self.assertIn(flag, pc.stdout)

        an = subprocess.run(
            [sys.executable, "-m", "task2.analyze_results", "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(an.returncode, 0, msg=an.stderr)
        for flag in ("--data-dir", "--out-dir", "--seed", "--smoke"):
            self.assertIn(flag, an.stdout)

    def test_real_bit_packing_matches_pdf_formula_and_roundtrips(self) -> None:
        set_seed(DEFAULT_SEED)
        # Non-multiple of 64 (L=150 -> 2 complete 64-groups + 22 trailing FP16 residual tokens)
        bsz, n_kv, seq_len, head_dim = 1, 2, 150, 128
        k = torch.randn((bsz, n_kv, seq_len, head_dim), dtype=torch.float16)
        v = torch.randn((bsz, n_kv, seq_len, head_dim), dtype=torch.float16)

        for bits in (4, 2):
            expected_1layer = kv_quant_reference_bytes(seq_len, 1, n_kv, head_dim, bits=bits)
            for k_ax, v_ax in (("channel", "token"), ("token", "token")):
                pk = pack_kv_tensor(k, bits=bits, axis=k_ax)
                pv = pack_kv_tensor(v, bits=bits, axis=v_ax)
                self.assertEqual(pk.serialized_bytes() + pv.serialized_bytes(), expected_1layer)

                k_hat = unpack_kv_tensor(pk, out_dtype=torch.float16)
                v_hat = unpack_kv_tensor(pv, out_dtype=torch.float16)
                self.assertEqual(k_hat.shape, k.shape)
                # Trailing (150 mod 64 = 22) tokens must be bit-exact FP16!
                self.assertTrue(torch.equal(k_hat[:, :, 128:, :], k[:, :, 128:, :]))
                self.assertTrue(torch.equal(v_hat[:, :, 128:, :], v[:, :, 128:, :]))

        # Budget constraints for eviction & hybrid at L=2048
        b_int4 = kv_quant_reference_bytes(2048, 28, 2, 128, bits=4)
        ev = max_eviction_positions(2048, b_int4, 28, 2, 128)
        hy = solve_hybrid_allocation(2048, b_int4, 28, 2, 128)
        self.assertLessEqual(ev["total_bytes"], b_int4)
        self.assertLessEqual(hy["total_bytes"], b_int4)
        self.assertGreaterEqual(hy["k_selective_fp16"], 72)

    def test_code_extractor(self) -> None:
        self.assertEqual(extract_five_digit_code("The target code is 48291."), "48291")
        self.assertEqual(extract_five_digit_code("No code here"), "")

    def test_end_to_end_c1_eval_bench_and_analyze_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            out_dir = root / "task2"

            cases = []
            for i in range(4):
                prefix = "alpha beta gamma " * 10
                ev = f"RECORD TARGET CODE: {10000 + i} "
                suffix = "delta epsilon " * 5 + "\nQuestion: What is the 5-digit target code?"
                user_text = prefix + ev + suffix
                cases.append(
                    {
                        "id": f"case_{i}",
                        "user_text": user_text,
                        "code": str(10000 + i),
                        "position": "middle",
                        "distractors": 0,
                        "target_tokens": 2048,
                        "evidence_char_start": len(prefix),
                        "evidence_char_end": len(prefix) + len(ev),
                    }
                )
            p = data_dir / "kv_retrieval" / "kv_retrieval_cases.jsonl.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, "wt", encoding="utf-8") as f:
                for c in cases:
                    f.write(json.dumps(c) + "\n")

            model = _make_tiny_qwen2_kv()
            tok = _CharOffsetTokenizer()
            parser = build_pc_parser()

            c1_rep = run_stage_c1(
                parser.parse_args(
                    ["--stage", "c1", "--smoke", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--device", "cpu", "--dtype", "fp32"]
                ),
                model=model,
                tokenizer=tok,
            )
            self.assertTrue(c1_rep["passed"])

            run_stage_eval_all(
                parser.parse_args(
                    ["--stage", "eval_all", "--smoke", "--resume", "--max-new-tokens", "4", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--device", "cpu", "--dtype", "fp32"]
                ),
                model=model,
                tokenizer=tok,
            )
            run_stage_bench_packed(
                parser.parse_args(
                    ["--stage", "bench_packed", "--smoke", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--device", "cpu"]
                )
            )
            analyze_task2(
                build_t2_analyze_parser().parse_args(
                    ["--smoke", "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--seed", "6304"]
                )
            )

            smoke_root = out_dir / "smoke"
            for rel in (
                "results/partC_c1_validation.json",
                "results/partC_kv_fidelity_matched_budget.csv",
                "results/partC_kv_breakdown_factorial.csv",
                "results/partC_bootstrap_ci.json",
                "results/partC_packed_encode_decode_bench.csv",
                "evidence/partC_qualitative_generations.json",
                "final_report_bundle/MANIFEST.json",
            ):
                fp = smoke_root / rel
                self.assertTrue(fp.exists() and fp.stat().st_size > 0, msg=f"Missing {rel}")


if __name__ == "__main__":
    unittest.main()
