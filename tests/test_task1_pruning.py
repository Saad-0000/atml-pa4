"""Unit and local smoke tests for Task 1 pruning (P1, P2, 2:4, and P3)."""

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

from common.io import atomic_write_json, read_json
from common.seeding import DEFAULT_SEED, set_seed
from task1.model_utils import enumerate_eligible_projections
from task1.pruning import (
    allocate_nonuniform_sparsities,
    apply_nonuniform_wanda_pruning,
    apply_uniform_pruning,
    collect_calibration_input_norms,
    compute_wanda_scores,
    prune_matrix_2to4,
    restore_eligible_weights,
    snapshot_eligible_weights,
)
from task1.run_p3_nonuniform import build_arg_parser as build_p3_parser
from task1.run_p3_nonuniform import run_p3_nonuniform
from task1.run_pruning import build_arg_parser as build_pruning_parser
from task1.run_pruning import run_pruning


class _ByteTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self) -> int:
        return 256

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [min(255, max(2, b)) for b in text.encode("utf-8")]


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


def _write_gz_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_synthetic_data_dir(root: Path) -> Path:
    data_dir = root / "data"
    atomic_write_json(
        data_dir / "local_source_revisions.json",
        {"c4": "rev_c4", "wikitext": "rev_wt2", "hellaswag": "rev_hs"},
    )
    c4_seqs = [
        {"source_row": i, "token_ids": [10 + ((i + j) % 200) for j in range(16)]}
        for i in range(6)
    ]
    _write_gz_jsonl(data_dir / "public_locked" / "c4_calibration_locked.jsonl.gz", c4_seqs)
    _write_gz_jsonl(data_dir / "public_locked" / "c4_development_locked.jsonl.gz", c4_seqs)

    ppl_recs = [
        {
            "token_ids": [10 + ((i + j) % 200) for j in range(16)],
            "score_target_offsets": [1, 16],
            "scored_tokens": 15,
        }
        for i in range(6)
    ]
    _write_gz_jsonl(
        data_dir / "public_locked" / "wikitext2_test_windows_locked.jsonl.gz", ppl_recs
    )
    _write_gz_jsonl(
        data_dir / "public_locked" / "c4_validation_windows_locked.jsonl.gz", ppl_recs
    )
    hs_recs = [
        {
            "order": i,
            "source_row_index": i * 2,
            "example": {
                "ind": i,
                "activity_label": "Demo",
                "ctx_a": f"Case {i} starts.",
                "ctx_b": "next",
                "endings": [
                    "short one.",
                    "a slightly longer ending.",
                    "third choice here.",
                    "fourth choice.",
                ],
                "label": str(i % 4),
            },
        }
        for i in range(20)
    ]
    _write_gz_jsonl(data_dir / "public_locked" / "hellaswag_val_locked.jsonl.gz", hs_recs)
    return data_dir


class TestTask1Pruning(unittest.TestCase):
    def test_cli_help_flags(self) -> None:
        p1 = subprocess.run(
            [sys.executable, "-m", "task1.run_pruning", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(p1.returncode, 0, msg=p1.stderr)
        for flag in ("--method", "--sparsities", "--run-2to4-at-50", "--data-dir", "--out-dir", "--seed", "--smoke", "--resume"):
            self.assertIn(flag, p1.stdout)

        p3 = subprocess.run(
            [sys.executable, "-m", "task1.run_p3_nonuniform", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(p3.returncode, 0, msg=p3.stderr)
        for flag in ("--sparsities", "--data-dir", "--out-dir", "--seed", "--smoke", "--resume"):
            self.assertIn(flag, p3.stdout)

    def test_magnitude_pruning_per_row_and_excluded_untouched(self) -> None:
        model = _make_tiny_qwen2(num_layers=2)
        clean = snapshot_eligible_weights(model)
        embed_before = model.model.embed_tokens.weight.detach().clone()
        bias_before = model.model.layers[0].self_attn.q_proj.bias.detach().clone()

        rep = apply_uniform_pruning(model, "p1_magnitude", sparsity=0.50)
        self.assertAlmostEqual(rep["achieved_sparsity"], 0.50, places=6)

        # Every row of every eligible projection has exactly 50% zeros
        for item in enumerate_eligible_projections(model):
            row_zeros = (item.module.weight == 0).sum(dim=1)
            expected = item.in_features // 2
            self.assertTrue(torch.all(row_zeros == expected).item())

        # Excluded tensors untouched
        self.assertTrue(torch.equal(model.model.embed_tokens.weight, embed_before))
        self.assertTrue(torch.equal(model.model.layers[0].self_attn.q_proj.bias, bias_before))

        # Restoration brings back exact original weights
        restore_eligible_weights(model, clean)
        for item in enumerate_eligible_projections(model):
            self.assertTrue(torch.equal(item.module.weight, clean[item.full_name]))

    def test_wanda_activation_weighting_and_2to4_structure(self) -> None:
        # Column 0 has tiny weight but huge activation -> Wanda must keep col 0 over col 1
        w = torch.tensor([[0.10, 0.20, 0.30, 0.40]], dtype=torch.float32)
        norms = torch.tensor([100.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        scores = compute_wanda_scores(w, norms)
        zeroed = prune_matrix_2to4(w, scores)
        self.assertEqual(zeroed, 2)
        self.assertNotEqual(float(w[0, 0].item()), 0.0)
        self.assertEqual(float(w[0, 1].item()), 0.0)
        self.assertEqual(float(w[0, 2].item()), 0.0)
        self.assertNotEqual(float(w[0, 3].item()), 0.0)

    def test_p3_nonuniform_allocation_exact_budget_and_bounds(self) -> None:
        model = _make_tiny_qwen2(num_layers=2)
        projs = enumerate_eligible_projections(model)
        # Assign varying sensitivities across the 14 projections
        sens = {p.full_name: float(i + 1) ** 1.5 for i, p in enumerate(projs)}

        for target_s in (0.50, 0.75):
            alloc = allocate_nonuniform_sparsities(
                projections=projs,
                sensitivities=sens,
                target_sparsity=target_s,
                lam=0.12,
                s_min=0.10,
                s_max=0.85,
            )
            self.assertEqual(alloc["zero_count_diff"], 0)
            self.assertGreaterEqual(alloc["min_unit_sparsity"], 0.10 - 1e-9)
            self.assertLessEqual(alloc["max_unit_sparsity"], 0.85 + 1e-9)
            # Highest sensitivity unit must have lower sparsity than lowest sensitivity unit
            by_name = {u["full_name"]: u["sparsity"] for u in alloc["per_unit"]}
            self.assertLess(by_name[projs[-1].full_name], by_name[projs[0].full_name])

            # Applying the allocation yields exact matched zero count on the model
            clean = snapshot_eligible_weights(model)
            dummy_norms = {p.full_name: torch.ones(p.in_features) for p in projs}
            apply_rep = apply_nonuniform_wanda_pruning(
                model, dummy_norms, alloc["zero_counts_by_name"]
            )
            self.assertEqual(apply_rep["total_zeroed_weights"], alloc["target_total_zeros"])
            restore_eligible_weights(model, clean)

    def test_end_to_end_p1_p2_2to4_and_p3_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = _make_synthetic_data_dir(root)
            out_dir = root / "task1"
            model = _make_tiny_qwen2(num_layers=2)
            tok = _ByteTokenizer()

            # P2 Wanda + 2:4 diagnostic smoke
            p_parser = build_pruning_parser()
            p2_args = p_parser.parse_args(
                [
                    "--method",
                    "p2_wanda",
                    "--sparsities",
                    "0.50",
                    "--run-2to4-at-50",
                    "--smoke",
                    "--data-dir",
                    str(data_dir),
                    "--out-dir",
                    str(out_dir),
                    "--seed",
                    "6304",
                    "--dtype",
                    "fp32",
                    "--device",
                    "cpu",
                    "--prefill-lengths",
                    "16",
                    "--decode-tokens",
                    "4",
                    "--warmup-reps",
                    "1",
                    "--timing-reps",
                    "2",
                ]
            )
            p2_res = run_pruning(p2_args, model=model, tokenizer=tok)
            self.assertIn("p2_wanda_s50", p2_res)
            self.assertIn("p2_wanda_2to4_s50", p2_res)
            smoke_res_dir = out_dir / "smoke" / "results"
            self.assertTrue((smoke_res_dir / "structured_2to4_diagnostic.json").exists())
            self.assertTrue((smoke_res_dir / "wanda_score_distributions.json").exists())

            # P3 Nonuniform smoke (reuses cached calib norms)
            p3_parser = build_p3_parser()
            p3_args = p3_parser.parse_args(
                [
                    "--sparsities",
                    "0.50",
                    "0.75",
                    "--smoke",
                    "--data-dir",
                    str(data_dir),
                    "--out-dir",
                    str(out_dir),
                    "--seed",
                    "6304",
                    "--dtype",
                    "fp32",
                    "--device",
                    "cpu",
                    "--prefill-lengths",
                    "16",
                    "--decode-tokens",
                    "4",
                    "--warmup-reps",
                    "1",
                    "--timing-reps",
                    "2",
                ]
            )
            p3_res = run_p3_nonuniform(p3_args, model=model, tokenizer=tok)
            self.assertIn("p3_nonuniform_s50", p3_res)
            self.assertIn("p3_nonuniform_s75", p3_res)
            self.assertEqual(p3_res["p3_nonuniform_s50"]["matched_budget_check"]["zero_count_diff"], 0)
            self.assertEqual(p3_res["p3_nonuniform_s75"]["matched_budget_check"]["zero_count_diff"], 0)
            p3_alloc_doc = read_json(smoke_res_dir / "p3_allocation_and_sensitivity.json")
            self.assertEqual(len(p3_alloc_doc["by_sparsity"]["s50"]["all_candidates_retained"]), 4)


if __name__ == "__main__":
    unittest.main()
