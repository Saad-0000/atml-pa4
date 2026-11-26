"""Unit and local smoke tests for shared infrastructure and Task 1 FP16 baseline."""

from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from common.io import atomic_write_json, resolve_smoke_out_dir
from common.seeding import DEFAULT_SEED, set_seed
from task1.benchmark import benchmark_prefill_and_decode
from task1.eval_hellaswag import (
    evaluate_hellaswag,
    format_hellaswag_example,
    preprocess_hellaswag_text,
    score_hellaswag_example,
)
from task1.eval_ppl import evaluate_perplexity
from task1.model_utils import ELIGIBLE_PROJECTION_NAMES, verify_model_structure
from task1.run_baseline import build_arg_parser, run_baseline
from task1.storage import compute_fp16_and_sparse_storage


class _ByteTokenizer:
    """Deterministic lightweight tokenizer for offline unit tests."""

    pad_token_id = 0
    eos_token_id = 1

    def __len__(self) -> int:
        return 256

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [min(255, max(2, b)) for b in text.encode("utf-8")]


def _make_tiny_qwen2(
    num_layers: int = 2,
    tie_word_embeddings: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Qwen2ForCausalLM:
    set_seed(DEFAULT_SEED)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = Qwen2ForCausalLM(cfg).to(dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _write_gz_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestTask1Baseline(unittest.TestCase):
    def test_cli_help_exposes_smoke_and_colab_flags(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "task1.run_baseline", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        for flag in ("--smoke", "--data-dir", "--out-dir", "--seed", "--resume"):
            self.assertIn(flag, proc.stdout)

    def test_resolve_smoke_out_dir_separation(self) -> None:
        self.assertEqual(
            resolve_smoke_out_dir("/tmp/atml-pa4/task1", smoke=False),
            Path("/tmp/atml-pa4/task1"),
        )
        self.assertEqual(
            resolve_smoke_out_dir("/tmp/atml-pa4/task1", smoke=True),
            Path("/tmp/atml-pa4/task1/smoke"),
        )
        self.assertEqual(
            resolve_smoke_out_dir("/tmp/atml-pa4/smoke/task1", smoke=True),
            Path("/tmp/atml-pa4/smoke/task1"),
        )

    def test_eligible_projections_and_tied_weights(self) -> None:
        model = _make_tiny_qwen2(num_layers=2, tie_word_embeddings=True)
        report = verify_model_structure(model)

        self.assertTrue(report["verified"])
        self.assertEqual(report["eligible_matrix_count"], 2 * len(ELIGIBLE_PROJECTION_NAMES))
        self.assertEqual(len(report["tied_aliases"]), 1)
        self.assertEqual(
            report["naive_total_params"] - report["unique_total_params"],
            256 * 32,
        )
        # Qwen2 attention q/k/v projections have biases; verify biases are excluded
        self.assertGreater(report["excluded_bias_params"], 0)
        for entry in report["eligible_matrices"]:
            self.assertIn(entry["proj_name"], ELIGIBLE_PROJECTION_NAMES)

    def test_storage_accounting_dense_and_16bit_sparse(self) -> None:
        model = _make_tiny_qwen2(num_layers=2, tie_word_embeddings=True)
        before = compute_fp16_and_sparse_storage(model)

        self.assertEqual(before["dense_fp16_total_bytes"], 2 * before["unique_total_params"])
        self.assertEqual(before["tied_alias_count"], 1)
        expected_mask_bytes = sum(
            (m["numel"] + 7) // 8 for m in verify_model_structure(model)["eligible_matrices"]
        )
        self.assertEqual(before["sparse_16bit_mask_bytes"], expected_mask_bytes)
        self.assertEqual(
            before["sparse_16bit_total_bytes"],
            before["sparse_16bit_mask_bytes"]
            + 2 * before["eligible_nonzero_params"]
            + before["uncompressed_fp16_bytes"],
        )

        # Zero out half of layer 0 q_proj.weight and verify sparse storage drops by 2 * zeroed
        w = model.model.layers[0].self_attn.q_proj.weight
        half_rows = w.shape[0] // 2
        zeroed_count = int(torch.count_nonzero(w[:half_rows]).item())
        w[:half_rows].zero_()

        after = compute_fp16_and_sparse_storage(model)
        self.assertEqual(after["dense_fp16_total_bytes"], before["dense_fp16_total_bytes"])
        self.assertEqual(after["sparse_16bit_mask_bytes"], before["sparse_16bit_mask_bytes"])
        self.assertEqual(
            before["sparse_16bit_total_bytes"] - after["sparse_16bit_total_bytes"],
            2 * zeroed_count,
        )

    def test_perplexity_exact_matches_reference_and_validates_offsets(self) -> None:
        model = _make_tiny_qwen2(num_layers=2)
        device = torch.device("cpu")
        windows = [
            {
                "token_ids": [10, 20, 30, 40, 50, 60],
                "score_target_offsets": [1, 2, 3, 4, 5],
                "scored_tokens": 5,
            },
            {
                "token_ids": [15, 25, 35, 45, 55],
                "score_target_offsets": [1, 3, 4],
                "scored_tokens": 3,
            },
        ]
        summary, per_win = evaluate_perplexity(
            model, windows, device=device, suite_name="unit_test", vocab_chunk_tokens=2
        )

        # Manual unchunked reference
        ref_nll = 0.0
        with torch.no_grad():
            for w in windows:
                ids = torch.tensor([w["token_ids"]], dtype=torch.long)
                logits = model(input_ids=ids, use_cache=False).logits[0]
                for off in w["score_target_offsets"]:
                    ref_nll += float(
                        F.cross_entropy(
                            logits[off - 1 : off].float(),
                            torch.tensor([w["token_ids"][off]], dtype=torch.long),
                            reduction="sum",
                        ).item()
                    )
        self.assertEqual(summary["scored_tokens"], 8)
        self.assertEqual(len(per_win), 2)
        self.assertAlmostEqual(summary["total_nll"], ref_nll, places=5)
        self.assertAlmostEqual(summary["perplexity"], math.exp(ref_nll / 8), places=4)

        # Verify 2-element half-open [start, end) range format used by atml-pa4-datasets
        range_windows = [
            {
                "token_ids": [10, 20, 30, 40, 50, 60, 70],
                "score_target_offsets": [1, 7],
                "scored_tokens": 6,
            }
        ]
        range_summary, _ = evaluate_perplexity(
            model, range_windows, device=device, suite_name="range_test"
        )
        self.assertEqual(range_summary["scored_tokens"], 6)

        # Offset 0 must be rejected (first token cannot be scored)
        bad_windows = [
            {
                "token_ids": [10, 20, 30],
                "score_target_offsets": [0, 1],
                "scored_tokens": 2,
            }
        ]
        with self.assertRaises(ValueError):
            evaluate_perplexity(model, bad_windows, device=device, suite_name="bad")

    def test_hellaswag_preprocessing_char_norm_and_batch_equivalence(self) -> None:
        self.assertEqual(
            preprocess_hellaswag_text("  Hello [title] world [step] here  "),
            "Hello. world here",
        )
        rec = {
            "order": 0,
            "source_row_index": 42,
            "example": {
                "ind": 100,
                "activity_label": "Cooking",
                "ctx_a": "A chef chops carrots.",
                "ctx_b": "then he",
                "endings": [
                    "puts them in a pot.",
                    "flies to Mars on a bicycle tomorrow morning.",
                    "adds salt.",
                    "stirs the soup gently.",
                ],
                "label": "0",
            },
        }
        ctx, endings, label, meta = format_hellaswag_example(rec)
        self.assertEqual(ctx, "Cooking: A chef chops carrots. Then he")
        self.assertEqual(label, 0)
        self.assertEqual(meta["source_row_index"], 42)

        model = _make_tiny_qwen2(num_layers=2)
        tok = _ByteTokenizer()
        batched = score_hellaswag_example(model, tok, ctx, endings, torch.device("cpu"))

        # Verify batched right-padded scoring equals unbatched single-sequence scoring
        with torch.no_grad():
            for i, ending in enumerate(endings):
                ctx_ids = tok.encode(ctx)
                full_ids = tok.encode(ctx + " " + ending)
                ids = torch.tensor([full_ids], dtype=torch.long)
                logits = model(input_ids=ids, use_cache=False).logits[0]
                start = len(ctx_ids)
                ref_ll = -float(
                    F.cross_entropy(
                        logits[start - 1 : len(full_ids) - 1].float(),
                        ids[0, start:],
                        reduction="sum",
                    ).item()
                )
                self.assertAlmostEqual(batched["log_likelihoods"][i], ref_ll, places=5)
                self.assertAlmostEqual(
                    batched["norm_log_likelihoods"][i],
                    ref_ll / len(ending),
                    places=5,
                )

    def test_end_to_end_smoke_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            out_dir = root / "task1"

            atomic_write_json(
                data_dir / "local_source_revisions.json",
                {"wikitext": "rev_wt2_test", "c4": "rev_c4_test", "hellaswag": "rev_hs_test"},
            )
            ppl_recs = [
                {
                    "token_ids": [10 + ((i + j) % 200) for j in range(16)],
                    "score_target_offsets": list(range(1, 16)),
                    "scored_tokens": 15,
                }
                for i in range(6)
            ]
            _write_gz_jsonl(
                data_dir / "public_locked" / "wikitext2_test_windows_locked.jsonl.gz",
                ppl_recs,
            )
            _write_gz_jsonl(
                data_dir / "public_locked" / "c4_validation_windows_locked.jsonl.gz",
                ppl_recs,
            )
            hs_recs = [
                {
                    "order": i,
                    "source_row_index": i * 3,
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
            _write_gz_jsonl(
                data_dir / "public_locked" / "hellaswag_val_locked.jsonl.gz",
                hs_recs,
            )

            model = _make_tiny_qwen2(num_layers=2, dtype=torch.float32)
            tok = _ByteTokenizer()
            parser = build_arg_parser()
            args = parser.parse_args(
                [
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
                    "32",
                    "--decode-tokens",
                    "4",
                    "--warmup-reps",
                    "1",
                    "--timing-reps",
                    "2",
                ]
            )

            res = run_baseline(args, model=model, tokenizer=tok)
            self.assertEqual(res["status"], "completed")
            self.assertTrue(res["smoke"])
            # Smoke outputs must be redirected to <out_dir>/smoke (not polluting <out_dir>/results)
            smoke_dir = out_dir / "smoke"
            self.assertTrue((smoke_dir / "results" / "baseline_fp16.json").exists())
            self.assertFalse((out_dir / "results" / "baseline_fp16.json").exists())
            # Reduced smoke counts: 4 PPL windows (4 * 15 = 60 tokens), 16 HellaSwag examples
            self.assertEqual(res["quality"]["wikitext2"]["num_windows"], 4)
            self.assertEqual(res["quality"]["wikitext2"]["scored_tokens"], 60)
            self.assertEqual(res["quality"]["hellaswag"]["num_examples"], 16)

            # Verify --resume returns cached result cleanly
            args_resume = parser.parse_args(
                [
                    "--smoke",
                    "--resume",
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
                ]
            )
            res2 = run_baseline(args_resume, model=model, tokenizer=tok)
            self.assertEqual(res2["quality"]["wikitext2"]["perplexity"], res["quality"]["wikitext2"]["perplexity"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU not available")
    def test_tiny_qwen2_fp16_cuda_benchmark_path(self) -> None:
        """Exercise the real CUDA FP16 prefill + cached decode benchmark path on tiny Qwen2."""
        device = torch.device("cuda")
        model = _make_tiny_qwen2(num_layers=2, dtype=torch.float16).to(device)
        bench = benchmark_prefill_and_decode(
            model=model,
            device=device,
            prefill_lengths=(32, 64),
            decode_prompt_len=32,
            decode_tokens=8,
            warmup=1,
            reps=2,
            seed=6304,
        )
        self.assertGreater(bench["prefill"]["32"]["mean_ms"], 0.0)
        self.assertGreater(bench["prefill"]["64"]["mean_ms"], 0.0)
        self.assertGreater(bench["decode"]["mean_ms"], 0.0)
        self.assertGreater(bench["overall_peak_allocated_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
