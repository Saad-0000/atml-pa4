"""Unit and local smoke tests for Task 2 Part A (attention ops) and Part B (MQAR)."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from common.seeding import DEFAULT_SEED, set_seed
from task2.attention_ops import (
    dense_causal_attention,
    sdpa_causal_attention,
    tiled_online_softmax_attention,
    verify_causality,
    verify_extreme_score_stability,
)
from task2.mqar_data import _fallback_make_example, verify_mqar_invariants
from task2.mqar_model import (
    MQARTransformer,
    build_sparse_mask,
    causal_elu_linear_attention,
    theoretical_attention_cost,
)
from task2.run_part_a import build_arg_parser as build_pa_parser
from task2.run_part_a import run_part_a
from task2.run_part_b import build_arg_parser as build_pb_parser
from task2.run_part_b import run_part_b


def _write_gz_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestTask2PartAB(unittest.TestCase):
    def test_cli_help_flags(self) -> None:
        pa = subprocess.run(
            [sys.executable, "-m", "task2.run_part_a", "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(pa.returncode, 0, msg=pa.stderr)
        for flag in ("--out-dir", "--seed", "--smoke"):
            self.assertIn(flag, pa.stdout)

        pb = subprocess.run(
            [sys.executable, "-m", "task2.run_part_b", "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(pb.returncode, 0, msg=pb.stderr)
        for flag in ("--stage", "--seeds", "--ckpt-every", "--resume", "--data-dir", "--out-dir", "--smoke"):
            self.assertIn(flag, pb.stdout)

    def test_part_a_causality_and_exact_fwd_bwd_agreement(self) -> None:
        set_seed(DEFAULT_SEED)
        dev = torch.device("cpu")
        for fn in (dense_causal_attention, tiled_online_softmax_attention, sdpa_causal_attention):
            self.assertTrue(verify_causality(fn, device=dev)["passed"])
            self.assertTrue(verify_extreme_score_stability(fn, device=dev)["all_finite_under_extreme_scores"])

        q = torch.randn(2, 4, 96, 32, requires_grad=True)
        k = torch.randn(2, 4, 96, 32, requires_grad=True)
        v = torch.randn(2, 4, 96, 32, requires_grad=True)
        grad_out = torch.randn(2, 4, 96, 32)

        o_ref = dense_causal_attention(q, k, v)
        o_ref.backward(grad_out)
        dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

        for fn in (
            lambda a, b, c: tiled_online_softmax_attention(a, b, c, tile_q=32, tile_kv=32),
            sdpa_causal_attention,
        ):
            q2 = q.detach().clone().requires_grad_(True)
            k2 = k.detach().clone().requires_grad_(True)
            v2 = v.detach().clone().requires_grad_(True)
            o2 = fn(q2, k2, v2)
            o2.backward(grad_out)
            self.assertLess((o2 - o_ref).abs().max().item(), 1e-5)
            self.assertLess((q2.grad - dq_ref).abs().max().item(), 1e-5)
            self.assertLess((k2.grad - dk_ref).abs().max().item(), 1e-5)
            self.assertLess((v2.grad - dv_ref).abs().max().item(), 1e-5)

    def test_mqar_invariants_and_b3_param_and_budget_match(self) -> None:
        samples = [
            _fallback_make_example("train", k, reg, i)
            for k in (4, 8, 16)
            for reg in ("near", "far", "mixed")
            for i in range(6)
        ]
        inv = verify_mqar_invariants(samples)
        self.assertTrue(inv["verified"])
        self.assertGreater(inv["far_non_divisible_by_4"], 0)

        # Verify B3 has exact same trainable parameter count as dense & <=64 attended positions/head
        m_dense = MQARTransformer(attn_mode="dense")
        m_b3 = MQARTransformer(attn_mode="b3_local_prefix")
        p_dense = sum(p.numel() for p in m_dense.parameters())
        p_b3 = sum(p.numel() for p in m_b3.parameters())
        self.assertEqual(p_dense, p_b3)

        mask_b3 = build_sparse_mask("b3_local_prefix", 256, 4, torch.device("cpu"))
        self.assertLessEqual(int(mask_b3.sum(dim=-1).max().item()), 64)
        self.assertEqual(theoretical_attention_cost("b3_local_prefix", 256)["max_kv_positions_per_query"], 64)

    def test_linear_elu_chunked_matches_step_recurrence(self) -> None:
        set_seed(DEFAULT_SEED)
        q = torch.randn(1, 2, 40, 16)
        k = torch.randn(1, 2, 40, 16)
        v = torch.randn(1, 2, 40, 16)
        out_chunked = causal_elu_linear_attention(q, k, v, chunk_size=12)

        # Explicit step-by-step recurrence
        qf, kf = F.elu(q) + 1.0, F.elu(k) + 1.0
        S = torch.zeros(1, 2, 16, 16)
        z = torch.zeros(1, 2, 16)
        outs = []
        for t in range(40):
            S = S + torch.einsum("bhd,bhe->bhde", kf[:, :, t], v[:, :, t])
            z = z + kf[:, :, t]
            num = torch.einsum("bhd,bhde->bhe", qf[:, :, t], S)
            den = torch.einsum("bhd,bhd->bh", qf[:, :, t], z).unsqueeze(-1) + 1e-6
            outs.append(num / den)
        out_ref = torch.stack(outs, dim=2)
        self.assertLess((out_chunked - out_ref).abs().max().item(), 1e-5)

    def test_end_to_end_part_a_and_part_b_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            out_dir = root / "task2"

            # Synthetic MQAR dev & test frozen splits
            dev_recs = [
                _fallback_make_example("development", k, reg, i)
                for k in (4, 8, 16)
                for reg in ("near", "far", "mixed")
                for i in range(8)
            ]
            test_recs = [
                _fallback_make_example("test", k, reg, i)
                for k in (4, 8, 16)
                for reg in ("near", "far", "mixed")
                for i in range(8)
            ]
            _write_gz_jsonl(data_dir / "mqar" / "mqar_development.jsonl.gz", dev_recs)
            _write_gz_jsonl(data_dir / "mqar" / "mqar_test.jsonl.gz", test_recs)

            # Part A smoke
            pa_args = build_pa_parser().parse_args(
                ["--smoke", "--out-dir", str(out_dir), "--seed", "6304", "--device", "cpu", "--lengths", "64"]
            )
            run_part_a(pa_args)
            smoke_res = out_dir / "smoke" / "results"
            self.assertTrue((smoke_res / "partA_fidelity_and_timing.csv").exists())
            self.assertTrue((smoke_res / "partA_backends.json").exists())

            # Part B b1, b3, sweep smoke
            for stage in ("b1", "b3"):
                pb_args = build_pb_parser().parse_args(
                    [
                        "--stage", stage,
                        "--seeds", "6304",
                        "--ckpt-every", "20",
                        "--resume",
                        "--smoke",
                        "--data-dir", str(data_dir),
                        "--out-dir", str(out_dir),
                        "--device", "cpu",
                    ]
                )
                run_part_b(pb_args)

            sw_args = build_pb_parser().parse_args(
                ["--stage", "sweep", "--smoke", "--out-dir", str(out_dir), "--seed", "6304", "--device", "cpu"]
            )
            run_part_b(sw_args)

            for fname in (
                "partB_learning_curves.csv",
                "partB_mqar_accuracy_summary.csv",
                "partB_distance_bins.csv",
                "partB_inference_sweep.csv",
            ):
                p = smoke_res / fname
                self.assertTrue(p.exists() and p.stat().st_size > 0, msg=f"Missing {fname}")


if __name__ == "__main__":
    unittest.main()
