"""MQAR dataset loading, staff generator import, invariant checks, and distance bins."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import Any, Callable

from common.io import read_jsonl

VALUE_OFFSET = 128
NUM_VALUE_CLASSES = 64
DISTANCE_BINS: tuple[tuple[str, int, int], ...] = (
    ("[0,32)", 0, 32),
    ("[32,64)", 32, 64),
    ("[64,128)", 64, 128),
    ("[128,256)", 128, 256),
)


def resolve_mqar_file(data_dir: str | Path, filename: str) -> Path:
    root = Path(data_dir)
    candidates = [
        root / "mqar" / filename,
        root / "data" / "mqar" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"MQAR file '{filename}' not found. Checked: {[str(x) for x in candidates]}")


def load_mqar_split(
    data_dir: str | Path,
    split: str,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load frozen mqar_development.jsonl.gz or mqar_test.jsonl.gz."""
    if split == "development":
        path = resolve_mqar_file(data_dir, "mqar_development.jsonl.gz")
    elif split == "test":
        path = resolve_mqar_file(data_dir, "mqar_test.jsonl.gz")
    else:
        raise ValueError(f"Unknown MQAR frozen split '{split}'.")
    return path, read_jsonl(path)


def _fallback_make_example(
    split: str,
    k: int,
    regime: str,
    index: int,
    seq_len: int = 256,
) -> dict[str, Any]:
    """PA4-compliant deterministic fallback MQAR generator for offline unit tests."""
    split_salt = {"train": 11, "development": 29, "test": 47}.get(split, 73)
    regime_salt = {"near": 101, "far": 211, "mixed": 307}[regime]
    rng = random.Random(6304 * 1_000_003 + split_salt * 9973 + regime_salt * 97 + k * 31 + index)

    keys = rng.sample(range(64, 128), k)
    values = rng.sample(range(128, 192), k)
    tokens = [rng.randint(2, 63) for _ in range(seq_len)]
    for i in range(k):
        tokens[2 * i] = keys[i]
        tokens[2 * i + 1] = values[i]

    order = list(range(k))
    rng.shuffle(order)

    used_slots: set[int] = set(range(2 * k))
    query_positions: list[int] = []
    answer_ids: list[int] = []
    source_positions: list[int] = []
    distances: list[int] = []

    for pair_idx in order:
        src_pos = 2 * pair_idx
        if regime == "near":
            lo = max(2 * k, src_pos + 1)
            hi = min(seq_len - 1, src_pos + 63)
        elif regime == "far":
            lo = max(2 * k, src_pos + 128)
            hi = min(seq_len - 1, src_pos + 250)
        else:
            lo = max(2 * k, src_pos + 1)
            hi = seq_len - 1

        candidates = [p for p in range(lo, hi + 1) if p not in used_slots]
        if not candidates:
            raise RuntimeError(f"No valid query slots for k={k}, regime={regime}, idx={index}")
        q_pos = rng.choice(candidates)
        used_slots.add(q_pos)
        tokens[q_pos] = keys[pair_idx]
        query_positions.append(q_pos)
        answer_ids.append(values[pair_idx])
        source_positions.append(src_pos)
        distances.append(q_pos - src_pos)

    return {
        "tokens": tokens,
        "K": int(k),
        "query_positions": query_positions,
        "answer_ids": answer_ids,
        "source_positions": source_positions,
        "distances": distances,
        "regime": regime,
    }


def load_mqar_generator(data_dir: str | Path | None = None) -> tuple[Callable[..., dict[str, Any]], str]:
    """Load staff scripts/mqar_generator.py (`make_example`) if present, else fallback."""
    candidates: list[Path] = []
    if data_dir is not None:
        d = Path(data_dir)
        candidates.extend([
            d / "scripts" / "mqar_generator.py",
            d.parent / "scripts" / "mqar_generator.py",
        ])
    candidates.append(Path("/content/atml-pa4-datasets/scripts/mqar_generator.py"))

    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("staff_mqar_generator", str(path))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "make_example"):
                    return mod.make_example, str(path)
    return _fallback_make_example, "builtin_fallback"


def verify_mqar_invariants(
    examples: list[dict[str, Any]],
    expected_seq_len: int = 256,
) -> dict[str, Any]:
    """Verify all PA4 PDF (pp. 7-8) MQAR structural and distance invariants."""
    if not examples:
        raise ValueError("Cannot verify invariants on empty example list.")

    far_non_mult4 = 0
    far_total = 0
    for idx, ex in enumerate(examples):
        tokens = ex["tokens"]
        k = int(ex["K"])
        q_pos = ex["query_positions"]
        ans = ex["answer_ids"]
        src = ex["source_positions"]
        dists = ex["distances"]
        regime = ex["regime"]

        if len(tokens) != expected_seq_len:
            raise AssertionError(f"Example {idx}: len(tokens)={len(tokens)} != {expected_seq_len}")
        if not (len(q_pos) == len(ans) == len(src) == len(dists) == k):
            raise AssertionError(f"Example {idx}: query metadata length mismatch for K={k}")
        if len(set(q_pos)) != k:
            raise AssertionError(f"Example {idx}: overlapping query positions {q_pos}")

        prefix_keys = [tokens[2 * i] for i in range(k)]
        prefix_vals = [tokens[2 * i + 1] for i in range(k)]
        if len(set(prefix_keys)) != k or any(not (64 <= x <= 127) for x in prefix_keys):
            raise AssertionError(f"Example {idx}: invalid prefix keys {prefix_keys}")
        if len(set(prefix_vals)) != k or any(not (128 <= x <= 191) for x in prefix_vals):
            raise AssertionError(f"Example {idx}: invalid prefix values {prefix_vals}")

        kv_map = dict(zip(prefix_keys, prefix_vals))
        q_set = set(q_pos)
        for pos in range(2 * k, expected_seq_len):
            if pos not in q_set:
                if not (2 <= tokens[pos] <= 63):
                    raise AssertionError(
                        f"Example {idx}: non-filler token {tokens[pos]} at filler pos {pos}"
                    )

        for qp, a_id, sp, d in zip(q_pos, ans, src, dists):
            if qp - sp != d:
                raise AssertionError(f"Example {idx}: distance {d} != q_pos({qp}) - src({sp})")
            q_key = tokens[qp]
            if kv_map.get(q_key) != a_id:
                raise AssertionError(f"Example {idx}: answer_id {a_id} mismatches key {q_key}")
            if regime == "near" and d >= 64:
                raise AssertionError(f"Example {idx}: near regime has distance {d} >= 64")
            if regime == "far":
                if d < 128 or d > 250:
                    raise AssertionError(f"Example {idx}: far regime distance {d} outside [128, 250]")
                far_total += 1
                if d % 4 != 0:
                    far_non_mult4 += 1

    return {
        "verified": True,
        "num_examples_checked": len(examples),
        "far_queries_checked": far_total,
        "far_non_divisible_by_4": far_non_mult4,
    }


def distance_bin_label(distance: int) -> str:
    for label, lo, hi in DISTANCE_BINS:
        if lo <= distance < hi:
            return label
    return f">={DISTANCE_BINS[-1][2]}"
