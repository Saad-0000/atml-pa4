"""Perplexity evaluation on locked non-overlapping 512-token windows."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn.functional as F

from common.timing import get_memory_stats, reset_peak_memory, sync_device


def resolve_target_offsets(
    rec: dict[str, Any],
    seq_len: int,
    suite_name: str,
    window_idx: int,
) -> list[int]:
    """Resolve scored target token indices from a locked perplexity window record.

    Supports both:
    - 2-element contiguous range [start, end) (half-open, end - start == scored_tokens)
      or [start, end] (inclusive, end - start + 1 == scored_tokens), as used by
      atml-pa4-datasets;
    - explicit list of target indices of length scored_tokens.
    """
    raw_offsets = rec.get("score_target_offsets")
    if raw_offsets is None:
        offsets = list(range(1, seq_len))
        expected_scored = int(rec.get("scored_tokens", len(offsets)))
    else:
        expected_scored = int(rec.get("scored_tokens", len(raw_offsets)))
        if len(raw_offsets) == 2:
            start, end = int(raw_offsets[0]), int(raw_offsets[1])
            if end - start == expected_scored and 1 <= start < end <= seq_len:
                offsets = list(range(start, end))
            elif end - start + 1 == expected_scored and 1 <= start <= end < seq_len:
                offsets = list(range(start, end + 1))
            elif expected_scored == 2:
                offsets = [start, end]
            else:
                raise ValueError(
                    f"Suite '{suite_name}' window {window_idx}: cannot interpret "
                    f"score_target_offsets={raw_offsets} with scored_tokens={expected_scored} "
                    f"and seq_len={seq_len}."
                )
        else:
            offsets = [int(o) for o in raw_offsets]

    if len(offsets) != expected_scored:
        raise ValueError(
            f"Suite '{suite_name}' window {window_idx}: resolved {len(offsets)} offsets "
            f"!= scored_tokens={expected_scored} (raw={raw_offsets})."
        )
    if len(set(offsets)) != len(offsets):
        raise ValueError(
            f"Suite '{suite_name}' window {window_idx} contains duplicate score_target_offsets."
        )
    if any(o < 1 or o >= seq_len for o in offsets):
        raise ValueError(
            f"Suite '{suite_name}' window {window_idx} has out-of-range offset "
            f"(must satisfy 1 <= offset < {seq_len})."
        )
    return offsets


def evaluate_perplexity(
    model: torch.nn.Module,
    windows: list[dict[str, Any]],
    device: torch.device,
    suite_name: str,
    vocab_chunk_tokens: int = 128,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate exact token-pooled perplexity over fixed non-overlapping windows.

    Conventions (ATML PA4 PDF p. 3):
    - Each window is evaluated independently (use_cache=False; no cross-document state).
    - Each target token in score_target_offsets is scored exactly once.
    - The first token of every sequence (offset 0) is excluded from the denominator.
    - Perplexity is exp(sum(NLL) / N) where N is total scored target tokens.
    """
    model.eval()
    sync_device(device)
    reset_peak_memory(device)
    t0 = time.perf_counter()

    total_nll = 0.0
    total_scored_tokens = 0
    per_window: list[dict[str, Any]] = []

    with torch.no_grad():
        for idx, rec in enumerate(windows):
            token_ids = rec["token_ids"]
            seq_len = len(token_ids)
            if seq_len < 2:
                raise ValueError(
                    f"Suite '{suite_name}' window {idx} has length {seq_len} < 2."
                )

            offsets = resolve_target_offsets(rec, seq_len, suite_name, idx)

            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[0]  # [seq_len, vocab]

            # Score target positions t using logits at t - 1, in FP32 chunks to bound VRAM.
            pred_positions = torch.tensor(
                [o - 1 for o in offsets], dtype=torch.long, device=device
            )
            target_ids = torch.tensor(
                [token_ids[o] for o in offsets], dtype=torch.long, device=device
            )

            win_nll = 0.0
            num_targets = int(target_ids.numel())
            for start in range(0, num_targets, vocab_chunk_tokens):
                end = min(start + vocab_chunk_tokens, num_targets)
                chunk_logits = logits.index_select(0, pred_positions[start:end]).float()
                chunk_targets = target_ids[start:end]
                win_nll += float(
                    F.cross_entropy(chunk_logits, chunk_targets, reduction="sum").item()
                )

            if not math.isfinite(win_nll):
                raise FloatingPointError(
                    f"Non-finite NLL ({win_nll}) on suite '{suite_name}' window {idx}."
                )

            total_nll += win_nll
            total_scored_tokens += num_targets

            win_record: dict[str, Any] = {
                "window_index": idx,
                "seq_len": seq_len,
                "scored_tokens": num_targets,
                "nll_sum": win_nll,
                "mean_nll": win_nll / num_targets,
            }
            for meta_key in ("window_id", "doc_id", "id", "source_row"):
                if meta_key in rec:
                    win_record[meta_key] = rec[meta_key]
            per_window.append(win_record)

    sync_device(device)
    elapsed_sec = time.perf_counter() - t0
    mem = get_memory_stats(device)

    mean_nll = total_nll / total_scored_tokens
    ppl = math.exp(mean_nll)

    summary = {
        "suite": suite_name,
        "num_windows": len(windows),
        "scored_tokens": int(total_scored_tokens),
        "total_nll": float(total_nll),
        "mean_nll": float(mean_nll),
        "perplexity": float(ppl),
        "eval_time_sec": float(elapsed_sec),
        "peak_allocated_bytes": mem["peak_allocated_bytes"],
        "peak_reserved_bytes": mem["peak_reserved_bytes"],
    }
    return summary, per_window
