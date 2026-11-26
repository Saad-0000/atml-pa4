"""Zero-shot HellaSwag evaluation matching lm-evaluation-harness conventions."""

from __future__ import annotations

import math
import re
import time
from typing import Any

import torch
import torch.nn.functional as F

from common.timing import get_memory_stats, reset_peak_memory, sync_device

HELLASWAG_TASK_VERSION = "hellaswag_val_locked_lm_eval_v1"
HELLASWAG_PROMPT_FORMAT_SPEC = (
    "query = preprocess(activity_label + ': ' + ctx_a + ' ' + ctx_b.capitalize()); "
    "continuation = ' ' + preprocess(ending); "
    "acc_norm divides conditional log-likelihood by len(preprocess(ending)) "
    "(character count excluding leading space delimiter, not token count)."
)


def preprocess_hellaswag_text(text: str) -> str:
    """Clean HellaSwag WikiHow markup matching lm-evaluation-harness."""
    cleaned = text.strip()
    cleaned = cleaned.replace(" [title]", ". ")
    cleaned = re.sub(r"\[.*?\]", "", cleaned)
    cleaned = cleaned.replace("  ", " ")
    return cleaned


def format_hellaswag_example(
    record: dict[str, Any],
) -> tuple[str, list[str], int, dict[str, Any]]:
    """Extract (context, endings, label, metadata) from a locked manifest record."""
    ex = record["example"] if "example" in record else record
    ctx_a = ex.get("ctx_a", "")
    ctx_b = ex.get("ctx_b", "")
    if ctx_a or ctx_b:
        ctx = f"{ctx_a} {ctx_b.capitalize()}"
    else:
        ctx = ex.get("ctx", "")
    activity_label = ex.get("activity_label", "")
    raw_query = f"{activity_label}: {ctx}" if activity_label else ctx
    context = preprocess_hellaswag_text(raw_query)

    raw_endings = ex["endings"]
    if len(raw_endings) != 4:
        raise ValueError(f"Expected 4 HellaSwag endings, got {len(raw_endings)}.")
    endings = [preprocess_hellaswag_text(e) for e in raw_endings]
    label = int(ex["label"])
    if label not in (0, 1, 2, 3):
        raise ValueError(f"Invalid HellaSwag label: {label}")

    meta = {
        "order": int(record.get("order", -1)),
        "source_row_index": int(record.get("source_row_index", -1)),
        "ind": ex.get("ind"),
        "activity_label": activity_label,
    }
    return context, endings, label, meta


def encode_context_and_ending(
    tokenizer: Any,
    context: str,
    ending: str,
) -> tuple[list[int], int, bool]:
    """Encode (context, ' ' + ending) and return (full_ids, cont_start, used_fallback)."""
    continuation = " " + ending
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    if len(full_ids) > len(ctx_ids) and full_ids[: len(ctx_ids)] == ctx_ids:
        return full_ids, len(ctx_ids), False

    # Fallback when BPE merges across the context/continuation boundary so all
    # four candidate endings share an identical context token prefix.
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not cont_ids:
        cont_ids = tokenizer.encode(ending, add_special_tokens=False)
    if not cont_ids:
        raise ValueError(f"Ending tokenized to empty sequence: {ending!r}")
    return ctx_ids + cont_ids, len(ctx_ids), True


def score_hellaswag_example(
    model: torch.nn.Module,
    tokenizer: Any,
    context: str,
    endings: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """Score all 4 HellaSwag endings via exact conditional log-likelihood."""
    encoded: list[list[int]] = []
    cont_starts: list[int] = []
    boundary_fallbacks = 0

    for ending in endings:
        full_ids, cont_start, used_fb = encode_context_and_ending(
            tokenizer, context, ending
        )
        encoded.append(full_ids)
        cont_starts.append(cont_start)
        boundary_fallbacks += int(used_fb)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    max_len = max(len(seq) for seq in encoded)
    batch_size = len(encoded)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(encoded):
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, :seq_len] = 1

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    log_likelihoods: list[float] = []
    norm_log_likelihoods: list[float] = []
    ending_char_lens: list[int] = []
    ending_token_lens: list[int] = []

    for i, (seq, cont_start, ending) in enumerate(zip(encoded, cont_starts, endings)):
        seq_len = len(seq)
        if cont_start < 1 or cont_start >= seq_len:
            raise ValueError(
                f"Invalid continuation span: cont_start={cont_start}, seq_len={seq_len}"
            )
        # Logits at positions [cont_start - 1, ..., seq_len - 2] predict tokens [cont_start, ..., seq_len - 1]
        pred_logits = logits[i, cont_start - 1 : seq_len - 1, :].float()
        targets = input_ids[i, cont_start:seq_len]
        ll = -float(F.cross_entropy(pred_logits, targets, reduction="sum").item())
        if not math.isfinite(ll):
            raise FloatingPointError(f"Non-finite HellaSwag log-likelihood: {ll}")

        char_len = max(len(ending), 1)
        log_likelihoods.append(ll)
        norm_log_likelihoods.append(ll / char_len)
        ending_char_lens.append(len(ending))
        ending_token_lens.append(seq_len - cont_start)

    pred_raw = int(max(range(4), key=lambda k: log_likelihoods[k]))
    pred_norm = int(max(range(4), key=lambda k: norm_log_likelihoods[k]))

    return {
        "log_likelihoods": log_likelihoods,
        "norm_log_likelihoods": norm_log_likelihoods,
        "ending_char_lens": ending_char_lens,
        "ending_token_lens": ending_token_lens,
        "pred_raw": pred_raw,
        "pred_norm": pred_norm,
        "boundary_fallbacks": boundary_fallbacks,
    }


def evaluate_hellaswag(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    diag_limit: int = 512,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate zero-shot HellaSwag and save per-example candidate scores."""
    model.eval()
    sync_device(device)
    reset_peak_memory(device)
    t0 = time.perf_counter()

    per_example: list[dict[str, Any]] = []
    total_correct_raw = 0
    total_correct_norm = 0
    diag_correct_raw = 0
    diag_correct_norm = 0
    total_boundary_fallbacks = 0

    for idx, rec in enumerate(records):
        context, endings, label, meta = format_hellaswag_example(rec)
        scored = score_hellaswag_example(model, tokenizer, context, endings, device)
        correct_raw = int(scored["pred_raw"] == label)
        correct_norm = int(scored["pred_norm"] == label)

        total_correct_raw += correct_raw
        total_correct_norm += correct_norm
        total_boundary_fallbacks += int(scored["boundary_fallbacks"])
        if idx < diag_limit:
            diag_correct_raw += correct_raw
            diag_correct_norm += correct_norm

        per_example.append(
            {
                "eval_index": idx,
                "order": meta["order"],
                "source_row_index": meta["source_row_index"],
                "ind": meta["ind"],
                "label": label,
                "log_likelihoods": scored["log_likelihoods"],
                "norm_log_likelihoods": scored["norm_log_likelihoods"],
                "ending_char_lens": scored["ending_char_lens"],
                "ending_token_lens": scored["ending_token_lens"],
                "pred_raw": scored["pred_raw"],
                "pred_norm": scored["pred_norm"],
                "correct_raw": correct_raw,
                "correct_norm": correct_norm,
            }
        )

    sync_device(device)
    elapsed_sec = time.perf_counter() - t0
    mem = get_memory_stats(device)

    n_total = len(records)
    n_diag = min(n_total, diag_limit)

    summary = {
        "task_version": HELLASWAG_TASK_VERSION,
        "prompt_format_spec": HELLASWAG_PROMPT_FORMAT_SPEC,
        "num_examples": n_total,
        "acc_raw": float(total_correct_raw / n_total) if n_total else 0.0,
        "acc_norm": float(total_correct_norm / n_total) if n_total else 0.0,
        "correct_raw": int(total_correct_raw),
        "correct_norm": int(total_correct_norm),
        "diag_subset_size": n_diag,
        "diag_acc_raw": float(diag_correct_raw / n_diag) if n_diag else 0.0,
        "diag_acc_norm": float(diag_correct_norm / n_diag) if n_diag else 0.0,
        "boundary_fallback_count": int(total_boundary_fallbacks),
        "eval_time_sec": float(elapsed_sec),
        "peak_allocated_bytes": mem["peak_allocated_bytes"],
        "peak_reserved_bytes": mem["peak_reserved_bytes"],
    }
    return summary, per_example
