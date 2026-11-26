"""Warmed-up, CUDA-synchronized prefill and cached decode benchmarks for Task 1."""

from __future__ import annotations

import statistics
import time
from typing import Any, Sequence

import torch

from common.timing import get_memory_stats, measure_latency, reset_peak_memory, sync_device


def _make_fixed_prompt(
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Create a deterministic fixed-shape batch=1 prompt tensor."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + seq_len)
    low = min(10, max(0, vocab_size - 2))
    high = min(max(low + 1, 1000), vocab_size)
    tokens = torch.randint(low, high, (1, seq_len), generator=gen, dtype=torch.long)
    return tokens.to(device)


def benchmark_prefill_and_decode(
    model: torch.nn.Module,
    device: torch.device,
    prefill_lengths: Sequence[int] = (512, 2048),
    decode_prompt_len: int = 512,
    decode_tokens: int = 64,
    warmup: int = 2,
    reps: int = 5,
    seed: int = 6304,
) -> dict[str, Any]:
    """Benchmark fixed-shape prefill (512, 2048) and 64-token cached decode.

    Separates prefill from decode and reports peak allocated/reserved GPU memory
    separately from logical model storage.
    """
    model.eval()
    vocab_size = int(getattr(model.config, "vocab_size", 1000))

    prefill_results: dict[str, Any] = {}
    with torch.no_grad():
        for length in prefill_lengths:
            prompt = _make_fixed_prompt(int(length), vocab_size, device, seed)

            def _run_prefill(inp: torch.Tensor = prompt) -> Any:
                return model(input_ids=inp, use_cache=True)

            stats = measure_latency(
                _run_prefill,
                warmup=warmup,
                reps=reps,
                device=device,
            )
            stats["prompt_tokens"] = int(length)
            stats["batch_size"] = 1
            stats["use_cache"] = True
            stats["tokens_per_sec"] = (
                float(length) / (stats["mean_ms"] / 1000.0)
                if stats["mean_ms"] > 0
                else 0.0
            )
            prefill_results[str(length)] = stats

        # Cached autoregressive decode benchmark (prefill excluded from decode timer)
        decode_prompt = _make_fixed_prompt(int(decode_prompt_len), vocab_size, device, seed + 17)

        def _single_decode_trial() -> tuple[float, float]:
            sync_device(device)
            t_pre0 = time.perf_counter()
            out = model(input_ids=decode_prompt, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
            sync_device(device)
            t_pre1 = time.perf_counter()

            t_dec0 = time.perf_counter()
            for _ in range(decode_tokens):
                out = model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                next_token = out.logits[:, -1:, :].argmax(dim=-1)
            sync_device(device)
            t_dec1 = time.perf_counter()
            return (t_pre1 - t_pre0) * 1000.0, (t_dec1 - t_dec0) * 1000.0

        for _ in range(warmup):
            _single_decode_trial()
        sync_device(device)

        reset_peak_memory(device)
        decode_times_ms: list[float] = []
        setup_prefill_times_ms: list[float] = []
        for _ in range(reps):
            pre_ms, dec_ms = _single_decode_trial()
            setup_prefill_times_ms.append(pre_ms)
            decode_times_ms.append(dec_ms)
        decode_mem = get_memory_stats(device)

    mean_dec_ms = float(statistics.mean(decode_times_ms))
    decode_result = {
        "unit": "ms",
        "batch_size": 1,
        "prompt_tokens": int(decode_prompt_len),
        "decode_tokens": int(decode_tokens),
        "warmup_reps": int(warmup),
        "timed_reps": int(reps),
        "times_ms": decode_times_ms,
        "mean_ms": mean_dec_ms,
        "median_ms": float(statistics.median(decode_times_ms)),
        "std_ms": float(statistics.stdev(decode_times_ms)) if len(decode_times_ms) > 1 else 0.0,
        "min_ms": float(min(decode_times_ms)),
        "max_ms": float(max(decode_times_ms)),
        "ms_per_token": mean_dec_ms / decode_tokens if decode_tokens > 0 else 0.0,
        "tokens_per_sec": (
            float(decode_tokens) / (mean_dec_ms / 1000.0) if mean_dec_ms > 0 else 0.0
        ),
        "setup_prefill_mean_ms": float(statistics.mean(setup_prefill_times_ms)),
        "peak_allocated_bytes": decode_mem["peak_allocated_bytes"],
        "peak_reserved_bytes": decode_mem["peak_reserved_bytes"],
    }

    overall_peak_allocated = max(
        [decode_result["peak_allocated_bytes"]]
        + [v["peak_allocated_bytes"] for v in prefill_results.values()]
    )
    overall_peak_reserved = max(
        [decode_result["peak_reserved_bytes"]]
        + [v["peak_reserved_bytes"] for v in prefill_results.values()]
    )

    return {
        "execution_path": "dense_fp16_eager_or_sdpa_not_sparse_or_lowbit_kernel",
        "device": str(device),
        "prefill": prefill_results,
        "decode": decode_result,
        "overall_peak_allocated_bytes": int(overall_peak_allocated),
        "overall_peak_reserved_bytes": int(overall_peak_reserved),
    }
