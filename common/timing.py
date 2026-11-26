"""Synchronized timing and GPU memory measurement helpers."""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import torch


def sync_device(device: torch.device | str | None = None) -> None:
    """Synchronize CUDA if the target device is a CUDA device."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def reset_peak_memory(device: torch.device | str | None = None) -> None:
    """Reset CUDA peak memory counters if running on CUDA."""
    if not torch.cuda.is_available():
        return
    if device is not None and torch.device(device).type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(device)


def get_memory_stats(device: torch.device | str | None = None) -> dict[str, int]:
    """Return current and peak allocated/reserved GPU memory in bytes."""
    if not torch.cuda.is_available() or (
        device is not None and torch.device(device).type != "cuda"
    ):
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def measure_latency(
    fn: Callable[[], Any],
    warmup: int = 2,
    reps: int = 5,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Measure warmed-up, CUDA-synchronized wall-clock latency in milliseconds."""
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    for _ in range(warmup):
        fn()
    sync_device(device)

    reset_peak_memory(device)
    times_ms: list[float] = []
    for _ in range(reps):
        sync_device(device)
        t0 = time.perf_counter()
        fn()
        sync_device(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    mem = get_memory_stats(device)
    return {
        "unit": "ms",
        "warmup_reps": warmup,
        "timed_reps": reps,
        "times_ms": times_ms,
        "mean_ms": float(statistics.mean(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "std_ms": float(statistics.stdev(times_ms)) if len(times_ms) > 1 else 0.0,
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "peak_allocated_bytes": mem["peak_allocated_bytes"],
        "peak_reserved_bytes": mem["peak_reserved_bytes"],
    }
