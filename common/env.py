"""Environment metadata collection for reproducible PA4 runs."""

from __future__ import annotations

import datetime
import platform
from pathlib import Path
from typing import Any

import huggingface_hub
import torch
import transformers

from common.io import load_local_source_revisions
from common.seeding import DEFAULT_SEED


def resolve_model_revision(model: Any, model_id: str) -> str:
    """Resolve exact checkpoint commit hash from a loaded model or HF Hub."""
    config = getattr(model, "config", None)
    commit_hash = getattr(config, "_commit_hash", None)
    if commit_hash:
        return str(commit_hash)
    try:
        api = huggingface_hub.HfApi()
        info = api.model_info(model_id)
        if info.sha:
            return str(info.sha)
    except Exception as exc:
        return f"unresolved ({type(exc).__name__})"
    return "unresolved"


def collect_env_metadata(
    data_dir: str | Path | None = None,
    model_revisions: dict[str, str] | None = None,
    seed: int = DEFAULT_SEED,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect software, hardware, SDPA, and dataset revision metadata."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    gpu_vram_bytes = (
        int(torch.cuda.get_device_properties(0).total_memory)
        if cuda_available
        else 0
    )

    sdpa_flags: dict[str, Any] = {}
    if cuda_available:
        sdpa_flags = {
            "sdpa_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "sdpa_mem_efficient_enabled": bool(
                torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            "sdpa_math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        }
    else:
        sdpa_flags = {
            "sdpa_flash_enabled": False,
            "sdpa_mem_efficient_enabled": False,
            "sdpa_math_enabled": True,
        }

    meta: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed_reference": seed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "gpu_name": gpu_name,
        "gpu_vram_bytes": gpu_vram_bytes,
        "transformers_version": transformers.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
        **sdpa_flags,
        "model_revisions": dict(model_revisions or {}),
    }
    if data_dir is not None:
        meta["local_source_revisions"] = load_local_source_revisions(data_dir)
    if extra:
        meta.update(extra)
    return meta
