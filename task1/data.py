"""Dataset loaders for Task 1 locked manifests (atml-pa4-datasets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.io import read_jsonl

TASK1_MANIFESTS: dict[str, str] = {
    "c4_calibration": "public_locked/c4_calibration_locked.jsonl.gz",
    "c4_development": "public_locked/c4_development_locked.jsonl.gz",
    "wikitext2": "public_locked/wikitext2_test_windows_locked.jsonl.gz",
    "c4_validation": "public_locked/c4_validation_windows_locked.jsonl.gz",
    "hellaswag": "public_locked/hellaswag_val_locked.jsonl.gz",
}


def resolve_manifest_path(data_dir: str | Path, key: str) -> Path:
    """Resolve a Task 1 manifest path under data_dir (or data_dir/data)."""
    if key not in TASK1_MANIFESTS:
        raise KeyError(f"Unknown Task 1 manifest key '{key}'.")
    rel = TASK1_MANIFESTS[key]
    root = Path(data_dir)
    candidates = [root / rel, root / "data" / rel]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Manifest '{key}' not found. Checked: {[str(c) for c in candidates]}"
    )


def load_ppl_windows(
    data_dir: str | Path,
    suite: str,
    limit: int | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load locked perplexity evaluation windows ('wikitext2' or 'c4_validation')."""
    if suite not in ("wikitext2", "c4_validation"):
        raise ValueError(f"Unknown perplexity suite '{suite}'.")
    path = resolve_manifest_path(data_dir, suite)
    records = read_jsonl(path, limit=limit)
    if not records:
        raise ValueError(f"Perplexity manifest is empty: {path}")
    return path, records


def load_hellaswag_val(
    data_dir: str | Path,
    limit: int | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load locked HellaSwag validation manifest (2,000 IDs; first 512 = diagnostic)."""
    path = resolve_manifest_path(data_dir, "hellaswag")
    records = read_jsonl(path, limit=limit)
    if not records:
        raise ValueError(f"HellaSwag manifest is empty: {path}")
    return path, records
