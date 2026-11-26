"""Atomic file I/O, dataset readers, and condition status tracking."""

from __future__ import annotations

import datetime
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

TASK_SUBDIRS = ("results", "checkpoints", "plots", "configs", "logs", "evidence")


def ensure_task_dirs(out_dir: str | Path) -> dict[str, Path]:
    """Create standard PA4 output subdirectories under out_dir."""
    root = Path(out_dir)
    paths: dict[str, Path] = {"root": root}
    root.mkdir(parents=True, exist_ok=True)
    for sub in TASK_SUBDIRS:
        sub_path = root / sub
        sub_path.mkdir(parents=True, exist_ok=True)
        paths[sub] = sub_path
    return paths


def resolve_smoke_out_dir(out_dir: str | Path, smoke: bool) -> Path:
    """Ensure smoke outputs never collide with full-run directories.

    If smoke is True and 'smoke' is not already a path component of out_dir,
    redirect outputs into <out_dir>/smoke.
    """
    path = Path(out_dir)
    if not smoke:
        return path
    lower_parts = [p.lower() for p in path.parts]
    if "smoke" in lower_parts:
        return path
    return path / "smoke"


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text atomically via a temporary sibling file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return target


def atomic_write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Serialize JSON atomically via a temporary sibling file."""
    serialized = json.dumps(payload, indent=indent, sort_keys=False) + "\n"
    return atomic_write_text(path, serialized)


def atomic_write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write newline-delimited JSON records atomically."""
    lines = [json.dumps(rec, sort_keys=False) for rec in records]
    text = ("\n".join(lines) + "\n") if lines else ""
    return atomic_write_text(path, text)


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records from a .jsonl or .jsonl.gz file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {p}:{line_num}: {exc}") from exc


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load records from a .jsonl or .jsonl.gz file with an optional limit."""
    records: list[dict[str, Any]] = []
    for rec in iter_jsonl(path):
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    return records


def load_local_source_revisions(data_dir: str | Path) -> dict[str, Any]:
    """Load data/local_source_revisions.json produced by setup_data.py --public."""
    data_root = Path(data_dir)
    candidates = [
        data_root / "local_source_revisions.json",
        data_root / "data" / "local_source_revisions.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return {
                "found": True,
                "path": str(candidate),
                "revisions": read_json(candidate),
            }
    return {
        "found": False,
        "path": str(candidates[0]),
        "revisions": None,
    }


VALID_CONDITION_STATUSES = {"completed", "failed", "unsupported", "pending"}


def update_condition_status(
    status_path: str | Path,
    condition_key: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record condition status (completed/failed/unsupported/pending) atomically."""
    if status not in VALID_CONDITION_STATUSES:
        raise ValueError(
            f"Invalid condition status '{status}'. Must be one of {sorted(VALID_CONDITION_STATUSES)}."
        )
    path = Path(status_path)
    ledger: dict[str, Any] = {}
    if path.exists():
        try:
            ledger = read_json(path)
        except Exception:
            ledger = {}
    ledger[condition_key] = {
        "status": status,
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **(details or {}),
    }
    atomic_write_json(path, ledger)
    return ledger
