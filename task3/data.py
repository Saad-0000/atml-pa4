"""Task 3 dataset loaders, chat formatting, round-robin scheduler, and answer extractors."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from common.io import read_jsonl

TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _resolve_file(data_dir: str | Path, rel: str) -> Path:
    root = Path(data_dir)
    for c in (root / rel, root / "data" / rel):
        if c.exists():
            return c
    raise FileNotFoundError(f"Task 3 dataset file '{rel}' not found under {data_dir}")


def _extract_prompt_text(rec: dict[str, Any]) -> str:
    for key in ("prompt", "question", "user_text", "input", "text"):
        if key in rec and rec[key]:
            return str(rec[key])
    if "example" in rec and isinstance(rec["example"], dict):
        return _extract_prompt_text(rec["example"])
    raise KeyError(f"Could not find prompt text in record keys: {list(rec.keys())}")


def _extract_gsm8k_gold(rec: dict[str, Any]) -> str:
    ex = rec["example"] if ("example" in rec and isinstance(rec["example"], dict)) else rec
    ans = str(ex.get("answer", ex.get("gold", "")))
    m = re.search(r"####\s*([^\n]+)", ans)
    raw = m.group(1).strip() if m else ans.strip()
    return raw.replace(",", "")


def extract_gsm8k_prediction(text: str) -> tuple[str, bool]:
    """Deterministic GSM8K final numeric answer extractor; returns (pred, extractable)."""
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", ""), True
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", ""), True
    return "", False


def check_short_form_match(pred_text: str, rec: dict[str, Any]) -> tuple[bool, bool]:
    """Check (exact_match, format_compliant) for short-form extraction examples."""
    cleaned = pred_text.strip()
    if not cleaned:
        return False, False
    gold = str(rec.get("answer", rec.get("gold", rec.get("target", "")))).strip()
    # Format compliance: concise single-line or matching expected key/value pattern
    compliant = len(cleaned.splitlines()) <= 2 and len(cleaned) <= 120
    if not gold:
        return False, compliant
    em = gold.lower() in cleaned.lower()
    return em, compliant


def load_task3_train_pool(data_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load 500 GSM8K training + 300 short-form training prompts."""
    gsm_path = _resolve_file(data_dir, "public_locked/gsm8k_train_locked.jsonl.gz")
    sf_path = _resolve_file(data_dir, "short_form/short_train.jsonl")

    math_pool = []
    for i, rec in enumerate(read_jsonl(gsm_path)):
        math_pool.append(
            {
                "source_id": str(rec.get("id", rec.get("source_row_index", f"gsm8k_train_{i}"))),
                "category": "math",
                "workload": "math",
                "prompt": _extract_prompt_text(rec),
                "gold": _extract_gsm8k_gold(rec),
                "max_new_tokens": 192,
            }
        )

    short_pool = []
    for i, rec in enumerate(read_jsonl(sf_path)):
        fam = rec.get("template_family", rec.get("family", 0))
        short_pool.append(
            {
                "source_id": str(rec.get("id", f"short_train_{i}")),
                "category": "short_form",
                "workload": "short_train",
                "template_namespace": f"train_fam_{fam}",
                "prompt": _extract_prompt_text(rec),
                "gold": str(rec.get("answer", rec.get("gold", rec.get("target", "")))),
                "max_new_tokens": 32,
            }
        )
    return math_pool, short_pool


def load_task3_test_prompts(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load 120 held-out test prompts: 60 GSM8K + 60 short-form (30 familiar, 30 shifted)."""
    gsm_path = _resolve_file(data_dir, "public_locked/gsm8k_test_locked.jsonl.gz")
    sf_path = _resolve_file(data_dir, "short_form/short_test.jsonl")

    prompts: list[dict[str, Any]] = []
    for i, rec in enumerate(read_jsonl(gsm_path)):
        prompts.append(
            {
                "prompt_id": str(rec.get("id", rec.get("source_row_index", f"gsm8k_test_{i}"))),
                "category": "math",
                "workload": "math",
                "prompt": _extract_prompt_text(rec),
                "gold": _extract_gsm8k_gold(rec),
                "standalone_cap": 512,
                "raw": rec,
            }
        )

    for i, rec in enumerate(read_jsonl(sf_path)):
        split_tag = str(rec.get("split", rec.get("template_split", "familiar" if i < 30 else "shifted"))).lower()
        is_shifted = "shift" in split_tag or "unseen" in split_tag or "held" in split_tag
        workload = "short_shifted" if is_shifted else "short_familiar"
        fam = rec.get("template_family", rec.get("family", 0))
        prompts.append(
            {
                "prompt_id": str(rec.get("id", f"short_test_{i}")),
                "category": "short_form",
                "workload": workload,
                "template_namespace": f"{workload}_fam_{fam}",
                "prompt": _extract_prompt_text(rec),
                "gold": str(rec.get("answer", rec.get("gold", rec.get("target", "")))),
                "standalone_cap": 32,
                "raw": rec,
            }
        )
    return prompts


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Common non-thinking assistant prompt format across all Task 3 conditions."""
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        except Exception:
            pass
    return f"User: {prompt}\nAssistant:"


def get_eos_token_ids(tokenizer: Any, model: Any | None = None) -> set[int]:
    """Return complete fixed EOS token ID set."""
    eos: set[int] = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos.add(int(tokenizer.eos_token_id))
    if model is not None:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "eos_token_id", None) is not None:
            e = gen_cfg.eos_token_id
            if isinstance(e, list):
                eos.update(int(x) for x in e)
            else:
                eos.add(int(e))
    return eos


def verify_tokenizer_agreement(
    target_tok: Any,
    draft_tok: Any,
    prompts: list[str],
) -> dict[str, Any]:
    """Verify full vocab size & chat-template token-ID agreement across all prompts."""
    vocab_match = len(target_tok) == len(draft_tok)
    mismatches = 0
    for p in prompts:
        t_ids = target_tok.encode(format_chat_prompt(target_tok, p), add_special_tokens=False)
        d_ids = draft_tok.encode(format_chat_prompt(draft_tok, p), add_special_tokens=False)
        if t_ids != d_ids:
            mismatches += 1
    return {
        "vocab_size_target": len(target_tok),
        "vocab_size_draft": len(draft_tok),
        "vocab_match": vocab_match,
        "prompts_checked": len(prompts),
        "prompt_id_mismatches": mismatches,
        "passed": vocab_match and mismatches == 0,
    }


class RoundRobinPromptSchedule:
    """Seeded 3:1 math/short-form round-robin schedule matching the 30k/10k quota."""

    def __init__(
        self,
        math_pool: list[dict[str, Any]],
        short_pool: list[dict[str, Any]],
        seed: int = 6304,
    ) -> None:
        rng = random.Random(seed)
        self.math_order = list(range(len(math_pool)))
        self.short_order = list(range(len(short_pool)))
        rng.shuffle(self.math_order)
        rng.shuffle(self.short_order)
        self.math_pool = math_pool
        self.short_pool = short_pool
        self.math_ptr = 0
        self.short_ptr = 0
        self.turn = 0  # 0,1,2 -> math; 3 -> short_form

    def next_prompt(self, math_done: bool, short_done: bool) -> dict[str, Any]:
        if math_done and short_done:
            raise StopIteration("Both math and short-form token quotas are satisfied.")
        want_short = (self.turn % 4 == 3)
        self.turn += 1
        if (want_short and not short_done) or math_done:
            idx = self.short_order[self.short_ptr % len(self.short_order)]
            self.short_ptr += 1
            return self.short_pool[idx]
        idx = self.math_order[self.math_ptr % len(self.math_order)]
        self.math_ptr += 1
        return self.math_pool[idx]

    def state_dict(self) -> dict[str, int]:
        return {
            "math_ptr": self.math_ptr,
            "short_ptr": self.short_ptr,
            "turn": self.turn,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.math_ptr = int(state["math_ptr"])
        self.short_ptr = int(state["short_ptr"])
        self.turn = int(state["turn"])
