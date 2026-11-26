"""Model loading, eligible-projection enumeration, and tied-weight verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B"
ELIGIBLE_PROJECTION_NAMES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_LAYER_IDX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class EligibleProjection:
    """Reference to an eligible nn.Linear projection inside a Transformer block."""

    layer_idx: int
    full_name: str
    proj_name: str
    module: nn.Linear

    @property
    def out_features(self) -> int:
        return int(self.module.weight.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.module.weight.shape[1])

    @property
    def numel(self) -> int:
        return int(self.module.weight.numel())


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    """Resolve CLI dtype string to torch.dtype."""
    key = dtype_name.lower()
    if key in ("fp16", "float16"):
        return torch.float16
    if key in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype '{dtype_name}'. Use 'fp16' or 'fp32'.")


def load_base_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    dtype_name: str = "fp16",
    device: torch.device | str | None = None,
) -> tuple[Any, Any, torch.device, torch.dtype]:
    """Load Qwen/Qwen2.5-1.5B (base) and tokenizer in eval mode with frozen grads."""
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    torch_dtype = resolve_dtype(dtype_name, dev)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer, dev, torch_dtype


def enumerate_eligible_projections(model: nn.Module) -> list[EligibleProjection]:
    """Enumerate eligible Transformer-block linear projections in canonical order."""
    eligible_set = set(ELIGIBLE_PROJECTION_NAMES)
    found: list[EligibleProjection] = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = full_name.rsplit(".", 1)[-1]
        if leaf_name not in eligible_set:
            continue
        match = _LAYER_IDX_RE.search(full_name)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        found.append(
            EligibleProjection(
                layer_idx=layer_idx,
                full_name=full_name,
                proj_name=leaf_name,
                module=module,
            )
        )
    proj_order = {name: i for i, name in enumerate(ELIGIBLE_PROJECTION_NAMES)}
    found.sort(key=lambda item: (item.layer_idx, proj_order[item.proj_name], item.full_name))
    return found


def deduplicate_parameters(
    model: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], list[dict[str, Any]]]:
    """Return unique parameters (counting tied/shared tensors once) and tied aliases."""
    seen: dict[tuple[int, int, int], str] = {}
    unique_params: list[tuple[str, nn.Parameter]] = []
    tied_aliases: list[dict[str, Any]] = []

    for name, param in model.named_parameters(remove_duplicate=False):
        key = (int(param.data_ptr()), int(param.numel()), int(param.element_size()))
        if key in seen:
            tied_aliases.append(
                {
                    "alias": name,
                    "canonical": seen[key],
                    "numel": int(param.numel()),
                    "dtype": str(param.dtype),
                }
            )
        else:
            seen[key] = name
            unique_params.append((name, param))
    return unique_params, tied_aliases


def verify_model_structure(model: nn.Module) -> dict[str, Any]:
    """Verify eligible matrix enumeration, excluded modules, and tied weights."""
    eligible = enumerate_eligible_projections(model)
    unique_params, tied_aliases = deduplicate_parameters(model)

    config = getattr(model, "config", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    tie_word_embeddings = getattr(config, "tie_word_embeddings", None)

    expected_eligible_count = (
        int(num_hidden_layers) * len(ELIGIBLE_PROJECTION_NAMES)
        if num_hidden_layers is not None
        else len(eligible)
    )
    if len(eligible) != expected_eligible_count:
        raise AssertionError(
            f"Expected {expected_eligible_count} eligible projections "
            f"({num_hidden_layers} layers x {len(ELIGIBLE_PROJECTION_NAMES)}), "
            f"found {len(eligible)}."
        )

    per_layer_counts: dict[int, list[str]] = {}
    eligible_weight_names: set[str] = set()
    for item in eligible:
        if item.module.weight.ndim != 2:
            raise AssertionError(
                f"Eligible projection {item.full_name} weight is not 2D: {item.module.weight.shape}"
            )
        per_layer_counts.setdefault(item.layer_idx, []).append(item.proj_name)
        eligible_weight_names.add(f"{item.full_name}.weight")

    for layer_idx, names in sorted(per_layer_counts.items()):
        if set(names) != set(ELIGIBLE_PROJECTION_NAMES) or len(names) != len(ELIGIBLE_PROJECTION_NAMES):
            raise AssertionError(
                f"Layer {layer_idx} projections mismatch: expected {ELIGIBLE_PROJECTION_NAMES}, got {names}"
            )

    for w_name in eligible_weight_names:
        lowered = w_name.lower()
        if any(forbidden in lowered for forbidden in ("embed", "lm_head", "norm", "bias")):
            raise AssertionError(f"Excluded parameter leaked into eligible set: {w_name}")

    if tie_word_embeddings is True and len(tied_aliases) == 0:
        raise AssertionError(
            "config.tie_word_embeddings is True, but no tied parameter aliases were detected."
        )

    naive_total_params = sum(int(p.numel()) for _, p in model.named_parameters(remove_duplicate=False))
    unique_total_params = sum(int(p.numel()) for _, p in unique_params)
    eligible_total_params = sum(item.numel for item in eligible)
    excluded_unique_params = unique_total_params - eligible_total_params

    bias_params = sum(
        int(p.numel()) for name, p in unique_params if name.endswith(".bias")
    )

    return {
        "verified": True,
        "num_hidden_layers": num_hidden_layers,
        "tie_word_embeddings": tie_word_embeddings,
        "eligible_projection_names": list(ELIGIBLE_PROJECTION_NAMES),
        "eligible_matrix_count": len(eligible),
        "eligible_weight_params": eligible_total_params,
        "excluded_unique_params": excluded_unique_params,
        "excluded_bias_params": bias_params,
        "unique_total_params": unique_total_params,
        "naive_total_params": naive_total_params,
        "tied_aliases": tied_aliases,
        "eligible_matrices": [
            {
                "layer_idx": item.layer_idx,
                "full_name": item.full_name,
                "proj_name": item.proj_name,
                "shape": [item.out_features, item.in_features],
                "numel": item.numel,
                "has_bias": item.module.bias is not None,
            }
            for item in eligible
        ],
    }
