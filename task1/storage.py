"""Storage and sparsity accounting for Task 1 (dense FP16 and 16-bit sparse model)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from task1.model_utils import deduplicate_parameters, enumerate_eligible_projections


def compute_fp16_and_sparse_storage(model: nn.Module) -> dict[str, Any]:
    """Compute dense FP16 storage and the PA4 16-bit unstructured sparse storage model.

    PA4 specification (p. 4):
    - Count shared/tied parameters once.
    - Leave embeddings, norms, biases, and LM head uncompressed.
    - Unstructured 16-bit sparse model: 1 mask bit per eligible weight (byte-aligned
      per eligible matrix) + 16 bits per nonzero eligible weight, plus uncompressed
      unique parameters in FP16 (2 bytes/param).
    - Reported storage is logical/serialized storage, explicitly separated from GPU VRAM.
    """
    eligible = enumerate_eligible_projections(model)
    unique_params, tied_aliases = deduplicate_parameters(model)

    eligible_ptrs = {int(item.module.weight.data_ptr()) for item in eligible}
    eligible_total_params = 0
    eligible_nonzero_params = 0
    sparse_mask_bytes = 0
    sparse_nonzero_value_bytes = 0

    for item in eligible:
        w = item.module.weight.detach()
        numel = int(w.numel())
        nnz = int(torch.count_nonzero(w).item())
        eligible_total_params += numel
        eligible_nonzero_params += nnz
        # 1 mask bit per eligible weight with byte-alignment padding per matrix
        sparse_mask_bytes += (numel + 7) // 8
        # 16 bits (2 bytes) per nonzero weight
        sparse_nonzero_value_bytes += nnz * 2

    uncompressed_unique_params = 0
    uncompressed_actual_tensor_bytes = 0
    for _, param in unique_params:
        if int(param.data_ptr()) in eligible_ptrs:
            continue
        uncompressed_unique_params += int(param.numel())
        uncompressed_actual_tensor_bytes += int(param.numel()) * int(param.element_size())

    dense_fp16_eligible_bytes = eligible_total_params * 2
    uncompressed_fp16_bytes = uncompressed_unique_params * 2
    dense_fp16_total_bytes = dense_fp16_eligible_bytes + uncompressed_fp16_bytes

    sparse_16bit_eligible_bytes = sparse_mask_bytes + sparse_nonzero_value_bytes
    sparse_16bit_total_bytes = sparse_16bit_eligible_bytes + uncompressed_fp16_bytes

    nonzero_fraction = (
        eligible_nonzero_params / eligible_total_params
        if eligible_total_params > 0
        else 0.0
    )
    achieved_sparsity = 1.0 - nonzero_fraction

    return {
        "storage_accounting_type": "logical_serialized_estimate_not_gpu_vram",
        "tied_weights_counted_once": True,
        "tied_alias_count": len(tied_aliases),
        "unique_total_params": eligible_total_params + uncompressed_unique_params,
        "eligible_weight_params": eligible_total_params,
        "uncompressed_unique_params": uncompressed_unique_params,
        "eligible_nonzero_params": eligible_nonzero_params,
        "eligible_zero_params": eligible_total_params - eligible_nonzero_params,
        "achieved_nonzero_fraction": float(nonzero_fraction),
        "achieved_sparsity": float(achieved_sparsity),
        "nominal_bits_per_eligible_weight": 16.0,
        "dense_fp16_eligible_bytes": int(dense_fp16_eligible_bytes),
        "uncompressed_fp16_bytes": int(uncompressed_fp16_bytes),
        "dense_fp16_total_bytes": int(dense_fp16_total_bytes),
        "serialized_packed_total_bytes": int(dense_fp16_total_bytes),
        "sparse_16bit_mask_bytes": int(sparse_mask_bytes),
        "sparse_16bit_nonzero_value_bytes": int(sparse_nonzero_value_bytes),
        "sparse_16bit_eligible_bytes": int(sparse_16bit_eligible_bytes),
        "sparse_16bit_total_bytes": int(sparse_16bit_total_bytes),
        "sparse_16bit_format_spec": (
            "1 mask bit per eligible weight (byte-aligned per matrix) + "
            "16 bits per nonzero eligible weight + FP16 uncompressed unique parameters"
        ),
        "uncompressed_actual_tensor_bytes": int(uncompressed_actual_tensor_bytes),
    }
