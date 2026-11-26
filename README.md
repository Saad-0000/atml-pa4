# ATML PA4 — Model Compression, Efficient Attention & Speculative Decoding

Reproducible codebase for **EE-5102 / CS-6304 Advanced Topics in Machine Learning (Fall 2026), Programming Assignment 4**.

## Environment & Dependencies

Required pinned versions (from *Shared Experimental Conventions* and `ATML_PA4_Colab_Runner.ipynb`):
- `transformers==4.57.1`
- `huggingface_hub==0.36.2`
- Reference seed: `6304`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Current Stages: Task 1 Complete (Steps 1–9: Baseline, Pruning, Quantization, Analysis)

### Implemented Modules
- `common/seeding.py`: Deterministic seeding (`DEFAULT_SEED = 6304`).
- `common/io.py`: Atomic JSON/JSONL writes, `.jsonl`/`.jsonl.gz` streaming, `local_source_revisions.json` loader, smoke-directory isolation (`resolve_smoke_out_dir`), and condition status ledger (`completed`/`failed`/`unsupported`/`pending`).
- `common/env.py`: Hardware/software/SDPA/HF-revision metadata capture (`configs/env_metadata.json`).
- `common/timing.py`: Warmed-up, CUDA-synchronized wall-clock latency and separate peak allocated/reserved GPU memory tracking.
- `task1/model_utils.py`: `Qwen/Qwen2.5-1.5B` base model loader, eligible projection enumeration (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), and tied-weight deduplication/verification.
- `task1/storage.py`: Logical dense FP16 storage and PA4 16-bit unstructured sparse format accounting (1 mask bit per eligible weight with per-matrix byte alignment + 16 bits per nonzero + FP16 uncompressed unique parameters, counting tied weights once).
- `task1/data.py`: Loaders for `atml-pa4-datasets` locked Task 1 manifests (`c4_calibration_locked`, `c4_development_locked`, `wikitext2_test_windows_locked`, `c4_validation_windows_locked`, `hellaswag_val_locked`).
- `task1/eval_ppl.py`: Exact non-overlapping window perplexity using `score_target_offsets` (excluding token 0, chunked FP32 cross-entropy, saving per-window NLL for paired window-level bootstrap).
- `task1/eval_hellaswag.py`: Zero-shot HellaSwag scoring matching `lm-evaluation-harness` preprocessing and `acc_norm` (character length of preprocessed ending excluding leading space delimiter), saving per-example candidate scores and reporting both full (2,000) and diagnostic (first 512) accuracy.
- `task1/benchmark.py`: Fixed-shape batch-1 prefill (512, 2048 tokens) and 64-token cached autoregressive decode benchmarks with separate prefill/decode timers and peak GPU memory reporting.
- `task1/pruning.py`: Unstructured per-row magnitude (P1), Wanda activation-aware pruning (P2), 2:4 semi-structured Wanda diagnostic (Step 8), calibration local projection output error, and exact matched-budget nonuniform allocator (P3).
- `task1/quantization.py`: Group-wise asymmetric weight-only PTQ (`group_size=128`, FP16 scales + uint8 zero-points), AWQ activation-dependent per-channel scale search with algebraic `down_proj -> up_proj` folding and explicit FP16 compensation vectors, exact packed-byte accounting, and Q3 greedy marginal-error-per-byte mixed-precision allocator.
- `task1/run_baseline.py`, `task1/run_pruning.py`, `task1/run_p3_nonuniform.py`, `task1/run_quantization.py`, `task1/run_q3_mixed.py`, `task1/analyze_results.py`: CLI entrypoints compatible with `ATML_PA4_Colab_Runner.ipynb` supporting `--smoke` and `--resume`.

### Running Local Verification Tests
```bash
python -m unittest discover -s tests -v
```

### Running Task 1 Baseline in Google Colab (T4 / L4 GPU)

1. Run **Section 0** (Cells 0.1–0.5) of `ATML_PA4_Colab_Runner.ipynb` to mount Drive, install `transformers==4.57.1` and `huggingface_hub==0.36.2`, clone `atml-pa4-datasets`, run `python scripts/setup_data.py --public`, and sync this repository.
2. **GPU Smoke Test** (Cell `task1-baseline-smoke` in `ATML_PA4_Colab_Runner.ipynb`):
   ```bash
   python -m task1.run_baseline --smoke \
     --data-dir /content/drive/MyDrive/atml-pa4/data \
     --out-dir /content/drive/MyDrive/atml-pa4/smoke/task1 \
     --seed 6304
   ```
   *(If `--smoke` is invoked against a non-smoke `--out-dir` such as `.../task1`, outputs are automatically redirected to `.../task1/smoke` so smoke runs never overwrite full-run artifacts.)*
3. **Full FP16 Baseline** (Step 1 cell in `ATML_PA4_Colab_Runner.ipynb`):
   ```bash
   python -m task1.run_baseline \
     --data-dir /content/drive/MyDrive/atml-pa4/data \
     --out-dir /content/drive/MyDrive/atml-pa4/task1 \
     --seed 6304
   ```

## Design Choices
- **Smoke-output isolation**: `resolve_smoke_out_dir` preserves explicit smoke paths (e.g., `.../smoke/task1` used by `ATML_PA4_Colab_Runner.ipynb`) and redirects any non-smoke `--out-dir` to `<out-dir>/smoke` when `--smoke` is set.
- **Tied-weight deduplication**: Parameters are deduplicated by `(data_ptr, numel, element_size)` so tied weights (`lm_head.weight` and `model.embed_tokens.weight` in `Qwen/Qwen2.5-1.5B`) are counted exactly once and explicitly recorded as aliases.
- **Unstructured 16-bit sparse storage**: Uses 1 mask bit per eligible weight with byte alignment per eligible matrix (`ceil(numel / 8)`) plus 2 bytes per nonzero eligible weight, plus 2 bytes per unique uncompressed parameter; reported separately from GPU allocated/reserved VRAM.
- **Perplexity VRAM bounding**: FP32 cross-entropy is computed in 128-token sequence chunks over the logits tensor so peak memory stays bounded on T4/L4 GPUs without truncating 512-token contexts.
- **HellaSwag scoring**: Follows `lm-evaluation-harness` WikiHow text cleanup (`[title]` -> `. `, bracket removal, double-space collapse) and normalizes `acc_norm` by `len(preprocess(ending))` (excluding the space delimiter). All four candidate endings per example are scored in a single right-padded batch with explicit `attention_mask` (`use_cache=False`), verified equivalent to unbatched scoring.
- **P3 Nonuniform Pruning (unit, sensitivity probe, and constrained allocation)**:
  - *Allocation unit*: Individual eligible projection (`196` units: 7 projections × 28 layers), allowing distinct sparsities across attention vs. MLP projections as well as across depth.
  - *Candidate sensitivity probes* (computed on the 128-sequence C4 calibration split only): (i) **OWL Layerwise Outlier Distribution (`owl_lod_m5`)** (Yin et al., 2024) measuring the fraction of Wanda scores exceeding `5 × mean`, and (ii) **Relative Calibration Output Error (`calib_rel_l2_err`)** measuring isolated `||(W_pruned - W_fp16) X||_F / ||W_fp16 X||_F` at the target sparsity.
  - *Constrained allocation rule*: Normalized sensitivity `u_l in [-1, 1]` maps to `s_l = clip(s_target - lambda * u_l + delta, 0.10, 0.85)`, where `delta` is solved by bisection so the parameter-weighted total zero count matches uniform P1/P2, followed by integer-weight slack rounding so `zero_count_diff == 0` while preserving `0.10 <= s_l <= 0.85` on every unit.
  - *Selection rule*: Evaluated over `(probe, lambda in {0.04, 0.08, 0.12, 0.16})` strictly on the 32-sequence C4 development manifest (`c4_development_locked.jsonl.gz`); all candidates (including unsuccessful ones) are saved in `results/p3_allocation_and_sensitivity.json`, and the single frozen winner is evaluated on held-out WikiText-2, C4 validation, and HellaSwag.
- **Q1/Q2/Q3 Quantization Conventions & Mixed-Precision Allocation**:
  - *Shared quantizer*: Group-wise asymmetric weight-only PTQ (`group_size=128` along input columns), zero included in `[w_min, w_max]`, FP16 stored scale (all-zero group → `scale=1.0`), uint8 zero-point computed from the stored FP16 scale with ties-to-even rounding (`torch.round`), and exact packed-byte accounting (`ceil(padded_elems * b / 8)` payload + `3` metadata bytes/group + FP16 compensation vectors on non-folded projections + FP16 uncompressed unique weights counted once).
  - *AWQ scale search & folding (Q2)*: Searches `alpha in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` on the 128-sequence C4 calibration split (`s = mean(|X|)^alpha / sqrt(max * min)`), folds `down_proj` inverse input scales algebraically into `up_proj` output rows before `up_proj` quantization, applies `ScaledLinearWrapper` on non-foldable projections (counting their FP16 scale bytes in the matched budget), and verifies full-precision output equivalence before rounding.
  - *Q3 Mixed-Precision Allocation*: Greedy marginal error-reduction-per-byte allocator over `b in {2, 3, 4, 8}` (HAWQ-V2 / BRECQ style), selecting among three calibration error objectives (`mse_per_byte`, `rel_l2_per_byte`, `l2_per_byte`) and minimum-bit floors strictly on the 32-sequence C4 development split while enforcing `achieved_packed_bytes <= Q2_uniform_budget_bytes`.
- **Task 2 Part A (Exact Attention Operators)**:
  - Dense causal attention accumulates `QK^T / sqrt(d)` in FP32 with safe all-masked row handling; online-softmax tiled attention (`tile_q = tile_kv = 128`) avoids materializing the `L x L` score matrix in forward while explicitly documenting that its Python autograd backward retains per-tile intermediates (not a fused FlashAttention kernel); PyTorch SDPA probes `flash_attention`, `efficient_attention`, `cudnn_attention`, and `math` backends explicitly.
- **Task 2 Part B (MQAR & B3 Controlled Intervention)**:
  - Exact 2-layer pre-norm decoder-only Transformer (`d=128`, `H=4`, `d_h=32`, 4× GELU MLP, learned token + sinusoidal positional embeddings, 64-class value head supervised only at query positions) with sparse rolling `latest.pt` checkpoints every 1,500 updates and GradScaler overflow retries that do not consume the 4,500-update budget.
  - *B3 Controlled Intervention (`b3_local_prefix`)*: Motivated by the `2(W-1)=126` receptive-field ceiling of `window64` and the stride-4 alignment blind spots of `local_dilated` on non-divisible-by-4 far distances, 2 heads attend to local `W=64` and 2 heads attend to 32 global prefix anchors `[0, 32)` + 32 recent positions `[i-31, i]` (exact 0% extra trainable parameters, matched 64 attended KV positions/query/head, 0 extra persistent state).

## Attribution
- Course protocol and dataset schemas: `ATML_PA4.pdf` and `atml-pa4-datasets` (`https://github.com/AbDu11aHHH/atml-pa4-datasets`).
- HellaSwag preprocessing and `acc_norm` convention: EleutherAI `lm-evaluation-harness`.
- Pretrained model and tokenizer: Hugging Face `transformers` (`Qwen/Qwen2.5-1.5B`).
