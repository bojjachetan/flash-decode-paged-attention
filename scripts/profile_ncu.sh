#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA device is required for Nsight Compute profiling.")
PY

ncu \
  --set full \
  --kernel-name regex:paged_attention_kernel \
  --target-processes all \
  --force-overwrite \
  --export flash_decode_paged_attention_ncu \
  "${PYTHON_BIN}" benchmarks/bench_decode_attention.py \
    --device cuda \
    --batch 8 \
    --heads 16 \
    --seq-len 2048 \
    --dim 128 \
    --block-size 16 \
    --dtype float16 \
    --iters 20 \
    --warmup 5
