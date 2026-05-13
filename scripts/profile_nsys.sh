#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; run this script on an NVIDIA GPU machine.")
PY

nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output=flash_decode_paged_attention \
  python3 benchmarks/bench_decode_attention.py \
    --device cuda \
    --batch 8 \
    --heads 16 \
    --seq-len 2048 \
    --dim 128 \
    --block-size 16 \
    --iters 100

