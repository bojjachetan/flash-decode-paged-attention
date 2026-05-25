#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cuda}"
DTYPE="${2:-float16}"
PYTHON_BIN="${PYTHON:-python3}"

echo "| device | dtype | B | H | S | D | block | extension | max error | CUDA median ms | reference median ms |"
echo "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"

while read -r B H S D BLOCK; do
  "${PYTHON_BIN}" benchmarks/bench_decode_attention.py \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --batch "${B}" \
    --heads "${H}" \
    --seq-len "${S}" \
    --dim "${D}" \
    --block-size "${BLOCK}" \
    --iters 50 \
    --warmup 10 \
    --format markdown \
    | tail -n 1
done <<'CONFIGS'
1 4 128 64 16
2 8 256 64 16
4 8 512 128 16
8 16 2048 128 16
CONFIGS
