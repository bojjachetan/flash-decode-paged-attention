# Flash Decode Paged Attention

CUDA/PyTorch implementation of decode-time attention over a paged KV cache for LLM inference.

This project was originally built as part of an undergraduate GPU programming project. It implements the core decode-attention operation used when an autoregressive model generates one token at a time while attending over a growing KV cache.

The KV cache is stored in fixed-size pages/blocks, and a block table maps each sequence's logical token blocks to physical cache blocks. The CUDA kernel computes attention directly over that paged layout:

- one CUDA block per `(batch, head)` decode query
- paged KV cache lookup through a block table
- numerically stable online softmax
- CPU/PyTorch reference implementation for correctness
- benchmark script that compares extension latency against the reference path
- profiling hook for `nsys`

## Project Layout

```text
csrc/
  bindings.cpp                 PyTorch extension bindings
  paged_attention_kernel.cu    CUDA kernel and launch checks
flash_decode/
  attention.py                 public API, CPU reference, CUDA dispatch
benchmarks/
  bench_decode_attention.py    latency and correctness benchmark
tests/
  test_attention.py            correctness tests for paged attention
docs/
  kernel_notes.md              implementation notes and extension ideas
scripts/
  profile_nsys.sh              Nsight Systems profiling helper
```

## Install

```bash
python3 -m pip install -e .
python3 - <<'PY'
import torch
import flash_decode
print(torch.cuda.is_available())
print(flash_decode.cuda_extension_available())
PY
```

## Run Tests

```bash
python3 -m unittest discover -s tests
```

## Run Benchmarks

Reference benchmark:

```bash
python3 benchmarks/bench_decode_attention.py --device cpu
```

CUDA benchmark:

```bash
python3 benchmarks/bench_decode_attention.py --device cuda --batch 8 --heads 16 --dim 128 --seq-len 2048 --block-size 16
```

Profile with Nsight Systems:

```bash
scripts/profile_nsys.sh
```

## API Example

```python
import torch
from flash_decode import make_paged_kv_cache, paged_attention

B, H, S, D = 4, 8, 1024, 128
q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

k_cache, v_cache, block_tables, seq_lens = make_paged_kv_cache(k, v, block_size=16)
out = paged_attention(q, k_cache, v_cache, block_tables, seq_lens)
```

## Current Kernel Scope

The first kernel is intentionally readable and correctness-focused:

- supports `float32`, `float16`, and `bfloat16` tensors when compiled through PyTorch
- supports head dimensions up to 256
- assumes contiguous tensors
- emits one output token per sequence/head
- uses one pass through K/V with online softmax
