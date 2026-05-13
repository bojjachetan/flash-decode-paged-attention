# Flash Decode Paged Attention

CUDA/PyTorch implementation of decode-time attention over a paged KV cache, built as a portfolio project for GPU inference and LLM systems roles.

During autoregressive decoding, each sequence usually generates one token at a time while attending over a growing KV cache. Production inference engines such as vLLM-style servers store that cache in pages/blocks so memory can be reused across requests. This repo implements the core idea in a compact kernel:

- one CUDA block per `(batch, head)` decode query
- paged KV cache lookup through a block table
- numerically stable online softmax
- CPU/PyTorch reference implementation for correctness
- benchmark script that compares extension latency against the reference path
- profiling hook for `nsys`

The local machine used to create this repo did not have `nvcc` or an NVIDIA GPU, so the CUDA extension is written and packaged but the runnable validation path falls back to PyTorch CPU. On a CUDA machine, `pip install -e .` will build the extension automatically.

## Why This Project

This is closer to real LLM inference work than a generic parallel algorithm demo. It touches the pieces inference teams care about:

- memory layout for KV cache blocks
- avoiding materializing the full attention matrix
- stable softmax in a streaming pass
- Python integration through a PyTorch extension
- reproducible correctness tests and benchmark commands
- room for meaningful follow-up optimizations

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
  test_attention.py            CPU tests plus optional CUDA extension checks
docs/
  kernel_notes.md              implementation notes and extension ideas
scripts/
  profile_nsys.sh              Nsight Systems profiling helper
```

## Install

CPU-only development:

```bash
python3 -m pip install -e .
```

CUDA development:

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

The tests always validate the reference implementation. If CUDA is available and the extension built correctly, they also compare the CUDA kernel against the reference.

## Run Benchmarks

CPU smoke check:

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

Good next steps are listed in [docs/kernel_notes.md](docs/kernel_notes.md), including vectorized loads, split-K for long contexts, causal/prefix masks, and quantized KV cache support.

