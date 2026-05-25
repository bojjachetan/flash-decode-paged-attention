# Correctness, Benchmarks, and Profiling

## Correctness

The test suite validates the paged-cache implementation against dense attention math.

Current correctness status:

```text
python3 -m unittest discover -s tests

OK
```

Coverage:

- Packs dense K/V tensors into paged cache blocks.
- Reconstructs logical sequence order through `block_tables`.
- Compares paged attention output against dense PyTorch attention.
- Validates variable sequence lengths.
- Exercises the public `paged_attention` API.
- Includes a CUDA extension comparison path for CUDA test environments.

## Benchmark Methodology

The benchmark script reports median, mean, and best latency after warmup:

```bash
python3 benchmarks/bench_decode_attention.py \
  --device cuda \
  --batch 8 \
  --heads 16 \
  --seq-len 2048 \
  --dim 128 \
  --block-size 16 \
  --dtype float16 \
  --iters 50 \
  --warmup 10 \
  --format markdown
```

The reported `max_error_vs_reference` compares the selected execution path against the PyTorch reference implementation.

To generate a multi-shape table:

```bash
scripts/benchmark_matrix.sh cuda float16
```

If multiple Python environments are installed, set `PYTHON=/path/to/python`.

## Benchmark Table

Example reference-path measurements:

| Path | Device | Dtype | B | H | S | D | Block | Max Error | Median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch reference | CPU | float32 | 1 | 4 | 128 | 64 | 16 | 0 | 0.101 |
| PyTorch reference | CPU | float32 | 2 | 8 | 256 | 64 | 16 | 0 | 0.466 |
| PyTorch reference | CPU | float32 | 4 | 8 | 512 | 128 | 16 | 0 | 2.115 |
| PyTorch reference | CPU | float32 | 8 | 16 | 2048 | 128 | 16 | 0 | 24.065 |

CUDA benchmark rows can be generated with:

```bash
python3 benchmarks/bench_decode_attention.py --device cuda --format markdown
```

## Profiling

### Nsight Systems

Use Nsight Systems for end-to-end timeline profiling:

```bash
scripts/profile_nsys.sh
```

Useful checks:

- Python benchmark warmup finishes before measured iterations.
- The CUDA extension path launches `paged_attention_kernel`.
- Kernel launches are separated from PyTorch reference work.
- No unexpected host-device synchronization appears inside the measured loop.

### Nsight Compute

Use Nsight Compute for kernel-level metrics:

```bash
scripts/profile_ncu.sh
```

Metrics to inspect:

- achieved occupancy
- DRAM throughput
- L2 hit rate
- warp stall reasons
- instruction mix around reductions and exponentials
- shared-memory usage

### Profiling Summary

From the kernel structure, the main performance pressure is K/V cache streaming. For each `(batch, head)` query, the kernel reads every active token's K and V vector once and keeps only the online-softmax state and output accumulator live. The design avoids writing attention scores to global memory, but long contexts are still handled by a single CUDA block per sequence/head, which limits parallelism for very large `seq_len`.

The most important optimization targets are therefore:

- vectorized K/V loads along `head_dim`
- split-context execution for long sequences
- partial online-softmax reduction across blocks
- grouped-query attention support
- quantized KV-cache reads
