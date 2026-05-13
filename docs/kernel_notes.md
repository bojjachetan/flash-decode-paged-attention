# Kernel Notes

## Problem

Decode attention computes one output token per active sequence and head:

```text
softmax(q @ K_cache.T / sqrt(head_dim)) @ V_cache
```

The interesting inference detail is the KV cache layout. Instead of storing every request in a single dense `[batch, heads, seq, dim]` slab, serving systems keep token blocks in a shared cache and use a block table to map each logical sequence block to a physical cache block.

This project models that layout as:

```text
k_cache:      [num_blocks, heads, block_size, head_dim]
v_cache:      [num_blocks, heads, block_size, head_dim]
block_tables: [batch, max_blocks_per_sequence]
seq_lens:     [batch]
q:            [batch, heads, head_dim]
out:          [batch, heads, head_dim]
```

## Current Algorithm

Each CUDA block owns one `(batch, head)` pair. Threads map to head dimensions. For every cached token, the block:

1. computes `q dot k[token]` using a block reduction
2. updates a numerically stable online softmax state
3. streams the corresponding `v[token]` into the output accumulator

The online softmax update avoids storing all attention scores:

```text
new_m = max(old_m, score)
alpha = exp(old_m - new_m)
beta = exp(score - new_m)
sum = sum * alpha + beta
acc = acc * alpha + beta * value
```

At the end, `acc / sum` is written to output.

## Tradeoffs

This first version prioritizes clarity:

- one block per query/head is simple and easy to inspect
- the full K/V stream is read once
- no attention matrix is materialized
- head dimensions up to 256 are supported
- very long contexts may underuse the GPU because a single block handles a whole sequence/head

## Good Follow-Up Optimizations

- Add vectorized loads for common dimensions such as 64, 128, and 256.
- Split long contexts across multiple blocks, then reduce partial softmax states.
- Add support for grouped-query attention where query heads share KV heads.
- Add int8 or FP8 KV cache paths with per-block scales.
- Benchmark memory bandwidth and achieved occupancy with Nsight Compute.
- Add causal, prefix, and sliding-window mask variants.
- Implement a small scheduler benchmark that simulates many mixed-length requests.

