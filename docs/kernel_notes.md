# Kernel Notes

## Operation

Decode attention computes one output token per active sequence and attention head:

```text
softmax(q @ K_cache.T / sqrt(head_dim)) @ V_cache
```

The CUDA kernel computes this directly over a paged KV cache. It does not build a dense `[seq_len]` score vector or a `[seq_len, head_dim]` temporary attention matrix.

## Thread and Block Mapping

The kernel launch uses a 2D grid:

```text
grid.x = batch
grid.y = heads
```

Each CUDA block owns one `(batch, head)` decode query:

```text
blockIdx.x -> batch index b
blockIdx.y -> head index h
threadIdx.x -> feature dimension work
```

The query vector `q[b, h, :]` is loaded into shared memory once. Then the CUDA block streams through the sequence tokens, resolving each token through the block table:

```text
logical_block = token / block_size
block_offset  = token - logical_block * block_size
physical_block = block_tables[b, logical_block]
cache_offset = (((physical_block * heads + h) * block_size + block_offset) * head_dim)
```

## Dot Product Reduction

For each cached token, every thread handles one or more dimensions of the `q dot k` computation. The current implementation supports `head_dim <= 256`, so a 256-thread block can cover common head sizes directly.

The partial products are reduced with warp shuffles and a small shared-memory cross-warp reduction:

```text
partial_dot = q_shared[dim] * k_cache[token, dim]
dot = block_reduce_sum(partial_dot)
score = dot / sqrt(head_dim)
```

This keeps the reduction inside the CUDA block and avoids global-memory scratch space.

## Online Softmax

A direct implementation would store all scores, run softmax, then multiply by V:

```text
scores[t] = q dot k[t]
probs = softmax(scores)
out = sum_t probs[t] * v[t]
```

The kernel instead uses online softmax. It keeps three running values:

```text
m   = running maximum score
sum = running softmax denominator
acc = running output accumulator
```

For a new score `s`:

```text
new_m = max(m, s)
alpha = exp(m - new_m)
beta  = exp(s - new_m)

sum = sum * alpha + beta
acc = acc * alpha + beta * v
m = new_m
```

At the end:

```text
out = acc / sum
```

This is numerically stable because scores are always exponentiated relative to the current maximum. It also lets the kernel stream through K/V once without storing the attention scores.

## Memory Access Pattern

For a fixed `(batch, head)`, tokens are visited in logical order. The block table maps each logical block to a physical block in the cache:

```text
k_cache[physical_block, head, block_offset, dim]
v_cache[physical_block, head, block_offset, dim]
```

For each token, adjacent threads load adjacent feature dimensions, so the per-token K/V vector reads are contiguous along `head_dim`. The block-table lookup adds one level of indirection at block boundaries.

## Current Kernel Scope

- Dtypes: `float32`, `float16`, `bfloat16` through PyTorch dispatch.
- Head dimension: up to 256.
- Output: one decode token per `(batch, head)`.
- Attention: full prefix attention over `seq_lens[b]` tokens.
- Inputs: contiguous tensors.

## Limitations

- One CUDA block processes a full sequence/head, so very long contexts do not split work across multiple thread blocks.
- The kernel prioritizes readability over vectorized memory instructions.
- It does not implement grouped-query attention.
- It does not implement int8/FP8 KV-cache quantization.
- It does not include causal, sliding-window, or prefix-mask variants.

## Next Steps

- Add vectorized loads for head dimensions such as 64, 128, and 256.
- Split long contexts across multiple blocks and reduce partial online-softmax states.
- Add grouped-query attention support.
- Add quantized KV-cache variants with per-block scales.
- Add mask variants for sliding-window and prefix attention.
- Add a scheduler benchmark with mixed sequence lengths.

