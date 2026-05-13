from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:
    from . import _C

    _EXTENSION_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local CUDA build.
    _C = None
    _EXTENSION_IMPORT_ERROR = exc


TensorTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def cuda_extension_available() -> bool:
    return _C is not None


def _validate_common(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    if q.dim() != 3:
        raise ValueError("q must have shape [batch, heads, head_dim]")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must have shape [blocks, heads, block_size, head_dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes")
    if q.shape[1] != k_cache.shape[1]:
        raise ValueError("q and cache tensors must have the same number of heads")
    if q.shape[2] != k_cache.shape[3]:
        raise ValueError("q and cache tensors must have the same head dimension")
    if q.shape[2] > 256:
        raise ValueError("this kernel supports head_dim <= 256")
    if block_tables.dim() != 2:
        raise ValueError("block_tables must have shape [batch, max_blocks_per_sequence]")
    if seq_lens.dim() != 1:
        raise ValueError("seq_lens must have shape [batch]")
    if block_tables.shape[0] != q.shape[0] or seq_lens.shape[0] != q.shape[0]:
        raise ValueError("block_tables and seq_lens must agree with q batch size")
    if block_tables.dtype != torch.int32:
        raise TypeError("block_tables must be torch.int32")
    if seq_lens.dtype != torch.int32:
        raise TypeError("seq_lens must be torch.int32")


def make_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 16,
    seq_lens: Optional[torch.Tensor] = None,
) -> TensorTuple:
    """Pack dense KV tensors into a simple paged cache.

    Args:
        k: Tensor shaped [batch, heads, seq_len, head_dim].
        v: Tensor shaped [batch, heads, seq_len, head_dim].
        block_size: Number of tokens per cache page.
        seq_lens: Optional int32 tensor shaped [batch]. Defaults to full length.

    Returns:
        k_cache, v_cache, block_tables, seq_lens.
    """

    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("k and v must have shape [batch, heads, seq_len, head_dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    batch, heads, seq_len, head_dim = k.shape
    if head_dim > 256:
        raise ValueError("this project supports head_dim <= 256")

    if seq_lens is None:
        seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=k.device)
    else:
        seq_lens = seq_lens.to(device=k.device, dtype=torch.int32).contiguous()
    if seq_lens.numel() and (int(seq_lens.min().item()) < 0 or int(seq_lens.max().item()) > seq_len):
        raise ValueError("seq_lens values must be between 0 and the dense KV sequence length")

    max_seq_len = int(seq_lens.max().item()) if batch else 0
    max_blocks = max(1, math.ceil(max_seq_len / block_size))
    total_blocks = batch * max_blocks

    k_cache = torch.zeros(
        (total_blocks, heads, block_size, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    v_cache = torch.zeros_like(k_cache)
    block_tables = torch.arange(total_blocks, dtype=torch.int32, device=k.device).view(batch, max_blocks)

    for b in range(batch):
        length = min(int(seq_lens[b].item()), seq_len)
        for logical_block in range(math.ceil(length / block_size)):
            start = logical_block * block_size
            end = min(start + block_size, length)
            physical_block = int(block_tables[b, logical_block].item())
            k_cache[physical_block, :, : end - start, :] = k[b, :, start:end, :]
            v_cache[physical_block, :, : end - start, :] = v[b, :, start:end, :]

    return (
        k_cache.contiguous(),
        v_cache.contiguous(),
        block_tables.contiguous(),
        seq_lens.contiguous(),
    )


def paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference implementation using regular PyTorch ops.

    This path is intentionally simple and works on CPU or CUDA. It is used for
    tests and as a fallback when the compiled CUDA extension is unavailable.
    """

    _validate_common(q, k_cache, v_cache, block_tables, seq_lens)
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_tables = block_tables.contiguous()
    seq_lens = seq_lens.contiguous()

    batch, heads, head_dim = q.shape
    block_size = k_cache.shape[2]
    scale = float(scale if scale is not None else 1.0 / math.sqrt(head_dim))
    out = torch.empty_like(q)

    for b in range(batch):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            out[b].zero_()
            continue

        pieces_k = []
        pieces_v = []
        remaining = seq_len
        logical_block = 0
        while remaining > 0:
            physical_block = int(block_tables[b, logical_block].item())
            take = min(block_size, remaining)
            pieces_k.append(k_cache[physical_block, :, :take, :])
            pieces_v.append(v_cache[physical_block, :, :take, :])
            remaining -= take
            logical_block += 1

        k_seq = torch.cat(pieces_k, dim=1).float()
        v_seq = torch.cat(pieces_v, dim=1).float()

        for h in range(heads):
            scores = torch.matmul(k_seq[h], q[b, h].float()) * scale
            probs = torch.softmax(scores, dim=0)
            out[b, h] = torch.matmul(probs, v_seq[h]).to(dtype=q.dtype)

    return out


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Run paged decode attention, using CUDA extension when available."""

    _validate_common(q, k_cache, v_cache, block_tables, seq_lens)
    scale = float(scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))

    can_use_extension = (
        _C is not None
        and q.is_cuda
        and k_cache.is_cuda
        and v_cache.is_cuda
        and block_tables.is_cuda
        and seq_lens.is_cuda
    )
    if can_use_extension:
        return _C.paged_attention_forward(
            q.contiguous(),
            k_cache.contiguous(),
            v_cache.contiguous(),
            block_tables.contiguous(),
            seq_lens.contiguous(),
            scale,
        )

    return paged_attention_reference(q, k_cache, v_cache, block_tables, seq_lens, scale)
