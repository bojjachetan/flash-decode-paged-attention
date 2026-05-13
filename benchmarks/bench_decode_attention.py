import argparse
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flash_decode import (
    cuda_extension_available,
    make_paged_kv_cache,
    paged_attention,
    paged_attention_reference,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark paged decode attention.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    return parser.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    sync(device)

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1_000)
    return statistics.mean(samples), statistics.median(samples), min(samples)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU smoke benchmark.")
        args.device = "cpu"
        args.dtype = "float32"

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    torch.manual_seed(123)
    q = torch.randn(args.batch, args.heads, args.dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.heads, args.seq_len, args.dim, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.heads, args.seq_len, args.dim, device=device, dtype=dtype)
    k_cache, v_cache, block_tables, seq_lens = make_paged_kv_cache(k, v, args.block_size)

    reference = paged_attention_reference(q, k_cache, v_cache, block_tables, seq_lens)
    actual = paged_attention(q, k_cache, v_cache, block_tables, seq_lens)
    max_error = (actual.float() - reference.float()).abs().max().item()

    print("Flash Decode Paged Attention Benchmark")
    print(f"device={device}, dtype={dtype}, extension={cuda_extension_available()}")
    print(
        f"batch={args.batch}, heads={args.heads}, seq_len={args.seq_len}, "
        f"head_dim={args.dim}, block_size={args.block_size}"
    )
    print(f"max_error_vs_reference={max_error:.6g}")

    if device.type == "cuda" and cuda_extension_available():
        mean_ms, median_ms, best_ms = measure(
            lambda: paged_attention(q, k_cache, v_cache, block_tables, seq_lens),
            args.warmup,
            args.iters,
            device,
        )
        print(f"cuda_kernel_ms mean={mean_ms:.3f} median={median_ms:.3f} best={best_ms:.3f}")

    mean_ms, median_ms, best_ms = measure(
        lambda: paged_attention_reference(q, k_cache, v_cache, block_tables, seq_lens),
        max(1, args.warmup // 2),
        max(3, args.iters // 5),
        device,
    )
    print(f"torch_reference_ms mean={mean_ms:.3f} median={median_ms:.3f} best={best_ms:.3f}")


if __name__ == "__main__":
    main()
