#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

namespace {

__inline__ __device__ float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__inline__ __device__ float block_reduce_sum(float value) {
  __shared__ float shared[32];
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp_id = threadIdx.x >> 5;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp_id] = value;
  }
  __syncthreads();

  value = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
  if (warp_id == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

template <typename scalar_t>
__global__ void paged_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    scalar_t* __restrict__ out,
    int batch,
    int heads,
    int head_dim,
    int block_size,
    int max_blocks_per_sequence,
    float scale) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;

  extern __shared__ float q_shared[];
  if (tid < head_dim) {
    q_shared[tid] = static_cast<float>(q[(b * heads + h) * head_dim + tid]);
  }

  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ float alpha;
  __shared__ float beta;

  if (tid == 0) {
    running_max = -INFINITY;
    running_sum = 0.0f;
  }

  float acc = 0.0f;
  const int seq_len = seq_lens[b];
  __syncthreads();

  for (int token = 0; token < seq_len; ++token) {
    const int logical_block = token / block_size;
    const int block_offset = token - logical_block * block_size;
    const int physical_block = block_tables[b * max_blocks_per_sequence + logical_block];
    const int cache_offset =
        (((physical_block * heads + h) * block_size + block_offset) * head_dim);

    float partial_dot = 0.0f;
    if (tid < head_dim) {
      partial_dot = q_shared[tid] * static_cast<float>(k_cache[cache_offset + tid]);
    }

    const float dot = block_reduce_sum(partial_dot);

    if (tid == 0) {
      const float score = dot * scale;
      const float new_max = fmaxf(running_max, score);
      alpha = __expf(running_max - new_max);
      beta = __expf(score - new_max);
      running_sum = running_sum * alpha + beta;
      running_max = new_max;
    }
    __syncthreads();

    if (tid < head_dim) {
      const float value = static_cast<float>(v_cache[cache_offset + tid]);
      acc = acc * alpha + beta * value;
    }
    __syncthreads();
  }

  if (tid < head_dim) {
    const float normalized = seq_len > 0 ? acc / running_sum : 0.0f;
    out[(b * heads + h) * head_dim + tid] = static_cast<scalar_t>(normalized);
  }
}

void validate_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens) {
  TORCH_CHECK(q.dim() == 3, "q must have shape [batch, heads, head_dim]");
  TORCH_CHECK(k_cache.dim() == 4, "k_cache must have shape [blocks, heads, block_size, head_dim]");
  TORCH_CHECK(v_cache.dim() == 4, "v_cache must have shape [blocks, heads, block_size, head_dim]");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache shape mismatch");
  TORCH_CHECK(q.size(1) == k_cache.size(1), "head count mismatch");
  TORCH_CHECK(q.size(2) == k_cache.size(3), "head_dim mismatch");
  TORCH_CHECK(q.size(2) <= 256, "head_dim > 256 is not supported by this demo kernel");
  TORCH_CHECK(block_tables.dim() == 2, "block_tables must have shape [batch, max_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [batch]");
  TORCH_CHECK(block_tables.size(0) == q.size(0), "block_tables batch mismatch");
  TORCH_CHECK(seq_lens.size(0) == q.size(0), "seq_lens batch mismatch");
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt32, "block_tables must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(q.scalar_type() == k_cache.scalar_type(), "q and k_cache dtype mismatch");
  TORCH_CHECK(q.scalar_type() == v_cache.scalar_type(), "q and v_cache dtype mismatch");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
  TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
  TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
  TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");
}

}  // namespace

torch::Tensor paged_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    double scale) {
  validate_inputs(q, k_cache, v_cache, block_tables, seq_lens);

  const int batch = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int head_dim = static_cast<int>(q.size(2));
  const int block_size = static_cast<int>(k_cache.size(2));
  const int max_blocks_per_sequence = static_cast<int>(block_tables.size(1));

  auto out = torch::empty_like(q);
  const dim3 grid(batch, heads);
  const int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "paged_attention_forward_cuda",
      [&] {
        paged_attention_kernel<scalar_t><<<grid, threads, shared_bytes, stream>>>(
            q.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            block_tables.data_ptr<int32_t>(),
            seq_lens.data_ptr<int32_t>(),
            out.data_ptr<scalar_t>(),
            batch,
            heads,
            head_dim,
            block_size,
            max_blocks_per_sequence,
            static_cast<float>(scale));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

