#include <torch/extension.h>

torch::Tensor paged_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    double scale);

torch::Tensor paged_attention_forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    double scale) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be a CUDA tensor");
  TORCH_CHECK(v_cache.is_cuda(), "v_cache must be a CUDA tensor");
  TORCH_CHECK(block_tables.is_cuda(), "block_tables must be a CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  return paged_attention_forward_cuda(q, k_cache, v_cache, block_tables, seq_lens, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "paged_attention_forward",
      &paged_attention_forward,
      "Paged KV-cache decode attention forward pass (CUDA)");
}

