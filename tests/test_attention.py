import math
import unittest

import torch

from flash_decode import (
    cuda_extension_available,
    make_paged_kv_cache,
    paged_attention,
    paged_attention_reference,
)


class PagedAttentionTests(unittest.TestCase):
    def test_reference_matches_dense_attention(self):
        torch.manual_seed(0)
        batch, heads, seq_len, head_dim = 3, 4, 17, 32
        q = torch.randn(batch, heads, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        seq_lens = torch.tensor([17, 9, 1], dtype=torch.int32)

        k_cache, v_cache, block_tables, seq_lens = make_paged_kv_cache(
            k, v, block_size=5, seq_lens=seq_lens
        )
        actual = paged_attention_reference(q, k_cache, v_cache, block_tables, seq_lens)

        expected = torch.empty_like(q)
        scale = 1.0 / math.sqrt(head_dim)
        for b in range(batch):
            length = int(seq_lens[b].item())
            for h in range(heads):
                scores = torch.matmul(k[b, h, :length], q[b, h]) * scale
                probs = torch.softmax(scores, dim=0)
                expected[b, h] = torch.matmul(probs, v[b, h, :length])

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_public_api_falls_back_to_reference(self):
        torch.manual_seed(1)
        q = torch.randn(2, 2, 16)
        k = torch.randn(2, 2, 11, 16)
        v = torch.randn(2, 2, 11, 16)
        k_cache, v_cache, block_tables, seq_lens = make_paged_kv_cache(k, v, block_size=4)

        actual = paged_attention(q, k_cache, v_cache, block_tables, seq_lens)
        expected = paged_attention_reference(q, k_cache, v_cache, block_tables, seq_lens)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_extension_matches_reference(self):
        self.assertTrue(cuda_extension_available(), "CUDA is available but extension is not built")
        torch.manual_seed(2)
        device = torch.device("cuda")
        q = torch.randn(2, 4, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 4, 33, 64, device=device, dtype=torch.float16)
        v = torch.randn(2, 4, 33, 64, device=device, dtype=torch.float16)
        seq_lens = torch.tensor([33, 19], dtype=torch.int32, device=device)
        k_cache, v_cache, block_tables, seq_lens = make_paged_kv_cache(
            k, v, block_size=8, seq_lens=seq_lens
        )

        actual = paged_attention(q, k_cache, v_cache, block_tables, seq_lens)
        expected = paged_attention_reference(q, k_cache, v_cache, block_tables, seq_lens)

        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2))


if __name__ == "__main__":
    unittest.main()

