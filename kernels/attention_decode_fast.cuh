#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Floats of scratch needed for a given batch size and split count.
size_t attention_decode_fast_workspace_floats(int n_seq, int n_heads, int num_splits);

// Batched decode attention over a paged KV cache, one new token per sequence.
// q:            [n_seq, n_heads, head_dim]
// k_pool/v_pool: [num_pages, page_size, n_kv_heads, head_dim]
// block_tables: [n_seq, max_pages] page numbers;  seq_lens: [n_seq] cached positions per sequence
// out:          [n_seq, n_heads, head_dim]
// Requires page_size == 16, head_dim == 64, n_heads == 4 * n_kv_heads (Llama-3.2-1B).
void attention_decode_fast(const __half* q, const __half* k_pool, const __half* v_pool,
                           const int* block_tables, const int* seq_lens, __half* out, float* workspace,
                           int n_seq, int max_pages, int page_size,
                           int n_heads, int n_kv_heads, int head_dim,
                           int num_splits, cudaStream_t stream = 0);
