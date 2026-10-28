#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Same job as attention_decode_naive, but K and V live in paged pools.
// q: [n_heads, head_dim]                      out: [n_heads, head_dim]
// k_pool, v_pool: [num_pages, page_size, n_kv_heads, head_dim]
// block_table: page number for each block of page_size positions (device pointer), covering S positions.
// Position j is stored on page block_table[j / page_size], slot j % page_size.
void attention_decode_paged(const __half* q, const __half* k_pool, const __half* v_pool,
                            const int* block_table, __half* out,
                            int S, int page_size, int n_heads, int n_kv_heads, int head_dim,
                            cudaStream_t stream = 0);
