#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Attention for ONE new token against a KV cache holding S positions (the new token's K/V included).
// q: [n_heads, head_dim]   k_cache, v_cache: [S, n_kv_heads, head_dim]   out: [n_heads, head_dim]
void attention_decode_naive(const __half* q, const __half* k_cache, const __half* v_cache, __half* out,
                            int S, int n_heads, int n_kv_heads, int head_dim,
                            cudaStream_t stream = 0);
