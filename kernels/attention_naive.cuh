#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Causal grouped-query attention over a whole sequence (no KV cache yet).
// q: [T, n_heads, head_dim]   k, v: [T, n_kv_heads, head_dim]   out: [T, n_heads, head_dim]
// Query head h attends using kv head h / (n_heads / n_kv_heads). Scale is 1/sqrt(head_dim).
void attention_naive(const __half* q, const __half* k, const __half* v, __half* out,
                     int T, int n_heads, int n_kv_heads, int head_dim,
                     cudaStream_t stream = 0);
