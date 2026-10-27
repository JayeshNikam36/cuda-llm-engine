#pragma once
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Rotation speed for each of the head_dim/2 pairs, with Llama-3 frequency scaling (host side).
std::vector<float> compute_rope_inv_freq(int head_dim, double theta, double factor,
                                         double low_freq_factor, double high_freq_factor,
                                         double original_max_pos);

// Rotates q ([n_tokens, n_heads, head_dim]) and k ([n_tokens, n_kv_heads, head_dim]) in place.
// Element i is paired with element i + head_dim/2. inv_freq is a device pointer.
void rope_apply(__half* q, __half* k, const int* positions, const float* inv_freq,
                int n_tokens, int n_heads, int n_kv_heads, int head_dim,
                cudaStream_t stream = 0);
