#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// out[t] = table[ids[t]] for each token t. `hidden` must be a multiple of 8.
void embedding_lookup(const __half* table, const int* ids, __half* out,
                      int n_tokens, int hidden, cudaStream_t stream = 0);
