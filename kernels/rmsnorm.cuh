#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// out[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w, for each row r. `hidden` must be a multiple of 8.
void rmsnorm(const __half* x, const __half* w, __half* out,
             int n_rows, int hidden, float eps, cudaStream_t stream = 0);
