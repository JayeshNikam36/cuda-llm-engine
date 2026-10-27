#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// out[i] = a[i] + b[i] for i in [0, n). `out` may be the same pointer as `a` (in-place).
void residual_add(const __half* a, const __half* b, __half* out, size_t n, cudaStream_t stream = 0);
