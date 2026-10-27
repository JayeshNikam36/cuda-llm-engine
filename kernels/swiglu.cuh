#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// out[i] = silu(gate[i]) * up[i] for i in [0, n).
void swiglu(const __half* gate, const __half* up, __half* out, size_t n, cudaStream_t stream = 0);
