#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// y[T, out] = x[T, in] * W^T, where W is stored [out, in] row-major (PyTorch Linear, no bias).
void linear(cublasHandle_t handle, const __half* x, const __half* w, __half* y,
            int T, int in_features, int out_features, cudaStream_t stream = 0);