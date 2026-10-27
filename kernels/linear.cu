#include <cstdio>
#include <cstdlib>
#include "linear.cuh"

void linear(cublasHandle_t handle, const __half* x, const __half* w, __half* y,
            int T, int in_features, int out_features, cudaStream_t stream) {
  cublasSetStream(handle, stream);
  const float alpha = 1.0f, beta = 0.0f;
  // Row-major buffers viewed as column-major: W is [in x out], x is [in x T], y is [out x T].
  // y_cm (out x T) = op(W_cm)^T (out x in) * x_cm (in x T).
  cublasStatus_t st = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N,
      out_features, T, in_features,
      &alpha,
      w, CUDA_R_16F, in_features,
      x, CUDA_R_16F, in_features,
      &beta,
      y, CUDA_R_16F, out_features,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasGemmEx failed: %d\n", (int)st);
    exit(1);
  }
}   