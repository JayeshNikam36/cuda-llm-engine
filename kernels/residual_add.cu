#include "residual_add.cuh"

// One thread per element: out[i] = a[i] + b[i].
// Safe in place (out == a), because each thread only reads and writes its own element.
__global__ void residual_add_kernel(const __half* __restrict__ a, const __half* __restrict__ b,
                                    __half* out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;                                   // last block may have spare threads

  float sum = __half2float(a[i]) + __half2float(b[i]);  // add in 32-bit float
  out[i] = __float2half_rn(sum);                        // round once to fp16
}

void residual_add(const __half* a, const __half* b, __half* out, size_t n, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);  // round up so every element is covered
  residual_add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
}