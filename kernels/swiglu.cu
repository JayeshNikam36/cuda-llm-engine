#include "swiglu.cuh"

// One thread per element: out[i] = silu(gate[i]) * up[i]
__global__ void swiglu_kernel(const __half* gate, const __half* up, __half* out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;                       // the last block may have spare threads

  float g = __half2float(gate[i]);          // do the math in 32-bit floats
  float u = __half2float(up[i]);

  float s = g / (1.0f + expf(-g));          // silu(g)
  s = __half2float(__float2half_rn(s));     // round silu to fp16, as PyTorch does

  out[i] = __float2half_rn(s * u);          // multiply, then round to fp16 again
}

void swiglu(const __half* gate, const __half* up, __half* out, size_t n, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);   // round up so every element is covered
  swiglu_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}