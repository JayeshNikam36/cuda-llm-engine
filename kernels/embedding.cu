#include "embedding.cuh"

// One block per token. Threads copy the row 8 half-values (16 bytes) at a time.
__global__ void embedding_kernel(const uint4* __restrict__ table, const int* __restrict__ ids,
                                 uint4* __restrict__ out, int vec_per_row) {
  const uint4* src = table + (size_t)ids[blockIdx.x] * vec_per_row;
  uint4* dst = out + (size_t)blockIdx.x * vec_per_row;
  for (int i = threadIdx.x; i < vec_per_row; i += blockDim.x) dst[i] = src[i];
}

void embedding_lookup(const __half* table, const int* ids, __half* out,
                      int n_tokens, int hidden, cudaStream_t stream) {
  int vec_per_row = hidden / 8;
  embedding_kernel<<<n_tokens, 256, 0, stream>>>(
      reinterpret_cast<const uint4*>(table), ids, reinterpret_cast<uint4*>(out), vec_per_row);
}
