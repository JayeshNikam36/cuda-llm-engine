#include "kv_scatter.cuh"

// One block per token row. Threads copy the row 16 bytes at a time.
__global__ void kv_scatter_kernel(const uint4* __restrict__ src, uint4* __restrict__ pool,
                                  const int* __restrict__ table, int start_pos, int page_size,
                                  int vec_per_row) {
  const int pos = start_pos + blockIdx.x;
  const size_t row = (size_t)table[pos / page_size] * page_size + (pos % page_size);
  const uint4* s = src + (size_t)blockIdx.x * vec_per_row;
  uint4* d = pool + row * vec_per_row;
  for (int i = threadIdx.x; i < vec_per_row; i += blockDim.x) d[i] = s[i];
}

__global__ void kv_scatter_batch_kernel(const uint4* __restrict__ src, uint4* __restrict__ pool,
                                        const int* __restrict__ tables, const int* __restrict__ positions,
                                        int max_pages, int page_size, int vec_per_row) {
  const int b = blockIdx.x;
  const int pos = positions[b];
  const int* table = tables + (size_t)b * max_pages;
  const size_t row = (size_t)table[pos / page_size] * page_size + (pos % page_size);
  const uint4* s = src + (size_t)b * vec_per_row;
  uint4* d = pool + row * vec_per_row;
  for (int i = threadIdx.x; i < vec_per_row; i += blockDim.x) d[i] = s[i];
}

void kv_scatter(const __half* src, __half* pool, const int* block_table,
                int start_pos, int T, int page_size, int kvdim, cudaStream_t stream) {
  const int vec_per_row = kvdim / 8;
  kv_scatter_kernel<<<T, 64, 0, stream>>>(reinterpret_cast<const uint4*>(src),
                                          reinterpret_cast<uint4*>(pool),
                                          block_table, start_pos, page_size, vec_per_row);
}

void kv_scatter_batch(const __half* src, __half* pool, const int* block_tables, const int* positions,
                      int n, int max_pages, int page_size, int kvdim, cudaStream_t stream) {
  const int vec_per_row = kvdim / 8;
  kv_scatter_batch_kernel<<<n, 64, 0, stream>>>(reinterpret_cast<const uint4*>(src),
                                                reinterpret_cast<uint4*>(pool),
                                                block_tables, positions, max_pages, page_size, vec_per_row);
}
