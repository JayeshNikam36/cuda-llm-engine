#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "attention_decode_paged.cuh"

// Row of the pool (one row = one token's K or V for all kv heads) where position j is stored:
// find the page through the block table, then the slot inside that page.
__device__ __forceinline__ size_t paged_row(const int* __restrict__ block_table, int j, int page_size) {
  return (size_t)block_table[j / page_size] * page_size + (j % page_size);
}

// One block per query head. Identical math to attention_decode_naive; only the addressing of
// K and V changed: position j goes through the block table instead of being row j.
__global__ void attention_decode_paged_kernel(const __half* __restrict__ q,
                                              const __half* __restrict__ k_pool,
                                              const __half* __restrict__ v_pool,
                                              const int* __restrict__ block_table,
                                              __half* __restrict__ out,
                                              int S, int page_size,
                                              int n_heads, int n_kv_heads, int head_dim) {
  extern __shared__ float scores[];            // S floats
  __shared__ float s_max, s_sum;

  const int h = blockIdx.x;
  const int g = h / (n_heads / n_kv_heads);    // grouped-query sharing
  const float scale = rsqrtf((float)head_dim);
  const __half* qv = q + (size_t)h * head_dim;

  // Scores: dot(q, key_j) for every cached position j.
  for (int j = threadIdx.x; j < S; j += blockDim.x) {
    const __half* kv = k_pool + (paged_row(block_table, j, page_size) * n_kv_heads + g) * head_dim;
    float dot = 0.f;
    for (int d = 0; d < head_dim; d++) dot += __half2float(qv[d]) * __half2float(kv[d]);
    scores[j] = dot * scale;
  }
  __syncthreads();

  // Softmax (subtract the max first).
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int j = 0; j < S; j++) m = fmaxf(m, scores[j]);
    s_max = m;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < S; j += blockDim.x) scores[j] = expf(scores[j] - s_max);
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int j = 0; j < S; j++) s += scores[j];
    s_sum = s;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < S; j += blockDim.x) scores[j] /= s_sum;
  __syncthreads();

  // Blend the value vectors, again through the block table.
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < S; j++)
      acc += scores[j] * __half2float(v_pool[(paged_row(block_table, j, page_size) * n_kv_heads + g) * head_dim + d]);
    out[(size_t)h * head_dim + d] = __float2half_rn(acc);
  }
}

void attention_decode_paged(const __half* q, const __half* k_pool, const __half* v_pool,
                            const int* block_table, __half* out,
                            int S, int page_size, int n_heads, int n_kv_heads, int head_dim,
                            cudaStream_t stream) {
  size_t shmem = (size_t)S * sizeof(float);
  if (shmem > 48 * 1024) {
    fprintf(stderr, "attention_decode_paged: S=%d needs %zu bytes of shared memory (limit 48 KB)\n", S, shmem);
    exit(1);
  }
  attention_decode_paged_kernel<<<n_heads, 128, shmem, stream>>>(
      q, k_pool, v_pool, block_table, out, S, page_size, n_heads, n_kv_heads, head_dim);
}