#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "attention_decode_naive.cuh"

// One block per query head. Same math as attention_naive, but the single query attends to
// every cached position j in [0, S) (they are all in the past, so no mask is needed).
__global__ void attention_decode_kernel(const __half* __restrict__ q, const __half* __restrict__ kc,
                                        const __half* __restrict__ vc, __half* __restrict__ out,
                                        int S, int n_heads, int n_kv_heads, int head_dim) {
  extern __shared__ float scores[];            // S floats
  __shared__ float s_max, s_sum;

  const int h = blockIdx.x;
  const int g = h / (n_heads / n_kv_heads);    // grouped-query sharing
  const float scale = rsqrtf((float)head_dim);
  const __half* qv = q + (size_t)h * head_dim;

  for (int j = threadIdx.x; j < S; j += blockDim.x) {
    const __half* kv = kc + ((size_t)j * n_kv_heads + g) * head_dim;
    float dot = 0.f;
    for (int d = 0; d < head_dim; d++) dot += __half2float(qv[d]) * __half2float(kv[d]);
    scores[j] = dot * scale;
  }
  __syncthreads();

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

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < S; j++)
      acc += scores[j] * __half2float(vc[((size_t)j * n_kv_heads + g) * head_dim + d]);
    out[(size_t)h * head_dim + d] = __float2half_rn(acc);
  }
}

void attention_decode_naive(const __half* q, const __half* k_cache, const __half* v_cache, __half* out,
                            int S, int n_heads, int n_kv_heads, int head_dim, cudaStream_t stream) {
  size_t shmem = (size_t)S * sizeof(float);
  if (shmem > 48 * 1024) {
    fprintf(stderr, "attention_decode_naive: S=%d needs %zu bytes of shared memory (limit 48 KB)\n", S, shmem);
    exit(1);
  }
  attention_decode_kernel<<<n_heads, 128, shmem, stream>>>(q, k_cache, v_cache, out, S, n_heads, n_kv_heads, head_dim);
}
