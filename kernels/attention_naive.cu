#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "attention_naive.cuh"

// One block per (token t, query head h). Slow on purpose: it is the simple, correct baseline
// that the fast Phase 3 kernel will be tested against.
__global__ void attention_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                                 const __half* __restrict__ v, __half* __restrict__ out,
                                 int T, int n_heads, int n_kv_heads, int head_dim) {
  extern __shared__ float scores[];            // T floats, sized at launch
  __shared__ float s_max, s_sum;

  const int t = blockIdx.x;                    // which word
  const int h = blockIdx.y;                    // which query head
  const int g = h / (n_heads / n_kv_heads);    // which kv head this query head shares
  const float scale = rsqrtf((float)head_dim); // 1 / sqrt(head_dim)

  const __half* qv = q + ((size_t)t * n_heads + h) * head_dim;

  // Step 1: score of this query against every key j <= t (j > t is the future: not allowed).
  for (int j = threadIdx.x; j <= t; j += blockDim.x) {
    const __half* kv = k + ((size_t)j * n_kv_heads + g) * head_dim;
    float dot = 0.f;
    for (int d = 0; d < head_dim; d++) dot += __half2float(qv[d]) * __half2float(kv[d]);
    scores[j] = dot * scale;
  }
  __syncthreads();

  // Step 2: softmax. Subtracting the max first prevents overflow and doesn't change the result.
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int j = 0; j <= t; j++) m = fmaxf(m, scores[j]);
    s_max = m;
  }
  __syncthreads();
  for (int j = threadIdx.x; j <= t; j += blockDim.x) scores[j] = expf(scores[j] - s_max);
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int j = 0; j <= t; j++) s += scores[j];
    s_sum = s;
  }
  __syncthreads();
  for (int j = threadIdx.x; j <= t; j += blockDim.x) scores[j] /= s_sum;
  __syncthreads();

  // Step 3: blend the value vectors. Each thread produces one output number at a time.
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j <= t; j++)
      acc += scores[j] * __half2float(v[((size_t)j * n_kv_heads + g) * head_dim + d]);
    out[((size_t)t * n_heads + h) * head_dim + d] = __float2half_rn(acc);
  }
}

void attention_naive(const __half* q, const __half* k, const __half* v, __half* out,
                     int T, int n_heads, int n_kv_heads, int head_dim, cudaStream_t stream) {
  size_t shmem = (size_t)T * sizeof(float);
  if (shmem > 48 * 1024) {
    fprintf(stderr, "attention_naive: T=%d needs %zu bytes of shared memory (limit 48 KB)\n", T, shmem);
    exit(1);
  }
  dim3 grid(T, n_heads);
  attention_kernel<<<grid, 128, shmem, stream>>>(q, k, v, out, T, n_heads, n_kv_heads, head_dim);
}