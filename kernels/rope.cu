#include <cmath>
#include "rope.cuh"

std::vector<float> compute_rope_inv_freq(int head_dim, double theta, double factor,
                                         double low_ff, double high_ff, double orig_ctx) {
  // Same steps and precision as Hugging Face (float32 throughout).
  const float two_pi = (float)(2.0 * 3.14159265358979323846);
  const float th = (float)theta, fac = (float)factor;
  const float lo = (float)low_ff, hi = (float)high_ff, oc = (float)orig_ctx;
  const float low_wavelen = (float)(orig_ctx / low_ff);
  const float high_wavelen = (float)(orig_ctx / high_ff);
  std::vector<float> out(head_dim / 2);
  for (int i = 0; i < head_dim / 2; i++) {
    float f = 1.0f / powf(th, (float)(2 * i) / (float)head_dim);
    float wavelen = two_pi / f;
    float base = (wavelen > low_wavelen) ? f / fac : f;
    float smooth = (oc / wavelen - lo) / (hi - lo);
    float smoothed = (1.0f - smooth) * base / fac + smooth * base;
    bool medium = !(wavelen < high_wavelen) && !(wavelen > low_wavelen);
    out[i] = medium ? smoothed : base;
  }
  return out;
}
__device__ __forceinline__ float r16(float x) { return __half2float(__float2half_rn(x)); }

// One block per token. Threads loop over every (head, pair) of both q and k.
__global__ void rope_kernel(__half* q, __half* k, const int* __restrict__ pos,
                            const float* __restrict__ inv_freq,
                            int n_heads, int n_kv, int hd) {
  const int t = blockIdx.x, half = hd / 2;
  const float p = (float)pos[t];
  const int total = (n_heads + n_kv) * half;
  for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
    const int head = idx / half, i = idx % half;
    __half* v = head < n_heads ? q + ((size_t)t * n_heads + head) * hd
                               : k + ((size_t)t * n_kv + (head - n_heads)) * hd;
    float s, c;
    sincosf(p * inv_freq[i], &s, &c);
    c = r16(c); s = r16(s);                       // Hugging Face stores cos and sin as fp16
    const float x1 = __half2float(v[i]), x2 = __half2float(v[i + half]);
    v[i]        = __float2half_rn(r16(x1 * c) - r16(x2 * s));
    v[i + half] = __float2half_rn(r16(x2 * c) + r16(x1 * s));
  }
}

void rope_apply(__half* q, __half* k, const int* positions, const float* inv_freq,
                int n_tokens, int n_heads, int n_kv_heads, int head_dim, cudaStream_t stream) {
  rope_kernel<<<n_tokens, 256, 0, stream>>>(q, k, positions, inv_freq, n_heads, n_kv_heads, head_dim);
}
