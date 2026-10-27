#include "rmsnorm.cuh"

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// One block per row. Each thread handles 8 half-values (one 16-byte vector) per step.
__global__ void rmsnorm_kernel(const uint4* __restrict__ x, const uint4* __restrict__ w,
                               uint4* __restrict__ out, int vec_per_row, int hidden, float eps) {
  const uint4* xr = x + (size_t)blockIdx.x * vec_per_row;
  uint4* orow = out + (size_t)blockIdx.x * vec_per_row;

  // Pass 1: sum of squares in fp32.
  float ss = 0.f;
  for (int i = threadIdx.x; i < vec_per_row; i += blockDim.x) {
    uint4 v = xr[i];
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 f = __half22float2(h[j]);
      ss += f.x * f.x + f.y * f.y;
    }
  }

  // Combine: within each warp (32 threads), then across warps through shared memory.
  ss = warp_sum(ss);
  __shared__ float warp_part[32];
  __shared__ float s_inv;
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) warp_part[wid] = ss;
  __syncthreads();
  if (wid == 0) {
    int nwarps = blockDim.x >> 5;
    float total = (lane < nwarps) ? warp_part[lane] : 0.f;
    total = warp_sum(total);
    if (lane == 0) s_inv = rsqrtf(total / hidden + eps);
  }
  __syncthreads();
  float inv = s_inv;

  // Pass 2: normalize, round to fp16 (as Hugging Face does), then multiply by the weight.
  for (int i = threadIdx.x; i < vec_per_row; i += blockDim.x) {
    uint4 v = xr[i];
    uint4 wv = w[i];
    uint4 o;
    const __half2* hx = reinterpret_cast<const __half2*>(&v);
    const __half2* hw = reinterpret_cast<const __half2*>(&wv);
    __half2* ho = reinterpret_cast<__half2*>(&o);
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 fx = __half22float2(hx[j]);
      float2 fw = __half22float2(hw[j]);
      float a = __half2float(__float2half_rn(fx.x * inv));
      float b = __half2float(__float2half_rn(fx.y * inv));
      ho[j] = __halves2half2(__float2half_rn(fw.x * a), __float2half_rn(fw.y * b));
    }
    orow[i] = o;
  }
}

void rmsnorm(const __half* x, const __half* w, __half* out,
             int n_rows, int hidden, float eps, cudaStream_t stream) {
  int vec_per_row = hidden / 8;
  rmsnorm_kernel<<<n_rows, 256, 0, stream>>>(
      reinterpret_cast<const uint4*>(x), reinterpret_cast<const uint4*>(w),
      reinterpret_cast<uint4*>(out), vec_per_row, hidden, eps);
}
