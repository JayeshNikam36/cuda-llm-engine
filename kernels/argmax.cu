#include <climits>
#include <cmath>
#include "argmax.cuh"

__device__ __forceinline__ void better(float& v, int& i, float ov, int oi) {
  if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
}

// One block (1024 threads) per row: each thread scans a strided slice, then a warp shuffle reduction
// and a second-stage reduction across warps pick the winner.
__global__ void argmax_rows_kernel(const __half* __restrict__ logits, int* __restrict__ out, int vocab) {
  const __half* row = logits + (size_t)blockIdx.x * vocab;
  float best = -INFINITY;
  int best_i = INT_MAX;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float v = __half2float(row[i]);
    if (v > best) { best = v; best_i = i; }      // ascending i, so ties keep the earlier index
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_down_sync(0xffffffff, best, o);
    const int oi = __shfl_down_sync(0xffffffff, best_i, o);
    better(best, best_i, ov, oi);
  }
  __shared__ float sv[32];
  __shared__ int si[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) { sv[wid] = best; si[wid] = best_i; }
  __syncthreads();
  if (wid == 0) {
    const int nw = blockDim.x >> 5;
    best = lane < nw ? sv[lane] : -INFINITY;
    best_i = lane < nw ? si[lane] : INT_MAX;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_down_sync(0xffffffff, best, o);
      const int oi = __shfl_down_sync(0xffffffff, best_i, o);
      better(best, best_i, ov, oi);
    }
    if (lane == 0) out[blockIdx.x] = best_i;
  }
}

void argmax_rows(const __half* logits, int* out, int n_rows, int vocab, cudaStream_t stream) {
  argmax_rows_kernel<<<n_rows, 1024, 0, stream>>>(logits, out, vocab);
}
