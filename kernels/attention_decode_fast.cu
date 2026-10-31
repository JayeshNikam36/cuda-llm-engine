#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "attention_decode_fast.cuh"

constexpr int HD = 64;        // head dimension
constexpr int G = 4;          // query heads per kv head
constexpr int PAGE = 16;      // positions per page
constexpr int THREADS = 128;
constexpr int GROUPS = THREADS / 8;   // 16 position groups of 8 threads

__device__ __forceinline__ void unpack8(const uint4& raw, float* f) {
  const __half2* h = reinterpret_cast<const __half2*>(&raw);
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 t = __half22float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ size_t part_index(int seq, int n_heads, int head, int num_splits, int split) {
  return ((size_t)seq * n_heads + head) * num_splits + split;
}

// Grid: (num_splits, n_kv_heads, n_seq). Each block handles one kv head (all G query heads that share it)
// for one chunk of one sequence's positions, and writes an un-normalized partial result.
__global__ void __launch_bounds__(THREADS)
decode_split_kernel(const __half* __restrict__ q, const __half* __restrict__ k_pool,
                    const __half* __restrict__ v_pool, const int* __restrict__ block_tables,
                    const int* __restrict__ seq_lens, float* __restrict__ part_o,
                    float* __restrict__ part_m, float* __restrict__ part_l,
                    int max_pages, int n_heads, int n_kv_heads, int num_splits) {
  const int split = blockIdx.x, g = blockIdx.y, seq = blockIdx.z;
  const int S = seq_lens[seq];
  const int total_pages = (S + PAGE - 1) / PAGE;
  const int pages_per_split = (total_pages + num_splits - 1) / num_splits;
  const int start = split * pages_per_split * PAGE;
  const int end = min(S, start + pages_per_split * PAGE);
  const int tid = threadIdx.x;

  if (start >= end) {   // this split has no positions (short sequence): write neutral values for the merge
    for (int hh = 0; hh < G; hh++) {
      const size_t ob = part_index(seq, n_heads, g * G + hh, num_splits, split);
      if (tid < HD) part_o[ob * HD + tid] = 0.f;
      if (tid == 0) { part_m[ob] = -INFINITY; part_l[ob] = 0.f; }
    }
    return;
  }

  const int grp = tid >> 3;       // which of the 16 position groups
  const int lane8 = tid & 7;      // which 8-value slice of the head this thread owns
  const unsigned gmask = 0xffu << (tid & 24);   // the 8 lanes of this group, for the shuffles
  const int* table = block_tables + (size_t)seq * max_pages;

  // This thread's slice of the 4 query heads, pre-scaled by 1/sqrt(64) = 0.125.
  float qf[G][8];
  #pragma unroll
  for (int hh = 0; hh < G; hh++) {
    const uint4 raw = *reinterpret_cast<const uint4*>(q + ((size_t)seq * n_heads + g * G + hh) * HD + lane8 * 8);
    unpack8(raw, qf[hh]);
    #pragma unroll
    for (int i = 0; i < 8; i++) qf[hh][i] *= 0.125f;
  }

  float m[G], l[G], acc[G][8];
  #pragma unroll
  for (int hh = 0; hh < G; hh++) {
    m[hh] = -INFINITY;
    l[hh] = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[hh][i] = 0.f;
  }

  // Each group walks positions grp, grp + 16, grp + 32, ... of this chunk.
  for (int j = start + grp; j < end; j += GROUPS) {
    const size_t row = (size_t)table[j >> 4] * PAGE + (j & (PAGE - 1));
    const size_t off = (row * n_kv_heads + g) * HD + lane8 * 8;
    const uint4 kraw = *reinterpret_cast<const uint4*>(k_pool + off);   // 16-byte loads
    const uint4 vraw = *reinterpret_cast<const uint4*>(v_pool + off);
    float kf[8], vf[8];
    unpack8(kraw, kf);
    unpack8(vraw, vf);

    #pragma unroll
    for (int hh = 0; hh < G; hh++) {
      float s = 0.f;
      #pragma unroll
      for (int i = 0; i < 8; i++) s += qf[hh][i] * kf[i];
      s += __shfl_xor_sync(gmask, s, 1);      // add up the 8 threads' partial dot products
      s += __shfl_xor_sync(gmask, s, 2);
      s += __shfl_xor_sync(gmask, s, 4);

      // Online softmax update.
      const float nm = fmaxf(m[hh], s);
      const float scale_old = expf(m[hh] - nm);   // first time: exp(-inf) = 0
      const float p = expf(s - nm);
      l[hh] = l[hh] * scale_old + p;
      #pragma unroll
      for (int i = 0; i < 8; i++) acc[hh][i] = acc[hh][i] * scale_old + p * vf[i];
      m[hh] = nm;
    }
  }

  // Combine the 16 groups of this block through shared memory.
  __shared__ float s_acc[GROUPS][G][HD];
  __shared__ float s_m[GROUPS][G], s_l[GROUPS][G];
  #pragma unroll
  for (int hh = 0; hh < G; hh++)
    #pragma unroll
    for (int i = 0; i < 8; i++) s_acc[grp][hh][lane8 * 8 + i] = acc[hh][i];
  if (lane8 == 0) {
    #pragma unroll
    for (int hh = 0; hh < G; hh++) { s_m[grp][hh] = m[hh]; s_l[grp][hh] = l[hh]; }
  }
  __syncthreads();

  for (int idx = tid; idx < G * HD; idx += THREADS) {
    const int hh = idx / HD, d = idx % HD;
    float M = -INFINITY;
    for (int gr = 0; gr < GROUPS; gr++) M = fmaxf(M, s_m[gr][hh]);
    float L = 0.f, A = 0.f;
    for (int gr = 0; gr < GROUPS; gr++) {
      const float w = expf(s_m[gr][hh] - M);   // groups that saw nothing have m = -inf, so w = 0
      L += s_l[gr][hh] * w;
      A += s_acc[gr][hh][d] * w;
    }
    const size_t ob = part_index(seq, n_heads, g * G + hh, num_splits, split);
    part_o[ob * HD + d] = A;
    if (d == 0) { part_m[ob] = M; part_l[ob] = L; }
  }
}

// Grid: (n_heads, n_seq), 64 threads (one per output dimension). Merges the splits of one (sequence, head).
__global__ void decode_merge_kernel(const float* __restrict__ part_o, const float* __restrict__ part_m,
                                    const float* __restrict__ part_l, __half* __restrict__ out,
                                    int n_heads, int num_splits) {
  const int h = blockIdx.x, seq = blockIdx.y, d = threadIdx.x;
  const size_t base = ((size_t)seq * n_heads + h) * num_splits;
  float M = -INFINITY;
  for (int s = 0; s < num_splits; s++) M = fmaxf(M, part_m[base + s]);
  float L = 0.f, A = 0.f;
  for (int s = 0; s < num_splits; s++) {
    const float w = expf(part_m[base + s] - M);   // empty splits have m = -inf, so w = 0
    L += part_l[base + s] * w;
    A += part_o[(base + s) * HD + d] * w;
  }
  out[((size_t)seq * n_heads + h) * HD + d] = __float2half_rn(A / L);
}

size_t attention_decode_fast_workspace_floats(int n_seq, int n_heads, int num_splits) {
  return (size_t)n_seq * n_heads * num_splits * (HD + 2);
}

void attention_decode_fast(const __half* q, const __half* k_pool, const __half* v_pool,
                           const int* block_tables, const int* seq_lens, __half* out, float* workspace,
                           int n_seq, int max_pages, int page_size,
                           int n_heads, int n_kv_heads, int head_dim,
                           int num_splits, cudaStream_t stream) {
  if (page_size != PAGE || head_dim != HD || n_heads != G * n_kv_heads) {
    fprintf(stderr, "attention_decode_fast: needs page_size 16, head_dim 64, 4 query heads per kv head\n");
    exit(1);
  }
  float* part_o = workspace;
  float* part_m = part_o + (size_t)n_seq * n_heads * num_splits * HD;
  float* part_l = part_m + (size_t)n_seq * n_heads * num_splits;
  dim3 grid(num_splits, n_kv_heads, n_seq);
  decode_split_kernel<<<grid, THREADS, 0, stream>>>(q, k_pool, v_pool, block_tables, seq_lens,
                                                    part_o, part_m, part_l,
                                                    max_pages, n_heads, n_kv_heads, num_splits);
  decode_merge_kernel<<<dim3(n_heads, n_seq), HD, 0, stream>>>(part_o, part_m, part_l, out, n_heads, num_splits);
}
