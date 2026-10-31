#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include "util.h"
#include "attention_decode_paged.cuh"
#include "attention_decode_fast.cuh"

int main() {
  const int NH = 32, NKV = 8, HD = 64, P = 16, NSEQ = getenv("NSEQ") ? atoi(getenv("NSEQ")) : 16;
  const int kvdim = NKV * HD;
  const char* pk = getenv("PEAK_GBPS");
  const double peak = pk ? atof(pk) : 175.0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  auto rand_half = [&](size_t n) {
    std::vector<__half> v(n);
    for (auto& x : v) x = __float2half(U(rng));
    return v;
  };

  const size_t flush_bytes = 128ull << 20;     // larger than the 24 MB L2, so it evicts the KV data
  void* dflush;
  CUDA_CHECK(cudaMalloc(&dflush, flush_bytes));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  char buf[512];
  std::string md = "# Decode attention: naive paged kernel vs fast split-KV kernel (v1)\n\n";
  snprintf(buf, sizeof(buf),
           "GPU: %s | peak used for %%: %.0f GB/s | %d sequences per measurement | L2 flushed before every timed run\n"
           "Naive: one launch per sequence. Fast: one launch for all %d sequences.\n\n", prop.name, peak, NSEQ, NSEQ);
  md += buf;
  md += "| context S | kernel | splits | us per sequence | useful GB/s | % of peak | speedup vs naive |\n|---:|---|---:|---:|---:|---:|---:|\n";

  const int Ss[] = {256, 1024, 2048};
  const int splits_list[] = {1, 2, 4, 8, 16, 32};
  const int MAXSPLIT = 32;

  for (int S : Ss) {
    const int pps = (S + P - 1) / P;
    const int total_pages = NSEQ * pps;
    const size_t pool_elems = (size_t)total_pages * P * kvdim;
    auto kp = rand_half(pool_elems), vp = rand_half(pool_elems);
    auto q = rand_half((size_t)NSEQ * NH * HD);
    std::vector<int> perm(total_pages);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> lens(NSEQ, S);

    __half *dk, *dv, *dq, *dout; int *dtab, *dlens; float* dws;
    CUDA_CHECK(cudaMalloc(&dk, pool_elems * 2));
    CUDA_CHECK(cudaMalloc(&dv, pool_elems * 2));
    CUDA_CHECK(cudaMalloc(&dq, q.size() * 2));
    CUDA_CHECK(cudaMalloc(&dout, q.size() * 2));
    CUDA_CHECK(cudaMalloc(&dtab, total_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dlens, NSEQ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dws, attention_decode_fast_workspace_floats(NSEQ, NH, MAXSPLIT) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dk, kp.data(), pool_elems * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, vp.data(), pool_elems * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dtab, perm.data(), total_pages * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dlens, lens.data(), NSEQ * sizeof(int), cudaMemcpyHostToDevice));

    // Median time in ms of fn(), with the cache flushed before each timed run.
    auto measure = [&](auto&& fn) {
      for (int w = 0; w < 3; w++) fn();
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> ms;
      for (int t = 0; t < 15; t++) {
        CUDA_CHECK(cudaMemset(dflush, t, flush_bytes));
        CUDA_CHECK(cudaEventRecord(e0));
        fn();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float x; CUDA_CHECK(cudaEventElapsedTime(&x, e0, e1));
        ms.push_back(x);
      }
      CUDA_CHECK(cudaGetLastError());
      std::sort(ms.begin(), ms.end());
      return (double)ms[ms.size() / 2];
    };

    const double useful = (double)S * kvdim * 2 * 2;   // unique K and V bytes per sequence
    auto row = [&](const char* name, int splits, double us, double naive_us) {
      double gbps = useful / (us * 1e-6) / 1e9;
      char sp[16];
      if (splits) snprintf(sp, sizeof(sp), "%d", splits); else snprintf(sp, sizeof(sp), "-");
      snprintf(buf, sizeof(buf), "| %d | %s | %s | %.1f | %.1f | %.1f%% | %.1fx |\n",
               S, name, sp, us, gbps, 100.0 * gbps / peak, naive_us / us);
      md += buf;
    };

    double naive_ms = measure([&] {
      for (int s = 0; s < NSEQ; s++)
        attention_decode_paged(dq + (size_t)s * NH * HD, dk, dv, dtab + (size_t)s * pps,
                               dout + (size_t)s * NH * HD, S, P, NH, NKV, HD);
    });
    double naive_us = naive_ms * 1000.0 / NSEQ;
    row("naive", 0, naive_us, naive_us);

    for (int splits : splits_list) {
      double ms = measure([&] {
        attention_decode_fast(dq, dk, dv, dtab, dlens, dout, dws, NSEQ, pps, P, NH, NKV, HD, splits);
      });
      row("fast v1", splits, ms * 1000.0 / NSEQ, naive_us);
    }
    snprintf(buf, sizeof(buf), "| %d | floor at peak | - | %.1f | %.1f | 100.0%% | - |\n", S, useful / (peak * 1e9) * 1e6, peak);
    md += buf;

    cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dout); cudaFree(dtab); cudaFree(dlens); cudaFree(dws);
  }
  md += "\nuseful bytes = unique K and V bytes for one sequence in one layer (S x 512 x 2 x 2). 'floor at peak' = time if those bytes streamed at the peak figure.\n";

  printf("%s", md.c_str());
  FILE* f = fopen("/workspace/docs/attention_decode_compare.md", "w");
  if (f) { fputs(md.c_str(), f); fclose(f); }
  return 0;
}
