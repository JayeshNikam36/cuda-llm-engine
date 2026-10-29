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

int main() {
  const int NH = 32, NKV = 8, HD = 64, P = 16, NSEQ = 16;
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

  char buf[512];
  std::string md = "# Baseline: naive paged decode attention (one layer, one sequence per call)\n\n";
  snprintf(buf, sizeof(buf), "GPU: %s | peak used for %%: %.0f GB/s | %d sequences per measurement, launched back to back\n\n",
           prop.name, peak, NSEQ);
  md += buf;
  md += "| context S | us per call | ideal us at peak | useful MB per call | useful GB/s | % of peak |\n|---:|---:|---:|---:|---:|---:|\n";

  const int Ss[] = {256, 1024, 2048};
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  for (int S : Ss) {
    const int pps = (S + P - 1) / P;
    const int total_pages = NSEQ * pps;
    const size_t pool_elems = (size_t)total_pages * P * kvdim;
    auto kp = rand_half(pool_elems), vp = rand_half(pool_elems);
    auto q = rand_half((size_t)NSEQ * NH * HD);

    std::vector<int> perm(total_pages);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);              // sequences get scattered pages

    __half *dk, *dv, *dq, *dout; int* dtab;
    CUDA_CHECK(cudaMalloc(&dk, pool_elems * 2));
    CUDA_CHECK(cudaMalloc(&dv, pool_elems * 2));
    CUDA_CHECK(cudaMalloc(&dq, q.size() * 2));
    CUDA_CHECK(cudaMalloc(&dout, q.size() * 2));
    CUDA_CHECK(cudaMalloc(&dtab, total_pages * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dk, kp.data(), pool_elems * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, vp.data(), pool_elems * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dtab, perm.data(), total_pages * sizeof(int), cudaMemcpyHostToDevice));

    auto run_all = [&] {
      for (int s = 0; s < NSEQ; s++)
        attention_decode_paged(dq + (size_t)s * NH * HD, dk, dv, dtab + s * pps,
                               dout + (size_t)s * NH * HD, S, P, NH, NKV, HD);
    };
    for (int w = 0; w < 3; w++) run_all();
    CUDA_CHECK(cudaDeviceSynchronize());

    const int TRIALS = 15;
    std::vector<float> ms;
    for (int t = 0; t < TRIALS; t++) {
      CUDA_CHECK(cudaEventRecord(e0));
      run_all();
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float x; CUDA_CHECK(cudaEventElapsedTime(&x, e0, e1));
      ms.push_back(x);
    }
    CUDA_CHECK(cudaGetLastError());
    std::sort(ms.begin(), ms.end());
    const double us = ms[TRIALS / 2] * 1000.0 / NSEQ;
    const double useful = (double)S * kvdim * 2 /*K and V*/ * 2 /*bytes*/;
    const double gbps = useful / (us * 1e-6) / 1e9;
    snprintf(buf, sizeof(buf), "| %d | %.1f | %.1f | %.2f | %.1f | %.1f%% |\n",
             S, us, useful / (peak * 1e9) * 1e6, useful / 1e6, gbps, 100.0 * gbps / peak);
    md += buf;

    cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dout); cudaFree(dtab);
  }
  md += "\nuseful bytes = unique K and V bytes for one sequence in one layer (S x 512 x 2 x 2). The kernel may request more than that, because 4 query heads share each KV head.\n";

  printf("%s", md.c_str());
  FILE* f = fopen("/workspace/docs/attention_decode_baseline.md", "w");
  if (f) { fputs(md.c_str(), f); fclose(f); }
  return 0;
}
