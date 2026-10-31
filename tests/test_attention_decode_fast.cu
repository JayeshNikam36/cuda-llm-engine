#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <numeric>
#include <random>
#include <vector>
#include "util.h"
#include "attention_decode_paged.cuh"
#include "attention_decode_fast.cuh"

static float h2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }
static uint16_t f2h(float f) { __half h = __float2half(f); uint16_t u; memcpy(&u, &h, 2); return u; }

int main() {
  const int NH = 32, NKV = 8, HD = 64, P = 16, kvdim = NKV * HD;
  const std::vector<int> lens = {1, 15, 16, 17, 33, 100, 402, 2048};   // includes page-boundary cases
  const int n_seq = (int)lens.size();
  int max_pages = 0, total_pages = 0;
  for (int S : lens) { int np = (S + P - 1) / P; max_pages = std::max(max_pages, np); total_pages += np; }

  std::mt19937 rng(2024);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::vector<uint16_t> q((size_t)n_seq * NH * HD);
  for (auto& x : q) x = f2h(U(rng));

  // Pages are assigned in shuffled order; every unused slot is NaN so a wrong read shows up.
  std::vector<int> perm(total_pages);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<int> tables((size_t)n_seq * max_pages, 0);
  std::vector<uint16_t> kp((size_t)total_pages * P * kvdim, 0x7F7F), vp((size_t)total_pages * P * kvdim, 0x7F7F);
  int next = 0;
  for (int s = 0; s < n_seq; s++) {
    int np = (lens[s] + P - 1) / P;
    for (int i = 0; i < np; i++) tables[(size_t)s * max_pages + i] = perm[next++];
    for (int j = 0; j < lens[s]; j++) {
      size_t dst = ((size_t)tables[(size_t)s * max_pages + j / P] * P + j % P) * kvdim;
      for (int d = 0; d < kvdim; d++) { kp[dst + d] = f2h(U(rng)); vp[dst + d] = f2h(U(rng)); }
    }
  }

  __half *dq, *dk, *dv, *dout, *dref; int *dtab, *dlens;
  CUDA_CHECK(cudaMalloc(&dq, q.size() * 2));
  CUDA_CHECK(cudaMalloc(&dk, kp.size() * 2));
  CUDA_CHECK(cudaMalloc(&dv, vp.size() * 2));
  CUDA_CHECK(cudaMalloc(&dout, q.size() * 2));
  CUDA_CHECK(cudaMalloc(&dref, q.size() * 2));
  CUDA_CHECK(cudaMalloc(&dtab, tables.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dlens, n_seq * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, kp.data(), kp.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, vp.data(), vp.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dtab, tables.data(), tables.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dlens, lens.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice));

  // Reference: the already-tested naive paged kernel, one sequence at a time.
  for (int s = 0; s < n_seq; s++)
    attention_decode_paged(dq + (size_t)s * NH * HD, dk, dv, dtab + (size_t)s * max_pages,
                           dref + (size_t)s * NH * HD, lens[s], P, NH, NKV, HD);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> ref(q.size());
  CUDA_CHECK(cudaMemcpy(ref.data(), dref, ref.size() * 2, cudaMemcpyDeviceToHost));

  int failures = 0;
  for (int splits : {1, 2, 3, 8, 16, 32}) {
    float* dws;
    CUDA_CHECK(cudaMalloc(&dws, attention_decode_fast_workspace_floats(n_seq, NH, splits) * sizeof(float)));
    CUDA_CHECK(cudaMemset(dout, 0x7F, q.size() * 2));      // unwritten outputs would be NaN
    attention_decode_fast(dq, dk, dv, dtab, dlens, dout, dws, n_seq, max_pages, P, NH, NKV, HD, splits);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> got(q.size());
    CUDA_CHECK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));

    double worst = 0; size_t exact = 0, total = 0;
    std::vector<int> bad_seqs;
    for (int s = 0; s < n_seq; s++) {
      size_t bad = 0;
      for (size_t i = (size_t)s * NH * HD; i < (size_t)(s + 1) * NH * HD; i++) {
        double g = h2f(got[i]), e = h2f(ref[i]), d = fabs(g - e);
        if (got[i] == ref[i]) exact++;
        total++;
        if (d > worst || std::isnan(d)) worst = std::isnan(d) ? NAN : d;
        if (!(d <= 1e-3 + 4e-3 * fabs(e))) bad++;
      }
      if (bad) bad_seqs.push_back(lens[s]);
    }
    bool ok = bad_seqs.empty();
    printf("splits %2d: %d sequences  max_abs_diff %.2e  exact %.2f%%  %s", splits, n_seq, worst,
           100.0 * exact / total, ok ? "PASS" : "FAIL");
    if (!ok) { printf("  (failing lengths:"); for (int L : bad_seqs) printf(" %d", L); printf(")"); }
    printf("\n");
    failures += !ok;
    cudaFree(dws);
  }
  return failures ? 1 : 0;
}
