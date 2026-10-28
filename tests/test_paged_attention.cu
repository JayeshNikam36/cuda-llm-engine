#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include "util.h"
#include "paging.h"
#include "attention_decode_naive.cuh"
#include "attention_decode_paged.cuh"

static float h2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }
static uint16_t f2h(float f) { __half h = __float2half(f); uint16_t u; memcpy(&u, &h, 2); return u; }

int main() {
  const int NH = 32, NKV = 8, HD = 64, P = 16, POOL = 64;
  const int kvdim = NKV * HD;
  int failures = 0;

  // Allocator sanity: every page handed out exactly once, released pages come back.
  {
    PageAllocator a(POOL);
    std::vector<int> got;
    for (int i = 0; i < POOL; i++) got.push_back(a.alloc());
    std::vector<int> sorted = got;
    std::sort(sorted.begin(), sorted.end());
    bool unique_all = true;
    for (int i = 0; i < POOL; i++) if (sorted[i] != i) unique_all = false;
    bool drained = a.num_free() == 0;
    a.release(got[5]);
    bool reuse = a.num_free() == 1 && a.alloc() == got[5];
    bool ok = unique_all && drained && reuse;
    printf("allocator (each page once, released pages reused): %s\n", ok ? "PASS" : "FAIL");
    failures += !ok;
  }

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  const int Ss[] = {1, 15, 16, 17, 31, 32, 33, 100, 402};

  for (int S : Ss) {
    std::vector<uint16_t> q(NH * HD), k((size_t)S * kvdim), v((size_t)S * kvdim);
    for (auto& x : q) x = f2h(U(rng));
    for (auto& x : k) x = f2h(U(rng));
    for (auto& x : v) x = f2h(U(rng));

    // Reference: contiguous cache, already-tested kernel.
    __half *dq, *dk, *dv, *dref, *dout;
    CUDA_CHECK(cudaMalloc(&dq, q.size() * 2));
    CUDA_CHECK(cudaMalloc(&dk, k.size() * 2));
    CUDA_CHECK(cudaMalloc(&dv, v.size() * 2));
    CUDA_CHECK(cudaMalloc(&dref, NH * HD * 2));
    CUDA_CHECK(cudaMalloc(&dout, NH * HD * 2));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, NH * HD * 2));
    attention_decode_naive(dq, dk, dv, dref, S, NH, NKV, HD);

    // Paged: scatter K and V across shuffled pages; every unused slot is NaN so a wrong read shows up.
    const int npages = (S + P - 1) / P;
    std::vector<int> perm(POOL);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> table(perm.begin(), perm.begin() + npages);
    std::vector<uint16_t> kp((size_t)POOL * P * kvdim, 0x7F7F), vp((size_t)POOL * P * kvdim, 0x7F7F);
    for (int j = 0; j < S; j++) {
      size_t dst = ((size_t)table[j / P] * P + (j % P)) * kvdim;
      memcpy(&kp[dst], &k[(size_t)j * kvdim], kvdim * 2);
      memcpy(&vp[dst], &v[(size_t)j * kvdim], kvdim * 2);
    }
    __half *dkp, *dvp; int* dtable;
    CUDA_CHECK(cudaMalloc(&dkp, kp.size() * 2));
    CUDA_CHECK(cudaMalloc(&dvp, vp.size() * 2));
    CUDA_CHECK(cudaMalloc(&dtable, npages * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dkp, kp.data(), kp.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvp, vp.data(), vp.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dtable, table.data(), npages * sizeof(int), cudaMemcpyHostToDevice));

    attention_decode_paged(dq, dkp, dvp, dtable, dout, S, P, NH, NKV, HD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> ref(NH * HD), got(NH * HD);
    CUDA_CHECK(cudaMemcpy(ref.data(), dref, ref.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
    double max_abs = 0; size_t exact = 0, bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
      double g = h2f(got[i]), e = h2f(ref[i]), d = fabs(g - e);
      if (got[i] == ref[i]) exact++;
      if (d > max_abs) max_abs = d;
      if (!(d <= 1e-3 + 4e-3 * fabs(e))) bad++;      // NaN also lands here
    }
    bool ok = bad == 0;
    printf("S = %3d (%2d pages, shuffled): max_abs_diff %.2e  exact %.2f%%  %s\n",
           S, npages, max_abs, 100.0 * exact / got.size(), ok ? "PASS" : "FAIL");
    failures += !ok;

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dref); cudaFree(dout);
    cudaFree(dkp); cudaFree(dvp); cudaFree(dtable);
  }
  return failures ? 1 : 0;
}
