#include <cstring>
#include <cstdint>
#include <cmath>
#include "util.h"
#include "attention_naive.cuh"

static float h2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

int main() {
  const char* re = getenv("REF_DIR");
  std::string R = re ? re : "/workspace/tests/reference/bin";
  const int NH = 32, NKV = 8, HD = 64;
  int failures = 0;
  for (int i = 0; i < 8; i++) {
    char p[256];
    auto load = [&](const char* name) {
      snprintf(p, sizeof(p), "%s/prompt_%02d_%s.f16", R.c_str(), i, name);
      return read_file<uint16_t>(p);
    };
    auto q = load("q_rope"), k = load("k_rope"), v = load("v"), expected = load("attn");
    int T = (int)(q.size() / (NH * HD));
    if (k.size() != (size_t)T * NKV * HD || v.size() != k.size() || expected.size() != q.size()) {
      printf("prompt %d: SIZE MISMATCH\n", i); failures++; continue;
    }
    __half *dq, *dk, *dv, *dout;
    CUDA_CHECK(cudaMalloc(&dq, q.size() * 2));
    CUDA_CHECK(cudaMalloc(&dk, k.size() * 2));
    CUDA_CHECK(cudaMalloc(&dv, v.size() * 2));
    CUDA_CHECK(cudaMalloc(&dout, expected.size() * 2));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, expected.size() * 2));
    attention_naive(dq, dk, dv, dout, T, NH, NKV, HD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got(expected.size());
    CUDA_CHECK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
    double max_abs = 0; size_t exact = 0, bad = 0;
    for (size_t j = 0; j < got.size(); j++) {
      double g = h2f(got[j]), e = h2f(expected[j]), d = fabs(g - e);
      if (got[j] == expected[j]) exact++;
      if (d > max_abs) max_abs = d;
      if (!(d <= 1e-3 + 4e-3 * fabs(e))) bad++;
    }
    bool ok = bad == 0;
    printf("prompt %d: %3d tokens  max_abs_diff %.2e  exact %.2f%%  %s\n",
           i, T, max_abs, 100.0 * exact / got.size(), ok ? "PASS" : "FAIL");
    if (!ok) failures++;
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
  }
  return failures ? 1 : 0;
}
