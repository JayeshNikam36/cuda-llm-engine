#include <cstring>
#include <cstdint>
#include <cmath>
#include "util.h"
#include "swiglu.cuh"

static float h2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

int main() {
  const char* re = getenv("REF_DIR");
  std::string R = re ? re : "/workspace/tests/reference/bin";
  int failures = 0;
  for (int i = 0; i < 8; i++) {
    char p[256];
    auto load = [&](const char* name) {
      snprintf(p, sizeof(p), "%s/prompt_%02d_%s.f16", R.c_str(), i, name);
      return read_file<uint16_t>(p);
    };
    auto gate = load("gate"), up = load("up"), expected = load("swiglu");
    size_t n = gate.size();
    if (up.size() != n || expected.size() != n) { printf("prompt %d: SIZE MISMATCH\n", i); failures++; continue; }

    __half *dg, *du, *dout;
    CUDA_CHECK(cudaMalloc(&dg, n * 2));
    CUDA_CHECK(cudaMalloc(&du, n * 2));
    CUDA_CHECK(cudaMalloc(&dout, n * 2));
    CUDA_CHECK(cudaMemcpy(dg, gate.data(), n * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(du, up.data(), n * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, n * 2));
    swiglu(dg, du, dout, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got(n);
    CUDA_CHECK(cudaMemcpy(got.data(), dout, n * 2, cudaMemcpyDeviceToHost));
    double max_abs = 0; size_t exact = 0, bad = 0;
    for (size_t k = 0; k < n; k++) {
      double g = h2f(got[k]), e = h2f(expected[k]), d = fabs(g - e);
      if (got[k] == expected[k]) exact++;
      if (d > max_abs) max_abs = d;
      if (!(d <= 1e-3 + 4e-3 * fabs(e))) bad++;
    }
    bool ok = bad == 0;
    printf("prompt %d: %zu values  max_abs_diff %.2e  exact %.2f%%  %s\n",
           i, n, max_abs, 100.0 * exact / n, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
    cudaFree(dg); cudaFree(du); cudaFree(dout);
  }
  return failures ? 1 : 0;
}
