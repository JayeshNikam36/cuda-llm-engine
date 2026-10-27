#include <cstring>
#include <cstdint>
#include <cmath>
#include "util.h"
#include "rmsnorm.cuh"

static float h2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";
  const float eps = 1e-5f;   // rms_norm_eps from config.json

  TensorInfo ti = find_tensor(W + "/index.tsv", "model.layers.0.input_layernorm.weight");
  int hidden = (int)ti.shape[0];
  auto w = read_file<uint16_t>(W + "/weights.bin", ti.offset, ti.nbytes / 2);
  __half* d_w;
  CUDA_CHECK(cudaMalloc(&d_w, ti.nbytes));
  CUDA_CHECK(cudaMemcpy(d_w, w.data(), ti.nbytes, cudaMemcpyHostToDevice));

  int failures = 0;
  for (int i = 0; i < 8; i++) {
    char p[256];
    snprintf(p, sizeof(p), "%s/prompt_%02d_embed.f16", R.c_str(), i);
    auto x = read_file<uint16_t>(p);
    snprintf(p, sizeof(p), "%s/prompt_%02d_norm0.f16", R.c_str(), i);
    auto expected = read_file<uint16_t>(p);
    int n = (int)(x.size() / hidden);
    if (expected.size() != x.size()) { printf("prompt %d: SIZE MISMATCH\n", i); failures++; continue; }

    __half *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, x.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_out, x.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    rmsnorm(d_x, d_w, d_out, n, hidden, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got(x.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost));

    double max_abs = 0; size_t exact = 0, bad = 0;
    for (size_t k = 0; k < got.size(); k++) {
      float g = h2f(got[k]), e = h2f(expected[k]);
      double d = fabs((double)g - (double)e);
      if (got[k] == expected[k]) exact++;
      if (d > max_abs) max_abs = d;
      if (!(d <= 1e-3 + 4e-3 * fabs((double)e))) bad++;   // about 2 fp16 steps
    }
    bool ok = bad == 0;
    printf("prompt %d: %3d tokens  max_abs_diff %.2e  exact %.2f%%  %s\n",
           i, n, max_abs, 100.0 * exact / got.size(), ok ? "PASS" : "FAIL");
    if (!ok) failures++;
    cudaFree(d_x); cudaFree(d_out);
  }
  cudaFree(d_w);
  return failures ? 1 : 0;
}
