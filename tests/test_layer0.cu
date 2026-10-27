#include <cstring>
#include <cstdint>
#include <cmath>
#include "util.h"
#include "weights.h"
#include "layer.h"
#include "rope.cuh"

static float h2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";

  LayerConfig cfg;
  LayerWeights w = load_layer_weights(W, 0);
  auto inv = compute_rope_inv_freq(cfg.head_dim, 500000.0, 32.0, 1.0, 4.0, 8192.0);
  float* d_inv;
  CUDA_CHECK(cudaMalloc(&d_inv, inv.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_inv, inv.data(), inv.size() * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  cublasCreate(&handle);
  const int MAXT = 402;
  LayerScratch s = alloc_scratch(cfg, MAXT);
  __half* d_x;  int* d_pos;
  CUDA_CHECK(cudaMalloc(&d_x, (size_t)MAXT * cfg.hidden * 2));
  CUDA_CHECK(cudaMalloc(&d_pos, MAXT * sizeof(int)));

  int failures = 0;
  for (int i = 0; i < 8; i++) {
    char p[256];
    snprintf(p, sizeof(p), "%s/prompt_%02d_embed.f16", R.c_str(), i);
    auto x_in = read_file<uint16_t>(p);
    snprintf(p, sizeof(p), "%s/prompt_%02d_layer0.f16", R.c_str(), i);
    auto expected = read_file<uint16_t>(p);
    int T = (int)(x_in.size() / cfg.hidden);
    if (expected.size() != x_in.size() || T > MAXT) { printf("prompt %d: SIZE MISMATCH\n", i); failures++; continue; }

    std::vector<int> pos(T);
    for (int t = 0; t < T; t++) pos[t] = t;
    CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), T * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x_in.data(), x_in.size() * 2, cudaMemcpyHostToDevice));

    layer_forward(handle, cfg, w, s, d_x, d_pos, d_inv, T);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got(x_in.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_x, got.size() * 2, cudaMemcpyDeviceToHost));

    double num = 0, den = 0, max_abs = 0; size_t exact = 0;
    for (size_t k = 0; k < got.size(); k++) {
      double g = h2f(got[k]), e = h2f(expected[k]), d = g - e;
      num += d * d; den += e * e;
      if (fabs(d) > max_abs) max_abs = fabs(d);
      if (got[k] == expected[k]) exact++;
    }
    double rel = sqrt(num / den);
    bool ok = rel < 5e-3;          // NaN also fails this comparison
    printf("prompt %d: %3d tokens  rel_L2 %.2e  max_abs_diff %.2e  exact %.2f%%  %s\n",
           i, T, rel, max_abs, 100.0 * exact / got.size(), ok ? "PASS" : "FAIL");
    if (!ok) failures++;
  }
  return failures ? 1 : 0;
}
