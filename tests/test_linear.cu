#include <cstring>
#include <cstdint>
#include <cmath>
#include "util.h"
#include "linear.cuh"

static float h2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

// Runs one projection for all 8 prompts. Returns the number of failing prompts.
static int run(cublasHandle_t handle, const std::string& W, const std::string& R,
               const char* weight_name, const char* expected_name) {
  TensorInfo ti = find_tensor(W + "/index.tsv", weight_name);
  int out_f = (int)ti.shape[0], in_f = (int)ti.shape[1];
  auto w = read_file<uint16_t>(W + "/weights.bin", ti.offset, ti.nbytes / 2);
  __half* d_w;
  CUDA_CHECK(cudaMalloc(&d_w, ti.nbytes));
  CUDA_CHECK(cudaMemcpy(d_w, w.data(), ti.nbytes, cudaMemcpyHostToDevice));
  printf("%s  [%d x %d]\n", weight_name, out_f, in_f);

  int failures = 0;
  for (int i = 0; i < 8; i++) {
    char p[256];
    snprintf(p, sizeof(p), "%s/prompt_%02d_norm0.f16", R.c_str(), i);
    auto x = read_file<uint16_t>(p);
    snprintf(p, sizeof(p), "%s/prompt_%02d_%s.f16", R.c_str(), i, expected_name);
    auto expected = read_file<uint16_t>(p);
    int T = (int)(x.size() / in_f);
    if (expected.size() != (size_t)T * out_f) { printf("  prompt %d: SIZE MISMATCH\n", i); failures++; continue; }

    __half *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, x.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_y, expected.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    linear(handle, d_x, d_w, d_y, T, in_f, out_f);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got(expected.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_y, got.size() * 2, cudaMemcpyDeviceToHost));
    double max_abs = 0; size_t exact = 0, bad = 0;
    for (size_t k = 0; k < got.size(); k++) {
      double g = h2f(got[k]), e = h2f(expected[k]), d = fabs(g - e);
      if (got[k] == expected[k]) exact++;
      if (d > max_abs) max_abs = d;
      if (!(d <= 3e-3 + 8e-3 * fabs(e))) bad++;
    }
    bool ok = bad == 0;
    printf("  prompt %d: %3d tokens  max_abs_diff %.2e  exact %.2f%%  %s\n",
           i, T, max_abs, 100.0 * exact / got.size(), ok ? "PASS" : "FAIL");
    if (!ok) failures++;
    cudaFree(d_x); cudaFree(d_y);
  }
  cudaFree(d_w);
  return failures;
}

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";
  cublasHandle_t handle;
  cublasCreate(&handle);
  int failures = 0;
  failures += run(handle, W, R, "model.layers.0.self_attn.q_proj.weight", "q_in");
  failures += run(handle, W, R, "model.layers.0.self_attn.k_proj.weight", "k_in");
  cublasDestroy(handle);
  return failures ? 1 : 0;
}