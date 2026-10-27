#include <cstring>
#include <cstdint>
#include <cmath>
#include "util.h"
#include "rope.cuh"

static float h2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

struct Cmp { double max_abs = 0; double exact_pct = 0; size_t bad = 0; };
static Cmp compare(const std::vector<uint16_t>& got, const std::vector<uint16_t>& exp) {
  Cmp c; size_t exact = 0;
  for (size_t k = 0; k < got.size(); k++) {
    double g = h2f(got[k]), e = h2f(exp[k]), d = fabs(g - e);
    if (got[k] == exp[k]) exact++;
    if (d > c.max_abs) c.max_abs = d;
    if (!(d <= 1e-3 + 4e-3 * fabs(e))) c.bad++;
  }
  c.exact_pct = 100.0 * exact / got.size();
  return c;
}

int main() {
  const char* re = getenv("REF_DIR");
  std::string R = re ? re : "/workspace/tests/reference/bin";
  const int NH = 32, NKV = 8, HD = 64;   // from config.json

  auto inv = compute_rope_inv_freq(HD, 500000.0, 32.0, 1.0, 4.0, 8192.0);
  auto ref_inv = read_file<float>(R + "/rope_inv_freq.f32");
  double max_rel = 0;
  for (int i = 0; i < HD / 2; i++)
    max_rel = fmax(max_rel, fabs((double)inv[i] - ref_inv[i]) / fabs((double)ref_inv[i]));
  printf("inv_freq: max relative difference vs Hugging Face = %.2e\n", max_rel);
  if (max_rel > 1e-5) { printf("FAIL: inv_freq (llama3 scaling logic differs)\n"); return 1; }

  float* d_inv;
    const bool use_ref = getenv("USE_REF_INVFREQ") != nullptr;
    const std::vector<float>& table = use_ref ? ref_inv : inv;
    printf("using %s rotation speeds\n", use_ref ? "Hugging Face" : "my");
    CUDA_CHECK(cudaMalloc(&d_inv, table.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_inv, table.data(), table.size() * sizeof(float), cudaMemcpyHostToDevice));

  int failures = 0;
  for (int i = 0; i < 8; i++) {
    char p[256];
    auto load = [&](const char* name) {
      snprintf(p, sizeof(p), "%s/prompt_%02d_%s.f16", R.c_str(), i, name);
      return read_file<uint16_t>(p);
    };
    auto q_in = load("q_in"), k_in = load("k_in"), q_exp = load("q_rope"), k_exp = load("k_rope");
    int T = (int)(q_in.size() / (NH * HD));
    if (k_in.size() != (size_t)T * NKV * HD || q_exp.size() != q_in.size() || k_exp.size() != k_in.size()) {
      printf("prompt %d: SIZE MISMATCH\n", i); failures++; continue;
    }
    std::vector<int> pos(T);
    for (int t = 0; t < T; t++) pos[t] = t;

    __half *dq, *dk; int* dpos;
    CUDA_CHECK(cudaMalloc(&dq, q_in.size() * 2));
    CUDA_CHECK(cudaMalloc(&dk, k_in.size() * 2));
    CUDA_CHECK(cudaMalloc(&dpos, T * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dq, q_in.data(), q_in.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, k_in.data(), k_in.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpos, pos.data(), T * sizeof(int), cudaMemcpyHostToDevice));
    rope_apply(dq, dk, dpos, d_inv, T, NH, NKV, HD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> q_got(q_in.size()), k_got(k_in.size());
    CUDA_CHECK(cudaMemcpy(q_got.data(), dq, q_got.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_got.data(), dk, k_got.size() * 2, cudaMemcpyDeviceToHost));
    Cmp cq = compare(q_got, q_exp), ck = compare(k_got, k_exp);
    bool ok = cq.bad == 0 && ck.bad == 0;
    printf("prompt %d: %3d tokens  Q max_diff %.2e exact %.2f%% | K max_diff %.2e exact %.2f%%  %s\n",
           i, T, cq.max_abs, cq.exact_pct, ck.max_abs, ck.exact_pct, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
    cudaFree(dq); cudaFree(dk); cudaFree(dpos);
  }
  cudaFree(d_inv);
  return failures ? 1 : 0;
}
