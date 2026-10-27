#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "util.h"
#include "model.h"
#include "embedding.cuh"
#include "rmsnorm.cuh"
#include "linear.cuh"

// One full forward pass on the GPU, ending with the last token's logits on the device.
// No host copies, so the timing is GPU work only.
static void forward_gpu(Model& m, int T) {
  const int H = m.cfg.hidden;
  embedding_lookup(m.embed, m.d_ids, m.d_x, T, H);
  for (int l = 0; l < m.n_layers; l++)
    layer_forward(m.handle, m.cfg, m.layers[l], m.scratch, m.d_x, m.d_pos, m.d_inv_freq, T);
  const __half* last = m.d_x + (size_t)(T - 1) * H;
  rmsnorm(last, m.final_norm, m.d_normed, 1, H, m.cfg.eps);
  linear(m.handle, m.d_normed, m.embed, m.d_logits, 1, H, m.vocab);
}

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  const char* pk = getenv("PEAK_GBPS");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";
  const double peak = pk ? atof(pk) : 175.0;     // measured practical ceiling in GB/s (docs/bandwidth.md)
  const double weight_bytes = 2.472e9;           // FP16 model size (docs/architecture.md)

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int MAXT = 402;
  Model m = load_model(W, MAXT);

  auto ids = read_file<int32_t>(R + "/prompt_07_ids.i32");
  if ((int)ids.size() < MAXT) { fprintf(stderr, "need %d prompt tokens, found %zu\n", MAXT, ids.size()); return 1; }
  std::vector<int> pos(MAXT);
  for (int t = 0; t < MAXT; t++) pos[t] = t;
  CUDA_CHECK(cudaMemcpy(m.d_ids, ids.data(), MAXT * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m.d_pos, pos.data(), MAXT * sizeof(int), cudaMemcpyHostToDevice));

  const int Ts[] = {1, 8, 32, 128, 402};
  const int WARMUP = 3, TRIALS = 15;
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  std::string md;
  char buf[512];
  snprintf(buf, sizeof(buf),
           "# Baseline: full forward pass, no KV cache\n\nGPU: %s | peak used for %%: %.0f GB/s | %d warmup + %d timed runs per row\n\n",
           prop.name, peak, WARMUP, TRIALS);
  md += buf;
  md += "| tokens in sequence | median ms | min ms | max ms | words/s | weight GB/s | % of peak |\n";
  md += "|---:|---:|---:|---:|---:|---:|---:|\n";

  for (int T : Ts) {
    for (int w = 0; w < WARMUP; w++) forward_gpu(m, T);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int t = 0; t < TRIALS; t++) {
      CUDA_CHECK(cudaEventRecord(e0));
      forward_gpu(m, T);
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float x; CUDA_CHECK(cudaEventElapsedTime(&x, e0, e1));
      ms.push_back(x);
    }
    CUDA_CHECK(cudaGetLastError());
    std::sort(ms.begin(), ms.end());
    float med = ms[TRIALS / 2];
    double gbps = weight_bytes / (med * 1e-3) / 1e9;
    snprintf(buf, sizeof(buf), "| %d | %.2f | %.2f | %.2f | %.1f | %.1f | %.1f%% |\n",
             T, med, ms.front(), ms.back(), 1000.0 / med, gbps, 100.0 * gbps / peak);
    md += buf;
  }
  md += "\nNotes: 'weight GB/s' = 2.472 GB of weights divided by the median time. It is only meaningful when the pass is memory-bound (few tokens); at long sequences compute and the naive attention dominate.\n";

  printf("%s", md.c_str());
  FILE* f = fopen("/workspace/docs/baseline_nocache.md", "w");
  if (f) { fputs(md.c_str(), f); fclose(f); }
  free_model(m);
  return 0;
}
