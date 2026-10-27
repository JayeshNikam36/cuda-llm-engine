#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include "util.h"
#include "model.h"
#include "embedding.cuh"
#include "rmsnorm.cuh"
#include "linear.cuh"
#include "rope.cuh"
#include "attention_naive.cuh"
#include "swiglu.cuh"
#include "residual_add.cuh"

static float timed(cudaEvent_t a, cudaEvent_t b, const std::function<void()>& fn) {
  CUDA_CHECK(cudaEventRecord(a));
  fn();
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";

  const int MAXT = 402;
  Model m = load_model(W, MAXT);
  auto ids = read_file<int32_t>(R + "/prompt_07_ids.i32");
  std::vector<int> pos(MAXT);
  for (int t = 0; t < MAXT; t++) pos[t] = t;
  CUDA_CHECK(cudaMemcpy(m.d_ids, ids.data(), MAXT * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m.d_pos, pos.data(), MAXT * sizeof(int), cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  const char* names[] = {"attn rmsnorm", "q/k/v projections", "rope", "attention (naive)", "o_proj",
                         "residual add 1", "mlp rmsnorm", "gate/up projections", "swiglu",
                         "down_proj", "residual add 2"};
  const int NS = 11;
  const int Ts[] = {1, 32, 402};
  const int WARMUP = 3, REPS = 11;
  const LayerConfig& c = m.cfg;
  const int H = c.hidden, qd = c.n_heads * c.head_dim, kvd = c.n_kv_heads * c.head_dim;
  LayerWeights& w = m.layers[0];
  LayerScratch& s = m.scratch;

  std::string md = "# Per-stage time of one layer (layer 0), scaled to 16 layers\n\n";
  for (int T : Ts) {
    std::vector<std::vector<float>> st(NS);
    std::vector<float> head;
    for (int rep = 0; rep < WARMUP + REPS; rep++) {
      embedding_lookup(m.embed, m.d_ids, m.d_x, T, H);   // fresh, realistic input every repetition
      float t[NS];
      t[0]  = timed(e0, e1, [&] { rmsnorm(m.d_x, w.attn_norm, s.norm, T, H, c.eps); });
      t[1]  = timed(e0, e1, [&] {
        linear(m.handle, s.norm, w.wq, s.q, T, H, qd);
        linear(m.handle, s.norm, w.wk, s.k, T, H, kvd);
        linear(m.handle, s.norm, w.wv, s.v, T, H, kvd); });
      t[2]  = timed(e0, e1, [&] { rope_apply(s.q, s.k, m.d_pos, m.d_inv_freq, T, c.n_heads, c.n_kv_heads, c.head_dim); });
      t[3]  = timed(e0, e1, [&] { attention_naive(s.q, s.k, s.v, s.attn, T, c.n_heads, c.n_kv_heads, c.head_dim); });
      t[4]  = timed(e0, e1, [&] { linear(m.handle, s.attn, w.wo, s.proj, T, qd, H); });
      t[5]  = timed(e0, e1, [&] { residual_add(m.d_x, s.proj, m.d_x, (size_t)T * H); });
      t[6]  = timed(e0, e1, [&] { rmsnorm(m.d_x, w.mlp_norm, s.norm, T, H, c.eps); });
      t[7]  = timed(e0, e1, [&] {
        linear(m.handle, s.norm, w.w_gate, s.gate, T, H, c.inter);
        linear(m.handle, s.norm, w.w_up, s.up, T, H, c.inter); });
      t[8]  = timed(e0, e1, [&] { swiglu(s.gate, s.up, s.act, (size_t)T * c.inter); });
      t[9]  = timed(e0, e1, [&] { linear(m.handle, s.act, w.w_down, s.proj, T, c.inter, H); });
      t[10] = timed(e0, e1, [&] { residual_add(m.d_x, s.proj, m.d_x, (size_t)T * H); });
      float th = timed(e0, e1, [&] {
        rmsnorm(m.d_x + (size_t)(T - 1) * H, m.final_norm, m.d_normed, 1, H, c.eps);
        linear(m.handle, m.d_normed, m.embed, m.d_logits, 1, H, m.vocab); });
      if (rep >= WARMUP) { for (int i = 0; i < NS; i++) st[i].push_back(t[i]); head.push_back(th); }
    }
    CUDA_CHECK(cudaGetLastError());

    float med[NS], sum = 0;
    for (int i = 0; i < NS; i++) { med[i] = median(st[i]); sum += med[i]; }
    float head_ms = median(head);
    float total = 16.0f * sum + head_ms;

    char buf[256];
    snprintf(buf, sizeof(buf), "## %d tokens (estimated total %.1f ms)\n\n", T, total);
    md += buf;
    md += "| stage | ms per layer | ms x 16 layers | share of pass |\n|---|---:|---:|---:|\n";
    for (int i = 0; i < NS; i++) {
      snprintf(buf, sizeof(buf), "| %s | %.3f | %.2f | %.1f%% |\n", names[i], med[i], 16.0f * med[i], 100.0f * 16.0f * med[i] / total);
      md += buf;
    }
    snprintf(buf, sizeof(buf), "| final norm + lm_head | n/a | %.2f | %.1f%% |\n\n", head_ms, 100.0f * head_ms / total);
    md += buf;
  }
  md += "Each stage is timed with its own events and a sync, so the estimated total runs a little above the measured pass in docs/baseline_nocache.md.\n";
  printf("%s", md.c_str());
  FILE* f = fopen("/workspace/docs/breakdown_nocache.md", "w");
  if (f) { fputs(md.c_str(), f); fclose(f); }
  free_model(m);
  return 0;
}
