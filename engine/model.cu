#include <cstdio>
#include <cstdlib>
#include "model.h"
#include "weights.h"
#include "embedding.cuh"
#include "rmsnorm.cuh"
#include "linear.cuh"
#include "rope.cuh"

Model load_model(const std::string& dir, int max_T) {
  Model m;
  m.max_T = max_T;
  std::vector<long> shape;
  m.embed = load_weight(dir, "model.embed_tokens.weight", &shape);
  m.vocab = (int)shape[0];
  m.cfg.hidden = (int)shape[1];
  m.final_norm = load_weight(dir, "model.norm.weight");
  for (int l = 0; l < m.n_layers; l++) m.layers.push_back(load_layer_weights(dir, l));

  auto inv = compute_rope_inv_freq(m.cfg.head_dim, 500000.0, 32.0, 1.0, 4.0, 8192.0);
  CUDA_CHECK(cudaMalloc(&m.d_inv_freq, inv.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(m.d_inv_freq, inv.data(), inv.size() * sizeof(float), cudaMemcpyHostToDevice));

  cublasCreate(&m.handle);
  m.scratch = alloc_scratch(m.cfg, max_T);
  CUDA_CHECK(cudaMalloc(&m.d_x, (size_t)max_T * m.cfg.hidden * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&m.d_normed, (size_t)m.cfg.hidden * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&m.d_logits, (size_t)m.vocab * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&m.d_ids, max_T * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&m.d_pos, max_T * sizeof(int)));
  return m;
}

void free_model(Model& m) {
  for (auto& l : m.layers) free_layer_weights(l);
  free_scratch(m.scratch);
  cudaFree(m.embed); cudaFree(m.final_norm); cudaFree(m.d_inv_freq);
  cudaFree(m.d_x); cudaFree(m.d_normed); cudaFree(m.d_logits);
  cudaFree(m.d_ids); cudaFree(m.d_pos);
  cublasDestroy(m.handle);
}

void model_last_logits(Model& m, const std::vector<int>& ids, std::vector<float>& logits) {
  const int T = (int)ids.size();
  if (T > m.max_T) { fprintf(stderr, "sequence of %d tokens exceeds max_T %d\n", T, m.max_T); exit(1); }
  const int H = m.cfg.hidden;

  std::vector<int> pos(T);
  for (int t = 0; t < T; t++) pos[t] = t;
  CUDA_CHECK(cudaMemcpy(m.d_ids, ids.data(), T * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m.d_pos, pos.data(), T * sizeof(int), cudaMemcpyHostToDevice));

  embedding_lookup(m.embed, m.d_ids, m.d_x, T, H);
  for (int l = 0; l < m.n_layers; l++)
    layer_forward(m.handle, m.cfg, m.layers[l], m.scratch, m.d_x, m.d_pos, m.d_inv_freq, T);

  // Only the last word's logits are needed: final norm on that row, then the tied output head.
  const __half* last = m.d_x + (size_t)(T - 1) * H;
  rmsnorm(last, m.final_norm, m.d_normed, 1, H, m.cfg.eps);
  linear(m.handle, m.d_normed, m.embed, m.d_logits, 1, H, m.vocab);
  CUDA_CHECK(cudaGetLastError());

  std::vector<__half> host(m.vocab);
  CUDA_CHECK(cudaMemcpy(host.data(), m.d_logits, (size_t)m.vocab * sizeof(__half), cudaMemcpyDeviceToHost));
  logits.resize(m.vocab);
  for (int v = 0; v < m.vocab; v++) logits[v] = __half2float(host[v]);
}
