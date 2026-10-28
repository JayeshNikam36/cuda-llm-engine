#include <cstdio>
#include <cstdlib>
#include "cached.h"
#include "util.h"
#include "embedding.cuh"
#include "rmsnorm.cuh"
#include "linear.cuh"
#include "rope.cuh"
#include "attention_naive.cuh"
#include "attention_decode_naive.cuh"
#include "swiglu.cuh"
#include "residual_add.cuh"

KVCache alloc_kv_cache(const Model& m, int max_seq) {
  KVCache c;
  c.max_seq = max_seq;
  const size_t bytes = (size_t)max_seq * m.cfg.n_kv_heads * m.cfg.head_dim * sizeof(__half);
  for (int l = 0; l < m.n_layers; l++) {
    __half *k = nullptr, *v = nullptr;
    CUDA_CHECK(cudaMalloc(&k, bytes));
    CUDA_CHECK(cudaMalloc(&v, bytes));
    CUDA_CHECK(cudaMemset(k, 0, bytes));
    CUDA_CHECK(cudaMemset(v, 0, bytes));
    c.k.push_back(k);
    c.v.push_back(v);
  }
  return c;
}

void free_kv_cache(KVCache& c) {
  for (auto p : c.k) cudaFree(p);
  for (auto p : c.v) cudaFree(p);
  c.k.clear();
  c.v.clear();
}

// Like layer_forward, but also appends this step's K and V to the cache.
//   start_pos == 0: prefill, attention over the T new tokens themselves (same as before).
//   start_pos  > 0: decode, T must be 1, attention over the whole cache.
static void layer_forward_cached(cublasHandle_t handle, const LayerConfig& c, const LayerWeights& w,
                                 LayerScratch& s, __half* x, const int* d_positions,
                                 const float* d_inv_freq, int T, int start_pos,
                                 __half* k_cache, __half* v_cache) {
  if (start_pos > 0 && T != 1) { fprintf(stderr, "decode step must have T == 1\n"); exit(1); }
  const int qdim = c.n_heads * c.head_dim, kvdim = c.n_kv_heads * c.head_dim;
  const size_t n_hidden = (size_t)T * c.hidden;

  rmsnorm(x, w.attn_norm, s.norm, T, c.hidden, c.eps);
  linear(handle, s.norm, w.wq, s.q, T, c.hidden, qdim);
  linear(handle, s.norm, w.wk, s.k, T, c.hidden, kvdim);
  linear(handle, s.norm, w.wv, s.v, T, c.hidden, kvdim);
  rope_apply(s.q, s.k, d_positions, d_inv_freq, T, c.n_heads, c.n_kv_heads, c.head_dim);

  // Append this step's keys and values to the cache.
  const size_t kvbytes = (size_t)T * kvdim * sizeof(__half);
  CUDA_CHECK(cudaMemcpyAsync(k_cache + (size_t)start_pos * kvdim, s.k, kvbytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpyAsync(v_cache + (size_t)start_pos * kvdim, s.v, kvbytes, cudaMemcpyDeviceToDevice));

  if (start_pos == 0)
    attention_naive(s.q, s.k, s.v, s.attn, T, c.n_heads, c.n_kv_heads, c.head_dim);
  else
    attention_decode_naive(s.q, k_cache, v_cache, s.attn, start_pos + 1, c.n_heads, c.n_kv_heads, c.head_dim);

  linear(handle, s.attn, w.wo, s.proj, T, qdim, c.hidden);
  residual_add(x, s.proj, x, n_hidden);

  rmsnorm(x, w.mlp_norm, s.norm, T, c.hidden, c.eps);
  linear(handle, s.norm, w.w_gate, s.gate, T, c.hidden, c.inter);
  linear(handle, s.norm, w.w_up,   s.up,   T, c.hidden, c.inter);
  swiglu(s.gate, s.up, s.act, (size_t)T * c.inter);
  linear(handle, s.act, w.w_down, s.proj, T, c.inter, c.hidden);
  residual_add(x, s.proj, x, n_hidden);
}

// Final norm on the last row, tied output head, copy logits to the host.
static void head_logits(Model& m, int T, std::vector<float>& logits) {
  const int H = m.cfg.hidden;
  const __half* last = m.d_x + (size_t)(T - 1) * H;
  rmsnorm(last, m.final_norm, m.d_normed, 1, H, m.cfg.eps);
  linear(m.handle, m.d_normed, m.embed, m.d_logits, 1, H, m.vocab);
  CUDA_CHECK(cudaGetLastError());
  std::vector<__half> host(m.vocab);
  CUDA_CHECK(cudaMemcpy(host.data(), m.d_logits, (size_t)m.vocab * sizeof(__half), cudaMemcpyDeviceToHost));
  logits.resize(m.vocab);
  for (int v = 0; v < m.vocab; v++) logits[v] = __half2float(host[v]);
}

void model_prefill(Model& m, KVCache& c, const std::vector<int>& ids, std::vector<float>& logits) {
  const int T = (int)ids.size();
  if (T > m.max_T || T > c.max_seq) { fprintf(stderr, "prompt of %d tokens is too long\n", T); exit(1); }
  std::vector<int> pos(T);
  for (int t = 0; t < T; t++) pos[t] = t;
  CUDA_CHECK(cudaMemcpy(m.d_ids, ids.data(), T * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m.d_pos, pos.data(), T * sizeof(int), cudaMemcpyHostToDevice));
  embedding_lookup(m.embed, m.d_ids, m.d_x, T, m.cfg.hidden);
  for (int l = 0; l < m.n_layers; l++)
    layer_forward_cached(m.handle, m.cfg, m.layers[l], m.scratch, m.d_x, m.d_pos, m.d_inv_freq,
                         T, 0, c.k[l], c.v[l]);
  head_logits(m, T, logits);
}

void model_decode(Model& m, KVCache& c, int token, int pos, std::vector<float>& logits) {
  if (pos + 1 > c.max_seq) { fprintf(stderr, "position %d exceeds the cache size %d\n", pos, c.max_seq); exit(1); }
  CUDA_CHECK(cudaMemcpy(m.d_ids, &token, sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m.d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice));
  embedding_lookup(m.embed, m.d_ids, m.d_x, 1, m.cfg.hidden);
  for (int l = 0; l < m.n_layers; l++)
    layer_forward_cached(m.handle, m.cfg, m.layers[l], m.scratch, m.d_x, m.d_pos, m.d_inv_freq,
                         1, pos, c.k[l], c.v[l]);
  head_logits(m, 1, logits);
}
