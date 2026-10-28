#include <cstdio>
#include <cstdlib>
#include "paged.h"
#include "util.h"
#include "embedding.cuh"
#include "rmsnorm.cuh"
#include "linear.cuh"
#include "rope.cuh"
#include "attention_naive.cuh"
#include "attention_decode_paged.cuh"
#include "kv_scatter.cuh"
#include "swiglu.cuh"
#include "residual_add.cuh"

PagedKVCache alloc_paged_cache(const Model& m, int num_pages, int page_size) {
  PagedKVCache c(num_pages, page_size);
  const size_t bytes = (size_t)num_pages * page_size * m.cfg.n_kv_heads * m.cfg.head_dim * sizeof(__half);
  for (int l = 0; l < m.n_layers; l++) {
    __half *k = nullptr, *v = nullptr;
    CUDA_CHECK(cudaMalloc(&k, bytes));
    CUDA_CHECK(cudaMalloc(&v, bytes));
    CUDA_CHECK(cudaMemset(k, 0x7F, bytes));   // 0x7F7F is a NaN in fp16: reading an unmapped slot poisons the output
    CUDA_CHECK(cudaMemset(v, 0x7F, bytes));
    c.k_pool.push_back(k);
    c.v_pool.push_back(v);
  }
  return c;
}

void free_paged_cache(PagedKVCache& c) {
  for (auto p : c.k_pool) cudaFree(p);
  for (auto p : c.v_pool) cudaFree(p);
  c.k_pool.clear();
  c.v_pool.clear();
}

Sequence new_sequence(int max_pages) {
  Sequence s;
  s.max_pages = max_pages;
  CUDA_CHECK(cudaMalloc(&s.d_table, max_pages * sizeof(int)));
  return s;
}

void free_sequence(PagedKVCache& c, Sequence& s) {
  for (int p : s.pages) c.alloc.release(p);
  s.pages.clear();
  s.len = 0;
  if (s.d_table) cudaFree(s.d_table);
  s.d_table = nullptr;
}

// Make sure the sequence owns enough pages for new_len tokens, then refresh its block table on the GPU.
static void reserve(PagedKVCache& c, Sequence& s, int new_len) {
  while ((int)s.pages.size() * c.page_size < new_len) s.pages.push_back(c.alloc.alloc());
  if ((int)s.pages.size() > s.max_pages) {
    fprintf(stderr, "sequence needs %zu pages, block table holds %d\n", s.pages.size(), s.max_pages);
    exit(1);
  }
  CUDA_CHECK(cudaMemcpy(s.d_table, s.pages.data(), s.pages.size() * sizeof(int), cudaMemcpyHostToDevice));
}

// Same layer math as layer_forward / layer_forward_cached, but K and V go to and come from pages.
//   start_pos == 0: prefill, attention over the T new tokens themselves.
//   start_pos  > 0: decode (T == 1), attention over every stored position through the block table.
static void layer_forward_paged(cublasHandle_t handle, const LayerConfig& c, const LayerWeights& w,
                                LayerScratch& s, __half* x, const int* d_positions,
                                const float* d_inv_freq, int T, int start_pos,
                                __half* k_pool, __half* v_pool, const int* d_table, int page_size) {
  if (start_pos > 0 && T != 1) { fprintf(stderr, "decode step must have T == 1\n"); exit(1); }
  const int qdim = c.n_heads * c.head_dim, kvdim = c.n_kv_heads * c.head_dim;
  const size_t n_hidden = (size_t)T * c.hidden;

  rmsnorm(x, w.attn_norm, s.norm, T, c.hidden, c.eps);
  linear(handle, s.norm, w.wq, s.q, T, c.hidden, qdim);
  linear(handle, s.norm, w.wk, s.k, T, c.hidden, kvdim);
  linear(handle, s.norm, w.wv, s.v, T, c.hidden, kvdim);
  rope_apply(s.q, s.k, d_positions, d_inv_freq, T, c.n_heads, c.n_kv_heads, c.head_dim);

  kv_scatter(s.k, k_pool, d_table, start_pos, T, page_size, kvdim);
  kv_scatter(s.v, v_pool, d_table, start_pos, T, page_size, kvdim);

  if (start_pos == 0)
    attention_naive(s.q, s.k, s.v, s.attn, T, c.n_heads, c.n_kv_heads, c.head_dim);
  else
    attention_decode_paged(s.q, k_pool, v_pool, d_table, s.attn, start_pos + 1, page_size,
                           c.n_heads, c.n_kv_heads, c.head_dim);

  linear(handle, s.attn, w.wo, s.proj, T, qdim, c.hidden);
  residual_add(x, s.proj, x, n_hidden);

  rmsnorm(x, w.mlp_norm, s.norm, T, c.hidden, c.eps);
  linear(handle, s.norm, w.w_gate, s.gate, T, c.hidden, c.inter);
  linear(handle, s.norm, w.w_up,   s.up,   T, c.hidden, c.inter);
  swiglu(s.gate, s.up, s.act, (size_t)T * c.inter);
  linear(handle, s.act, w.w_down, s.proj, T, c.inter, c.hidden);
  residual_add(x, s.proj, x, n_hidden);
}

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

void model_prefill_paged(Model& m, PagedKVCache& c, Sequence& seq, const std::vector<int>& ids,
                         std::vector<float>& logits) {
  const int T = (int)ids.size();
  if (seq.len != 0) { fprintf(stderr, "prefill needs an empty sequence\n"); exit(1); }
  if (T > m.max_T) { fprintf(stderr, "prompt of %d tokens is too long\n", T); exit(1); }
  reserve(c, seq, T);
  std::vector<int> pos(T);
  for (int t = 0; t < T; t++) pos[t] = t;
  CUDA_CHECK(cudaMemcpy(m.d_ids, ids.data(), T * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m.d_pos, pos.data(), T * sizeof(int), cudaMemcpyHostToDevice));
  embedding_lookup(m.embed, m.d_ids, m.d_x, T, m.cfg.hidden);
  for (int l = 0; l < m.n_layers; l++)
    layer_forward_paged(m.handle, m.cfg, m.layers[l], m.scratch, m.d_x, m.d_pos, m.d_inv_freq,
                        T, 0, c.k_pool[l], c.v_pool[l], seq.d_table, c.page_size);
  head_logits(m, T, logits);
  seq.len = T;
}

void model_decode_paged(Model& m, PagedKVCache& c, Sequence& seq, int token, std::vector<float>& logits) {
  const int pos = seq.len;
  reserve(c, seq, pos + 1);
  CUDA_CHECK(cudaMemcpy(m.d_ids, &token, sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m.d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice));
  embedding_lookup(m.embed, m.d_ids, m.d_x, 1, m.cfg.hidden);
  for (int l = 0; l < m.n_layers; l++)
    layer_forward_paged(m.handle, m.cfg, m.layers[l], m.scratch, m.d_x, m.d_pos, m.d_inv_freq,
                        1, pos, c.k_pool[l], c.v_pool[l], seq.d_table, c.page_size);
  head_logits(m, 1, logits);
  seq.len = pos + 1;
}
