#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "batched.h"
#include "util.h"
#include "embedding.cuh"
#include "rmsnorm.cuh"
#include "linear.cuh"
#include "rope.cuh"
#include "attention_decode_fast.cuh"
#include "kv_scatter.cuh"
#include "swiglu.cuh"
#include "residual_add.cuh"
#include "argmax.cuh"

using Clock = std::chrono::steady_clock;
static double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

BatchState alloc_batch_state(const Model& m, int max_batch, int max_pages) {
  BatchState b;
  b.max_batch = max_batch;
  b.max_pages = max_pages;
  CUDA_CHECK(cudaMalloc(&b.d_tables, (size_t)max_batch * max_pages * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b.d_lens, max_batch * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b.d_next, max_batch * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b.d_ws, attention_decode_fast_workspace_floats(max_batch, m.cfg.n_heads, 32) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b.d_normed, (size_t)max_batch * m.cfg.hidden * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&b.d_logits, (size_t)max_batch * m.vocab * sizeof(__half)));
  CUDA_CHECK(cudaEventCreate(&b.ev0));
  CUDA_CHECK(cudaEventCreate(&b.ev1));
  return b;
}

void free_batch_state(BatchState& b) {
  cudaFree(b.d_tables); cudaFree(b.d_lens); cudaFree(b.d_next); cudaFree(b.d_ws);
  cudaFree(b.d_normed); cudaFree(b.d_logits);
  cudaEventDestroy(b.ev0); cudaEventDestroy(b.ev1);
}

// Same layer math as before, with n rows (one new token per sequence), paged writes, and the fast attention kernel.
static void layer_forward_batched(cublasHandle_t handle, const LayerConfig& c, const LayerWeights& w,
                                  LayerScratch& s, __half* x, const int* d_positions, const float* d_inv_freq,
                                  int n, __half* k_pool, __half* v_pool, const BatchState& b,
                                  int page_size, int num_splits) {
  const int qdim = c.n_heads * c.head_dim, kvdim = c.n_kv_heads * c.head_dim;
  const size_t n_hidden = (size_t)n * c.hidden;

  rmsnorm(x, w.attn_norm, s.norm, n, c.hidden, c.eps);
  linear(handle, s.norm, w.wq, s.q, n, c.hidden, qdim);
  linear(handle, s.norm, w.wk, s.k, n, c.hidden, kvdim);
  linear(handle, s.norm, w.wv, s.v, n, c.hidden, kvdim);
  rope_apply(s.q, s.k, d_positions, d_inv_freq, n, c.n_heads, c.n_kv_heads, c.head_dim);

  kv_scatter_batch(s.k, k_pool, b.d_tables, d_positions, n, b.max_pages, page_size, kvdim);
  kv_scatter_batch(s.v, v_pool, b.d_tables, d_positions, n, b.max_pages, page_size, kvdim);

  attention_decode_fast(s.q, k_pool, v_pool, b.d_tables, b.d_lens, s.attn, b.d_ws,
                        n, b.max_pages, page_size, c.n_heads, c.n_kv_heads, c.head_dim, num_splits);

  linear(handle, s.attn, w.wo, s.proj, n, qdim, c.hidden);
  residual_add(x, s.proj, x, n_hidden);

  rmsnorm(x, w.mlp_norm, s.norm, n, c.hidden, c.eps);
  linear(handle, s.norm, w.w_gate, s.gate, n, c.hidden, c.inter);
  linear(handle, s.norm, w.w_up,   s.up,   n, c.hidden, c.inter);
  swiglu(s.gate, s.up, s.act, (size_t)n * c.inter);
  linear(handle, s.act, w.w_down, s.proj, n, c.inter, c.hidden);
  residual_add(x, s.proj, x, n_hidden);
}

// Make sure the sequence owns enough pages for new_len tokens (this duplicates the small helper in paged.cu).
static void ensure_pages(PagedKVCache& c, Sequence& s, int new_len) {
  while ((int)s.pages.size() * c.page_size < new_len) s.pages.push_back(c.alloc.alloc());
}

// Prepares the batch inputs and enqueues the whole forward pass up to the logits [n, vocab] in b.d_logits.
// Records b.ev0 just before the first kernel. Returns n.
static int decode_step_gpu(Model& m, PagedKVCache& c, BatchState& b, std::vector<Sequence*>& seqs,
                           const std::vector<int>& tokens, int num_splits, StepTimes* times) {
  const int n = (int)seqs.size();
  if (n <= 0 || n > b.max_batch || n > m.max_T || (int)tokens.size() != n) {
    fprintf(stderr, "model_decode_batch: bad batch size %d (max %d)\n", n, b.max_batch);
    exit(1);
  }
  if (num_splits > 32) { fprintf(stderr, "num_splits %d exceeds the workspace limit of 32\n", num_splits); exit(1); }
  const int H = m.cfg.hidden;

  auto t0 = Clock::now();
  std::vector<int> tables((size_t)n * b.max_pages, 0), lens(n), pos(n);
  int max_len = 0;
  for (int i = 0; i < n; i++) {
    Sequence& s = *seqs[i];
    ensure_pages(c, s, s.len + 1);
    if ((int)s.pages.size() > b.max_pages) { fprintf(stderr, "sequence %d needs more pages than the batch table holds\n", i); exit(1); }
    std::copy(s.pages.begin(), s.pages.end(), tables.begin() + (size_t)i * b.max_pages);
    pos[i] = s.len;
    lens[i] = s.len + 1;
    max_len = std::max(max_len, lens[i]);
  }
  CUDA_CHECK(cudaMemcpy(b.d_tables, tables.data(), tables.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.d_lens, lens.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m.d_pos, pos.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(m.d_ids, tokens.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  const int splits = num_splits > 0 ? num_splits : choose_num_splits(n, m.cfg.n_kv_heads, max_len);
  auto t1 = Clock::now();
  if (times) times->prep_ms += ms_between(t0, t1);

  CUDA_CHECK(cudaEventRecord(b.ev0));
  embedding_lookup(m.embed, m.d_ids, m.d_x, n, H);
  for (int l = 0; l < m.n_layers; l++)
    layer_forward_batched(m.handle, m.cfg, m.layers[l], m.scratch, m.d_x, m.d_pos, m.d_inv_freq,
                          n, c.k_pool[l], c.v_pool[l], b, c.page_size, splits);
  rmsnorm(m.d_x, m.final_norm, b.d_normed, n, H, m.cfg.eps);                   // every row, not just the last
  linear(m.handle, b.d_normed, m.embed, b.d_logits, n, H, m.vocab);            // one read of the output matrix
  CUDA_CHECK(cudaGetLastError());
  return n;
}

void model_decode_batch(Model& m, PagedKVCache& c, BatchState& b, std::vector<Sequence*>& seqs,
                        const std::vector<int>& tokens, std::vector<float>& logits,
                        int num_splits, StepTimes* times) {
  const int n = decode_step_gpu(m, c, b, seqs, tokens, num_splits, times);
  CUDA_CHECK(cudaEventRecord(b.ev1));
  CUDA_CHECK(cudaEventSynchronize(b.ev1));
  float gpu_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, b.ev0, b.ev1));

  auto t2 = Clock::now();
  std::vector<__half> host((size_t)n * m.vocab);
  CUDA_CHECK(cudaMemcpy(host.data(), b.d_logits, host.size() * sizeof(__half), cudaMemcpyDeviceToHost));
  auto t3 = Clock::now();
  logits.resize(host.size());
  for (size_t i = 0; i < host.size(); i++) logits[i] = __half2float(host[i]);
  auto t4 = Clock::now();
  if (times) {
    times->gpu_ms += gpu_ms;
    times->copy_ms += ms_between(t2, t3);
    times->convert_ms += ms_between(t3, t4);
  }
  for (int i = 0; i < n; i++) seqs[i]->len += 1;
}

void model_decode_batch_greedy(Model& m, PagedKVCache& c, BatchState& b, std::vector<Sequence*>& seqs,
                               const std::vector<int>& tokens, std::vector<int>& next_tokens,
                               int num_splits, StepTimes* times) {
  const int n = decode_step_gpu(m, c, b, seqs, tokens, num_splits, times);
  argmax_rows(b.d_logits, b.d_next, n, m.vocab);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(b.ev1));
  CUDA_CHECK(cudaEventSynchronize(b.ev1));
  float gpu_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, b.ev0, b.ev1));

  auto t2 = Clock::now();
  next_tokens.resize(n);
  CUDA_CHECK(cudaMemcpy(next_tokens.data(), b.d_next, n * sizeof(int), cudaMemcpyDeviceToHost));
  auto t3 = Clock::now();
  if (times) {
    times->gpu_ms += gpu_ms;
    times->copy_ms += ms_between(t2, t3);
  }
  for (int i = 0; i < n; i++) seqs[i]->len += 1;
}
