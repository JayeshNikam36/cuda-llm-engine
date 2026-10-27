#include <cstdio>
#include <cstdlib>
#include "layer.h"
#include "weights.h"
#include "rmsnorm.cuh"
#include "linear.cuh"
#include "rope.cuh"
#include "attention_naive.cuh"
#include "swiglu.cuh"
#include "residual_add.cuh"

LayerWeights load_layer_weights(const std::string& dir, int layer) {
  auto get = [&](const char* suffix) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    return load_weight(dir, name);
  };
  LayerWeights w;
  w.attn_norm = get("input_layernorm.weight");
  w.wq        = get("self_attn.q_proj.weight");
  w.wk        = get("self_attn.k_proj.weight");
  w.wv        = get("self_attn.v_proj.weight");
  w.wo        = get("self_attn.o_proj.weight");
  w.mlp_norm  = get("post_attention_layernorm.weight");
  w.w_gate    = get("mlp.gate_proj.weight");
  w.w_up      = get("mlp.up_proj.weight");
  w.w_down    = get("mlp.down_proj.weight");
  return w;
}

void free_layer_weights(LayerWeights& w) {
  cudaFree(w.attn_norm); cudaFree(w.wq); cudaFree(w.wk); cudaFree(w.wv); cudaFree(w.wo);
  cudaFree(w.mlp_norm); cudaFree(w.w_gate); cudaFree(w.w_up); cudaFree(w.w_down);
}

LayerScratch alloc_scratch(const LayerConfig& c, int max_T) {
  LayerScratch s;
  s.max_T = max_T;
  auto mk = [&](size_t per_token) {
    __half* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, (size_t)max_T * per_token * sizeof(__half)));
    return p;
  };
  s.norm = mk(c.hidden);
  s.q    = mk((size_t)c.n_heads * c.head_dim);
  s.k    = mk((size_t)c.n_kv_heads * c.head_dim);
  s.v    = mk((size_t)c.n_kv_heads * c.head_dim);
  s.attn = mk((size_t)c.n_heads * c.head_dim);
  s.proj = mk(c.hidden);
  s.gate = mk(c.inter);
  s.up   = mk(c.inter);
  s.act  = mk(c.inter);
  return s;
}

void free_scratch(LayerScratch& s) {
  cudaFree(s.norm); cudaFree(s.q); cudaFree(s.k); cudaFree(s.v); cudaFree(s.attn);
  cudaFree(s.proj); cudaFree(s.gate); cudaFree(s.up); cudaFree(s.act);
}

void layer_forward(cublasHandle_t handle, const LayerConfig& c, const LayerWeights& w,
                   LayerScratch& s, __half* x, const int* d_positions,
                   const float* d_inv_freq, int T) {
  if (T > s.max_T) { fprintf(stderr, "layer_forward: T=%d exceeds scratch size %d\n", T, s.max_T); exit(1); }
  const int qdim = c.n_heads * c.head_dim, kvdim = c.n_kv_heads * c.head_dim;
  const size_t n_hidden = (size_t)T * c.hidden;

  // ---- attention block ----
  rmsnorm(x, w.attn_norm, s.norm, T, c.hidden, c.eps);
  linear(handle, s.norm, w.wq, s.q, T, c.hidden, qdim);
  linear(handle, s.norm, w.wk, s.k, T, c.hidden, kvdim);
  linear(handle, s.norm, w.wv, s.v, T, c.hidden, kvdim);
  rope_apply(s.q, s.k, d_positions, d_inv_freq, T, c.n_heads, c.n_kv_heads, c.head_dim);
  attention_naive(s.q, s.k, s.v, s.attn, T, c.n_heads, c.n_kv_heads, c.head_dim);
  linear(handle, s.attn, w.wo, s.proj, T, qdim, c.hidden);
  residual_add(x, s.proj, x, n_hidden);                       // x = x + attention output

  // ---- MLP block ----
  rmsnorm(x, w.mlp_norm, s.norm, T, c.hidden, c.eps);
  linear(handle, s.norm, w.w_gate, s.gate, T, c.hidden, c.inter);
  linear(handle, s.norm, w.w_up,   s.up,   T, c.hidden, c.inter);
  swiglu(s.gate, s.up, s.act, (size_t)T * c.inter);
  linear(handle, s.act, w.w_down, s.proj, T, c.inter, c.hidden);
  residual_add(x, s.proj, x, n_hidden);                       // x = x + MLP output
}
