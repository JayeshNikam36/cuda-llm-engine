#pragma once
#include <string>
#include <cublas_v2.h>
#include <cuda_fp16.h>

struct LayerConfig {
  int hidden = 2048, n_heads = 32, n_kv_heads = 8, head_dim = 64, inter = 8192;
  float eps = 1e-5f;
};

struct LayerWeights {
  __half *attn_norm, *wq, *wk, *wv, *wo, *mlp_norm, *w_gate, *w_up, *w_down;
};

LayerWeights load_layer_weights(const std::string& dir, int layer);
void free_layer_weights(LayerWeights& w);

// Temporary buffers reused by every layer (sized for up to max_T tokens).
struct LayerScratch {
  __half *norm, *q, *k, *v, *attn, *proj, *gate, *up, *act;
  int max_T = 0;
};
LayerScratch alloc_scratch(const LayerConfig& c, int max_T);
void free_scratch(LayerScratch& s);

// x: [T, hidden], read and updated in place (x = layer(x)).
void layer_forward(cublasHandle_t handle, const LayerConfig& c, const LayerWeights& w,
                   LayerScratch& s, __half* x, const int* d_positions,
                   const float* d_inv_freq, int T);
