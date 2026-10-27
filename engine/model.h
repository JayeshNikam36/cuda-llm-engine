#pragma once
#include <string>
#include <vector>
#include <cublas_v2.h>
#include "layer.h"

struct Model {
  LayerConfig cfg;
  int n_layers = 16, vocab = 128256, max_T = 0;
  __half *embed = nullptr, *final_norm = nullptr;
  std::vector<LayerWeights> layers;
  float* d_inv_freq = nullptr;
  cublasHandle_t handle = nullptr;
  LayerScratch scratch;
  __half *d_x = nullptr, *d_normed = nullptr, *d_logits = nullptr;
  int *d_ids = nullptr, *d_pos = nullptr;
};

Model load_model(const std::string& dir, int max_T);
void free_model(Model& m);

// Runs the whole token sequence through the model and returns the logits of the LAST token.
void model_last_logits(Model& m, const std::vector<int>& ids, std::vector<float>& logits);
