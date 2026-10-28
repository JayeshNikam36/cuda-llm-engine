#pragma once
#include <vector>
#include "model.h"

struct KVCache {
  std::vector<__half*> k, v;   // one [max_seq, n_kv_heads, head_dim] buffer per layer, for K and for V
  int max_seq = 0;
};

KVCache alloc_kv_cache(const Model& m, int max_seq);
void free_kv_cache(KVCache& c);

// Processes the whole prompt, fills cache positions [0, T), returns the last token's logits.
void model_prefill(Model& m, KVCache& c, const std::vector<int>& ids, std::vector<float>& logits);

// Processes ONE new token at position `pos` (the cache already holds [0, pos)), appends its K and V,
// and returns the logits for the next token.
void model_decode(Model& m, KVCache& c, int token, int pos, std::vector<float>& logits);
