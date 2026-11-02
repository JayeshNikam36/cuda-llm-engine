#pragma once
#include <vector>
#include "paged.h"

// Scratch for batched decode: block tables, sequence lengths, attention workspace, and per-row logits.
struct BatchState {
  int max_batch = 0, max_pages = 0;
  int *d_tables = nullptr, *d_lens = nullptr;
  float* d_ws = nullptr;
  __half *d_normed = nullptr, *d_logits = nullptr;
};

BatchState alloc_batch_state(const Model& m, int max_batch, int max_pages);
void free_batch_state(BatchState& b);

// One decode step for n sequences, one new token each (tokens[i] is fed to seqs[i] at position seqs[i]->len).
// Returns n * vocab logits, row i for sequence i, and advances every sequence by one position.
// num_splits <= 0 picks the split count automatically.
void model_decode_batch(Model& m, PagedKVCache& c, BatchState& b, std::vector<Sequence*>& seqs,
                        const std::vector<int>& tokens, std::vector<float>& logits, int num_splits = 0);
