#pragma once
#include <vector>
#include "paged.h"

// Optional timing of one decode step (all in milliseconds, accumulated with +=).
struct StepTimes {
  double prep_ms = 0;      // host-side batch preparation and small uploads
  double gpu_ms = 0;       // GPU time from the first kernel to the last (events)
  double copy_ms = 0;      // copying results back to the host
  double convert_ms = 0;   // half to float conversion of the logits on the CPU (logits path only)
};

// Scratch for batched decode: block tables, sequence lengths, attention workspace, and per-row logits.
struct BatchState {
  int max_batch = 0, max_pages = 0;
  int *d_tables = nullptr, *d_lens = nullptr, *d_next = nullptr;
  float* d_ws = nullptr;
  __half *d_normed = nullptr, *d_logits = nullptr;
  cudaEvent_t ev0 = nullptr, ev1 = nullptr;
};

BatchState alloc_batch_state(const Model& m, int max_batch, int max_pages);
void free_batch_state(BatchState& b);

// One decode step for n sequences, one new token each (tokens[i] is fed to seqs[i] at position seqs[i]->len).
// Returns n * vocab logits on the host (row i for sequence i) and advances every sequence by one position.
// num_splits <= 0 picks the split count automatically.
void model_decode_batch(Model& m, PagedKVCache& c, BatchState& b, std::vector<Sequence*>& seqs,
                        const std::vector<int>& tokens, std::vector<float>& logits,
                        int num_splits = 0, StepTimes* times = nullptr);

// Same step, but the greedy choice (argmax) is made on the GPU and only n integers come back.
void model_decode_batch_greedy(Model& m, PagedKVCache& c, BatchState& b, std::vector<Sequence*>& seqs,
                               const std::vector<int>& tokens, std::vector<int>& next_tokens,
                               int num_splits = 0, StepTimes* times = nullptr);
