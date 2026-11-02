#pragma once
#include <vector>
#include "batched.h"

struct Request {
  int id = 0;
  std::vector<int> prompt;
  int max_new = 32;
  int arrival_step = 0;            // becomes eligible at this scheduler iteration
  // filled in by the scheduler
  std::vector<int> output;
  double t_arrive_ms = -1, t_first_ms = -1, t_done_ms = -1;
};

struct SchedulerStats {
  int decode_steps = 0;
  int total_words = 0;
  double total_ms = 0, mean_batch = 0;
  // time split, summed over the whole run (milliseconds)
  double prefill_ms = 0, prep_ms = 0, gpu_ms = 0, copy_ms = 0, convert_ms = 0, argmax_ms = 0;
};

// Runs every request to completion with continuous batching. Requests must be sorted by arrival_step.
// At most max_running sequences decode together (max_running <= bs.max_batch).
// gpu_argmax: choose each next word on the GPU instead of copying the logits to the CPU.
SchedulerStats run_scheduler(Model& m, PagedKVCache& cache, BatchState& bs,
                             std::vector<Request>& reqs, int max_running, bool gpu_argmax = false);
