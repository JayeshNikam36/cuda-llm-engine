#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include "scheduler.h"
#include "util.h"

static int argmax_row(const float* lg, int vocab) {
  int best = 0;
  for (int v = 1; v < vocab; v++) if (lg[v] > lg[best]) best = v;
  return best;
}

SchedulerStats run_scheduler(Model& m, PagedKVCache& cache, BatchState& bs,
                             std::vector<Request>& reqs, int max_running) {
  if (max_running > bs.max_batch) { fprintf(stderr, "max_running %d exceeds the batch state size %d\n", max_running, bs.max_batch); exit(1); }
  struct Active { Request* r = nullptr; Sequence seq; int need_pages = 0; };
  std::vector<Active> active;
  std::deque<Request*> waiting;
  std::vector<float> logits;
  size_t next = 0, done = 0;
  int iter = 0;
  long batch_sum = 0;
  SchedulerStats st;
  const int P = cache.page_size;

  auto t0 = std::chrono::steady_clock::now();
  auto now_ms = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };

  auto retire_finished = [&] {
    for (size_t i = 0; i < active.size();) {
      if ((int)active[i].r->output.size() >= active[i].r->max_new) {
        active[i].r->t_done_ms = now_ms();
        free_sequence(cache, active[i].seq);          // pages go back to the pool immediately
        active.erase(active.begin() + i);
        done++;
      } else {
        i++;
      }
    }
  };

  while (done < reqs.size()) {
    // 1. arrivals
    while (next < reqs.size() && reqs[next].arrival_step <= iter) {
      reqs[next].t_arrive_ms = now_ms();
      waiting.push_back(&reqs[next]);
      next++;
    }

    // 2. admit at most one request per iteration (its prompt is processed whole)
    if (!waiting.empty() && (int)active.size() < max_running) {
      Request* r = waiting.front();
      const int need = ((int)r->prompt.size() + r->max_new + P - 1) / P;      // pages for the full length
      int promised = 0;                                                      // pages running requests may still take
      for (auto& a : active) promised += a.need_pages - (int)a.seq.pages.size();
      if (cache.alloc.num_free() - promised >= need) {
        waiting.pop_front();
        Active a;
        a.r = r;
        a.need_pages = need;
        a.seq = new_sequence(bs.max_pages);
        model_prefill_paged(m, cache, a.seq, r->prompt, logits);
        r->output.push_back(argmax_row(logits.data(), m.vocab));
        r->t_first_ms = now_ms();
        active.push_back(std::move(a));
      } else if (active.empty()) {
        fprintf(stderr, "request %d can never fit in the page pool\n", r->id);
        exit(1);
      }
    }
    retire_finished();                                                       // covers max_new == 1

    // 3. one decode step for everything running
    if (!active.empty()) {
      std::vector<Sequence*> ptrs;
      std::vector<int> tokens;
      for (auto& a : active) { ptrs.push_back(&a.seq); tokens.push_back(a.r->output.back()); }
      model_decode_batch(m, cache, bs, ptrs, tokens, logits);
      for (size_t i = 0; i < active.size(); i++)
        active[i].r->output.push_back(argmax_row(logits.data() + i * (size_t)m.vocab, m.vocab));
      st.decode_steps++;
      batch_sum += (long)active.size();
      retire_finished();                                                     // 4. finished requests leave now
    }
    iter++;
  }

  st.total_ms = now_ms();
  for (auto& r : reqs) st.total_words += (int)r.output.size();
  st.mean_batch = st.decode_steps ? (double)batch_sum / st.decode_steps : 0;
  return st;
}
