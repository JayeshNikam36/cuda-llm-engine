#include <algorithm>
#include <cstdint>
#include <random>
#include "util.h"
#include "scheduler.h"

static double median_of(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double max_of(const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); }

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";
  const int NPROMPT = 8, NREQ = 24, NUM_PAGES = 256, P = 16, MAXRUN = 8, SPACING = 2;

  printf("loading model...\n");
  Model m = load_model(W, 402 + 32);
  PagedKVCache cache = alloc_paged_cache(m, NUM_PAGES, P);
  std::mt19937 rng(5);
  std::shuffle(cache.alloc.free_list.begin(), cache.alloc.free_list.end(), rng);
  BatchState bs = alloc_batch_state(m, MAXRUN, 64);

  std::vector<std::vector<int>> prompts(NPROMPT), expected(NPROMPT);
  for (int i = 0; i < NPROMPT; i++) {
    char p[256];
    snprintf(p, sizeof(p), "%s/prompt_%02d_ids.i32", R.c_str(), i);
    auto ids = read_file<int32_t>(p);
    prompts[i].assign(ids.begin(), ids.end());
    snprintf(p, sizeof(p), "%s/prompt_%02d_greedy.i32", R.c_str(), i);
    auto g = read_file<int32_t>(p);
    expected[i].assign(g.begin(), g.end());
  }
  std::vector<Request> workload(NREQ);
  for (int i = 0; i < NREQ; i++) {
    workload[i].id = i;
    workload[i].prompt = prompts[i % NPROMPT];
    workload[i].max_new = (int)expected[i % NPROMPT].size();
    workload[i].arrival_step = i * SPACING;
  }

  // One throwaway run so neither mode pays for cuBLAS and kernel warmup.
  { std::vector<Request> w(workload.begin(), workload.begin() + 2); run_scheduler(m, cache, bs, w, MAXRUN); }

  int failures = 0;
  for (int round = 1; round <= 2; round++) {
    double wps[2] = {0, 0};
    for (int mode = 0; mode < 2; mode++) {
      const int max_running = mode == 0 ? 1 : MAXRUN;
      std::vector<Request> reqs = workload;
      SchedulerStats st = run_scheduler(m, cache, bs, reqs, max_running);

      int correct = 0;
      std::vector<double> ttft, lat;
      for (auto& r : reqs) {
        if (r.output == expected[r.id % NPROMPT]) correct++;
        ttft.push_back(r.t_first_ms - r.t_arrive_ms);
        lat.push_back(r.t_done_ms - r.t_arrive_ms);
      }
      wps[mode] = st.total_words / (st.total_ms * 1e-3);
      printf("round %d, max %d at once: %d requests, %d decode steps, mean batch %.1f, total %.2f s, %.0f words/s\n"
             "          time to first word: median %.0f ms, max %.0f ms | latency: median %.0f ms, max %.0f ms | outputs correct: %d/%d  %s\n",
             round, max_running, NREQ, st.decode_steps, st.mean_batch, st.total_ms * 1e-3, wps[mode],
             median_of(ttft), max_of(ttft), median_of(lat), max_of(lat), correct, NREQ,
             correct == NREQ ? "PASS" : "FAIL");
      if (correct != NREQ) failures++;
    }
    printf("round %d: up to %d at once gives %.2fx the throughput of one at a time\n\n", round, MAXRUN, wps[1] / wps[0]);
  }

  bool leak_ok = cache.alloc.num_free() == cache.alloc.total;
  printf("page leak check (all %d pages returned): %s\n", cache.alloc.total, leak_ok ? "PASS" : "FAIL");
  if (!leak_ok) failures++;
  free_batch_state(bs);
  free_paged_cache(cache);
  free_model(m);
  return failures ? 1 : 0;
}
