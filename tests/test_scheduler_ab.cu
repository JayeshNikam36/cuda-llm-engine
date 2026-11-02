#include <algorithm>
#include <cstdint>
#include <random>
#include "util.h"
#include "scheduler.h"

static double median_of(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";
  const int NPROMPT = 8, NREQ = 24, NUM_PAGES = 256, P = 16, MAXRUN = 8, ROUNDS = 5;

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
  for (int i = 0; i < NREQ; i++) {                 // all requests available at once, like the vLLM comparison
    workload[i].id = i;
    workload[i].prompt = prompts[i % NPROMPT];
    workload[i].max_new = (int)expected[i % NPROMPT].size();
    workload[i].arrival_step = 0;
  }

  // Warm-up for both paths, so no configuration pays for first-use costs.
  for (int gpu = 0; gpu < 2; gpu++) {
    std::vector<Request> w(workload.begin(), workload.begin() + 2);
    run_scheduler(m, cache, bs, w, MAXRUN, gpu == 1);
  }

  struct Cfg { const char* name; int max_running; bool gpu; };
  const Cfg cfgs[4] = {{"batched, CPU argmax", MAXRUN, false}, {"batched, GPU argmax", MAXRUN, true},
                       {"one at a time, CPU argmax", 1, false}, {"one at a time, GPU argmax", 1, true}};
  std::vector<double> wps[4];
  int failures = 0;
  for (int round = 1; round <= ROUNDS; round++) {
    for (int k = 0; k < 4; k++) {
      const int c = (round % 2 == 0) ? 3 - k : k;   // alternate the order every round
      std::vector<Request> reqs = workload;
      SchedulerStats st = run_scheduler(m, cache, bs, reqs, cfgs[c].max_running, cfgs[c].gpu);
      int correct = 0;
      for (auto& r : reqs) if (r.output == expected[r.id % NPROMPT]) correct++;
      const double w = st.total_words / (st.total_ms * 1e-3);
      if (round > 1) wps[c].push_back(w);          // round 1 is warm-up and is not counted
      const double s = st.decode_steps > 0 ? st.decode_steps : 1;
      printf("round %d | %-26s | %4.0f words/s | %3d steps | prefill %4.0f ms total | per step (ms): prep %.2f, GPU %.2f, copy %.2f, convert %.2f, argmax %.2f | correct %d/%d %s\n",
             round, cfgs[c].name, w, st.decode_steps, st.prefill_ms,
             st.prep_ms / s, st.gpu_ms / s, st.copy_ms / s, st.convert_ms / s, st.argmax_ms / s,
             correct, NREQ, correct == NREQ ? "PASS" : "FAIL");
      if (correct != NREQ) failures++;
    }
  }

  printf("\nmedian over rounds 2 to %d (round 1 is warm-up):\n", ROUNDS);
  double med[4];
  for (int c = 0; c < 4; c++) { med[c] = median_of(wps[c]); printf("  %-26s %4.0f words/s\n", cfgs[c].name, med[c]); }
  printf("GPU argmax vs CPU argmax: batched %.2fx, one at a time %.2fx\n", med[1] / med[0], med[3] / med[2]);

  bool leak_ok = cache.alloc.num_free() == cache.alloc.total;
  printf("page leak check (all %d pages returned): %s\n", cache.alloc.total, leak_ok ? "PASS" : "FAIL");
  if (!leak_ok) failures++;
  free_batch_state(bs);
  free_paged_cache(cache);
  free_model(m);
  return failures ? 1 : 0;
}
