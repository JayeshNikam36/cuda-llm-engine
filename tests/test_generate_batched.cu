#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include "util.h"
#include "batched.h"

struct Run {
  Sequence seq;
  std::vector<int> prompt, gen;
  std::vector<int32_t> greedy;
  double rel = 0;
};

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";
  const char* sp = getenv("SPLITS");
  const int forced = sp ? atoi(sp) : 0;

  const int N = 8, NUM_PAGES = 128, P = 16;
  printf("loading model...\n");
  Model m = load_model(W, 402 + 32);
  PagedKVCache cache = alloc_paged_cache(m, NUM_PAGES, P);
  std::mt19937 rng(11);
  std::shuffle(cache.alloc.free_list.begin(), cache.alloc.free_list.end(), rng);
  BatchState bs = alloc_batch_state(m, N, 64);

  auto argmax = [&](const float* lg) {
    int best = 0;
    for (int v = 1; v < m.vocab; v++) if (lg[v] > lg[best]) best = v;
    return best;
  };

  std::vector<Run> runs(N);
  std::vector<float> logits;
  for (int i = 0; i < N; i++) {
    char p[256];
    snprintf(p, sizeof(p), "%s/prompt_%02d_ids.i32", R.c_str(), i);
    auto ids = read_file<int32_t>(p);
    snprintf(p, sizeof(p), "%s/prompt_%02d_greedy.i32", R.c_str(), i);
    runs[i].greedy = read_file<int32_t>(p);
    snprintf(p, sizeof(p), "%s/prompt_%02d_logits.f32", R.c_str(), i);
    auto ref_logits = read_file<float>(p);
    runs[i].prompt.assign(ids.begin(), ids.end());

    runs[i].seq = new_sequence(64);
    model_prefill_paged(m, cache, runs[i].seq, runs[i].prompt, logits);
    double num = 0, den = 0;
    for (int v = 0; v < m.vocab; v++) {
      double d = (double)logits[v] - ref_logits[v];
      num += d * d; den += (double)ref_logits[v] * ref_logits[v];
    }
    runs[i].rel = sqrt(num / den);
    runs[i].gen.push_back(argmax(logits.data()));
  }

  std::vector<Sequence*> ptrs;
  for (auto& r : runs) ptrs.push_back(&r.seq);
  std::vector<int> tokens(N);
  const int n_new = (int)runs[0].greedy.size();

  auto t0 = std::chrono::steady_clock::now();
  for (int s = 1; s < n_new; s++) {
    for (int i = 0; i < N; i++) tokens[i] = runs[i].gen.back();
    model_decode_batch(m, cache, bs, ptrs, tokens, logits, forced);
    for (int i = 0; i < N; i++) runs[i].gen.push_back(argmax(logits.data() + (size_t)i * m.vocab));
  }
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  int failures = 0;
  for (int i = 0; i < N; i++) {
    int lead = 0;
    while (lead < n_new && runs[i].gen[lead] == runs[i].greedy[lead]) lead++;
    bool ok = lead == n_new && runs[i].rel < 5e-3;
    printf("prompt %d: %3zu tokens  step-0 rel_L2 %.2e  generated %d/%d match  %s\n",
           i, runs[i].prompt.size(), runs[i].rel, lead, n_new, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
  }
  printf("batched decode: %d steps x %d sequences: %.1f ms per step, %.0f words/s aggregate (%s)\n",
         n_new - 1, N, ms / (n_new - 1), N * (n_new - 1) / (ms * 1e-3),
         forced ? "forced splits" : "auto splits");

  for (auto& r : runs) free_sequence(cache, r.seq);
  bool leak_ok = cache.alloc.num_free() == cache.alloc.total;
  printf("page leak check (all %d pages returned): %s\n", cache.alloc.total, leak_ok ? "PASS" : "FAIL");
  if (!leak_ok) failures++;
  free_batch_state(bs);
  free_paged_cache(cache);
  free_model(m);
  return failures ? 1 : 0;
}
