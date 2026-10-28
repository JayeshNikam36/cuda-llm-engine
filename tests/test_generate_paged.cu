#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include "util.h"
#include "paged.h"

struct Run {
  int id = 0;
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

  printf("loading model...\n");
  Model m = load_model(W, 402 + 32);
  const int NUM_PAGES = 128, P = 16;
  PagedKVCache cache = alloc_paged_cache(m, NUM_PAGES, P);
  std::mt19937 rng(7);
  std::shuffle(cache.alloc.free_list.begin(), cache.alloc.free_list.end(), rng);   // physical order != logical order

  auto argmax = [&](const std::vector<float>& lg) {
    int best = 0;
    for (int v = 1; v < m.vocab; v++) if (lg[v] > lg[best]) best = v;
    return best;
  };

  int failures = 0;
  for (int pair = 0; pair < 4; pair++) {
    Run runs[2];
    std::vector<float> logits;
    for (int r = 0; r < 2; r++) {
      Run& u = runs[r];
      u.id = pair * 2 + r;
      char p[256];
      snprintf(p, sizeof(p), "%s/prompt_%02d_ids.i32", R.c_str(), u.id);
      auto ids = read_file<int32_t>(p);
      snprintf(p, sizeof(p), "%s/prompt_%02d_greedy.i32", R.c_str(), u.id);
      u.greedy = read_file<int32_t>(p);
      snprintf(p, sizeof(p), "%s/prompt_%02d_logits.f32", R.c_str(), u.id);
      auto ref_logits = read_file<float>(p);
      u.prompt.assign(ids.begin(), ids.end());

      u.seq = new_sequence(64);                          // room for 64 pages = 1024 tokens
      model_prefill_paged(m, cache, u.seq, u.prompt, logits);
      double num = 0, den = 0;
      for (int v = 0; v < m.vocab; v++) {
        double d = (double)logits[v] - ref_logits[v];
        num += d * d; den += (double)ref_logits[v] * ref_logits[v];
      }
      u.rel = sqrt(num / den);
      u.gen.push_back(argmax(logits));
    }

    // Decode the two sequences in alternation, so their pages end up interleaved in the pool.
    const int n_new = (int)runs[0].greedy.size();
    for (int s = 1; s < n_new; s++)
      for (int r = 0; r < 2; r++) {
        model_decode_paged(m, cache, runs[r].seq, runs[r].gen.back(), logits);
        runs[r].gen.push_back(argmax(logits));
      }

    for (int r = 0; r < 2; r++) {
      Run& u = runs[r];
      int lead = 0;
      while (lead < n_new && u.gen[lead] == u.greedy[lead]) lead++;
      bool ok = lead == n_new && u.rel < 5e-3;
      printf("prompt %d: %3zu tokens, %2zu pages [", u.id, u.prompt.size(), u.seq.pages.size());
      for (size_t i = 0; i < std::min<size_t>(5, u.seq.pages.size()); i++) printf("%s%d", i ? " " : "", u.seq.pages[i]);
      printf("%s]  step-0 rel_L2 %.2e  generated %d/%d match  %s\n",
             u.seq.pages.size() > 5 ? " ..." : "", u.rel, lead, n_new, ok ? "PASS" : "FAIL");
      if (!ok) failures++;
    }
    for (int r = 0; r < 2; r++) free_sequence(cache, runs[r].seq);
  }

  bool leak_ok = cache.alloc.num_free() == cache.alloc.total;
  printf("page leak check (all %d pages returned): %s\n", cache.alloc.total, leak_ok ? "PASS" : "FAIL");
  if (!leak_ok) failures++;
  free_paged_cache(cache);
  free_model(m);
  return failures ? 1 : 0;
}
