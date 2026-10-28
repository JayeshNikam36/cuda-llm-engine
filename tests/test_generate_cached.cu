#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "util.h"
#include "cached.h"

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";

  printf("loading model...\n");
  Model m = load_model(W, 402 + 32);
  KVCache cache = alloc_kv_cache(m, 402 + 32);

  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ms_since = [](std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };

  // One throwaway run so the first timed prompt is not charged for cuBLAS and kernel warmup.
  {
    std::vector<float> tmp;
    model_prefill(m, cache, {1, 2, 3, 4}, tmp);
    model_decode(m, cache, 5, 4, tmp);
  }

  int failures = 0;
  for (int i = 0; i < 8; i++) {
    char p[256];
    snprintf(p, sizeof(p), "%s/prompt_%02d_ids.i32", R.c_str(), i);
    auto ids = read_file<int32_t>(p);
    snprintf(p, sizeof(p), "%s/prompt_%02d_greedy.i32", R.c_str(), i);
    auto greedy = read_file<int32_t>(p);
    snprintf(p, sizeof(p), "%s/prompt_%02d_logits.f32", R.c_str(), i);
    auto ref_logits = read_file<float>(p);

    std::vector<int> prompt(ids.begin(), ids.end()), gen;
    std::vector<float> logits;
    const int n_new = (int)greedy.size();
    const int T = (int)prompt.size();

    auto argmax = [&] {
      int best = 0;
      for (int v = 1; v < m.vocab; v++) if (logits[v] > logits[best]) best = v;
      return best;
    };

    auto t0 = now();
    model_prefill(m, cache, prompt, logits);
    double prefill_ms = ms_since(t0);

    double num = 0, den = 0;
    for (int v = 0; v < m.vocab; v++) {
      double d = (double)logits[v] - ref_logits[v];
      num += d * d; den += (double)ref_logits[v] * ref_logits[v];
    }
    double rel = sqrt(num / den);
    gen.push_back(argmax());

    auto t1 = now();
    for (int s = 1; s < n_new; s++) {
      model_decode(m, cache, gen.back(), T + s - 1, logits);
      gen.push_back(argmax());
    }
    double decode_ms = ms_since(t1) / (n_new - 1);

    int lead = 0;
    while (lead < n_new && gen[lead] == greedy[lead]) lead++;
    bool ok = lead == n_new && rel < 5e-3;
    printf("prompt %d: %3d tokens  step-0 rel_L2 %.2e  generated %d/%d match  prefill %6.1f ms  decode %5.1f ms/word  %s\n",
           i, T, rel, lead, n_new, prefill_ms, decode_ms, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
  }
  free_kv_cache(cache);
  free_model(m);
  return failures ? 1 : 0;
}
