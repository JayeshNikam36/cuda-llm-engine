#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "util.h"
#include "model.h"

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";

  printf("loading model...\n");
  Model m = load_model(W, 402 + 32);
  FILE* out = fopen("/workspace/generated.txt", "w");

  int failures = 0;
  for (int i = 0; i < 8; i++) {
    char p[256];
    snprintf(p, sizeof(p), "%s/prompt_%02d_ids.i32", R.c_str(), i);
    auto ids = read_file<int32_t>(p);
    snprintf(p, sizeof(p), "%s/prompt_%02d_greedy.i32", R.c_str(), i);
    auto greedy = read_file<int32_t>(p);
    snprintf(p, sizeof(p), "%s/prompt_%02d_logits.f32", R.c_str(), i);
    auto ref_logits = read_file<float>(p);

    std::vector<int> tokens(ids.begin(), ids.end()), gen;
    std::vector<float> logits;
    const int n_new = (int)greedy.size();
    int lead = 0;            // how many leading tokens agree with Hugging Face
    bool diverged = false;
    double rel = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_new; s++) {
      model_last_logits(m, tokens, logits);
      if (s == 0) {
        double num = 0, den = 0;
        for (int v = 0; v < m.vocab; v++) {
          double d = (double)logits[v] - ref_logits[v];
          num += d * d; den += (double)ref_logits[v] * ref_logits[v];
        }
        rel = sqrt(num / den);
      }
      int best = 0;
      for (int v = 1; v < m.vocab; v++) if (logits[v] > logits[best]) best = v;
      gen.push_back(best);
      if (!diverged && best == greedy[s]) lead++; else diverged = true;
      tokens.push_back(best);
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    fprintf(out, "%d:", i);
    for (int t : gen) fprintf(out, " %d", t);
    fprintf(out, "\n");

    bool ok = lead == n_new && rel < 5e-3;
    printf("prompt %d: %3zu tokens  step-0 logits rel_L2 %.2e  generated %d/%d match  %.0f ms  %s\n",
           i, ids.size(), rel, lead, n_new, ms, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
  }
  fclose(out);
  free_model(m);
  return failures ? 1 : 0;
}
