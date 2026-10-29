#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "util.h"
#include "model.h"

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  Model m = load_model(W, 402 + 32);

  size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  const size_t margin = 256ull << 20;                       // headroom for cuBLAS workspaces and fragmentation
  const size_t kv_budget = free_b > margin ? free_b - margin : 0;
  const size_t bytes_per_token = 2ull * m.n_layers * m.cfg.n_kv_heads * m.cfg.head_dim * sizeof(__half);
  const int P = 16;
  const long total_tokens = (long)(kv_budget / bytes_per_token);
  const long pool_pages = (long)(kv_budget / (bytes_per_token * P));

  char buf[512];
  std::string md = "# KV cache capacity: paged vs contiguous reservation\n\n";
  snprintf(buf, sizeof(buf),
           "Measured after loading the model: %.2f GB free of %.2f GB total; KV budget %.2f GB (256 MB headroom).\n"
           "KV cost: %zu bytes per token (K and V, 16 layers). Budget holds %ld tokens = %ld pages of %d tokens.\n\n",
           free_b / 1e9, total_b / 1e9, kv_budget / 1e9, bytes_per_token, total_tokens, pool_pages, P);
  md += buf;
  md += "Contiguous = every request reserves max_seq tokens up front. Paged = pages taken as the sequence grows.\n"
        "The request lengths are simulated (seeded); only the memory numbers are measured.\n\n";
  md += "| workload (final length per request) | paged: requests | paged: memory used | contiguous (reserve 2048): requests | used | contiguous (reserve 4096): requests | used | paged vs 2048 |\n";
  md += "|---|---:|---:|---:|---:|---:|---:|---:|\n";

  struct Workload { const char* name; std::vector<int> lens; };
  std::mt19937 rng(42);
  const int N = 4000;
  Workload ws[3];
  ws[0].name = "fixed 512";
  ws[1].name = "uniform 128-1024";
  ws[2].name = "chat-like (median 300, tail to 2048)";
  std::uniform_int_distribution<int> U(128, 1024);
  std::lognormal_distribution<double> LN(std::log(300.0), 0.7);
  for (int i = 0; i < N; i++) {
    ws[0].lens.push_back(512);
    ws[1].lens.push_back(U(rng));
    ws[2].lens.push_back(std::min(2048, std::max(16, (int)LN(rng))));
  }

  for (auto& w : ws) {
    // Paged: admit requests in order until the page pool runs out.
    long pages_used = 0, tokens_paged = 0, n_paged = 0;
    for (int len : w.lens) {
      long need = (len + P - 1) / P;
      if (pages_used + need > pool_pages) break;
      pages_used += need; tokens_paged += len; n_paged++;
    }
    double used_paged = n_paged ? 100.0 * tokens_paged / (pages_used * (double)P) : 0;

    auto contiguous = [&](int reserve, long& count, double& util) {
      count = total_tokens / reserve;
      long toks = 0;
      for (long i = 0; i < count && i < (long)w.lens.size(); i++) toks += w.lens[i];
      util = count ? 100.0 * toks / ((double)count * reserve) : 0;
    };
    long c2 = 0, c4 = 0; double u2 = 0, u4 = 0;
    contiguous(2048, c2, u2);
    contiguous(4096, c4, u4);

    snprintf(buf, sizeof(buf), "| %s | %ld | %.0f%% | %ld | %.0f%% | %ld | %.0f%% | %.1fx |\n",
             w.name, n_paged, used_paged, c2, u2, c4, u4, c2 ? (double)n_paged / c2 : 0.0);
    md += buf;
  }
  md += "\n'used' = share of the reserved KV memory that holds real tokens. The rest is wasted: reserved but empty.\n";

  printf("%s", md.c_str());
  FILE* f = fopen("/workspace/docs/capacity.md", "w");
  if (f) { fputs(md.c_str(), f); fclose(f); }
  free_model(m);
  return 0;
}
