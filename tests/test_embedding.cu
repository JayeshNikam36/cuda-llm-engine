#include <cstring>
#include <cstdint>
#include "util.h"
#include "embedding.cuh"

int main() {
  const char* we = getenv("WEIGHTS_DIR");
  const char* re = getenv("REF_DIR");
  std::string W = we ? we : "/models/llama-3.2-1b-fp16";
  std::string R = re ? re : "/workspace/tests/reference/bin";

  TensorInfo ti = find_tensor(W + "/index.tsv", "model.embed_tokens.weight");
  int vocab = (int)ti.shape[0], hidden = (int)ti.shape[1];
  auto table = read_file<uint16_t>(W + "/weights.bin", ti.offset, ti.nbytes / 2);
  printf("embedding table: %d x %d\n", vocab, hidden);

  __half* d_table;
  CUDA_CHECK(cudaMalloc(&d_table, ti.nbytes));
  CUDA_CHECK(cudaMemcpy(d_table, table.data(), ti.nbytes, cudaMemcpyHostToDevice));

  int failures = 0;
  for (int i = 0; i < 8; i++) {
    char p[256];
    snprintf(p, sizeof(p), "%s/prompt_%02d_ids.i32", R.c_str(), i);
    auto ids = read_file<int32_t>(p);
    snprintf(p, sizeof(p), "%s/prompt_%02d_embed.f16", R.c_str(), i);
    auto expected = read_file<uint16_t>(p);
    int n = (int)ids.size();
    if (expected.size() != (size_t)n * hidden) { printf("prompt %d: SIZE MISMATCH\n", i); failures++; continue; }

    int* d_ids; __half* d_out;
    CUDA_CHECK(cudaMalloc(&d_ids, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * hidden * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    embedding_lookup(d_table, d_ids, d_out, n, hidden);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got((size_t)n * hidden);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    bool ok = memcmp(got.data(), expected.data(), got.size() * sizeof(uint16_t)) == 0;
    printf("prompt %d: %3d tokens  %s\n", i, n, ok ? "PASS (exact match)" : "FAIL");
    if (!ok) failures++;
    cudaFree(d_ids); cudaFree(d_out);
  }
  cudaFree(d_table);
  return failures ? 1 : 0;
}
