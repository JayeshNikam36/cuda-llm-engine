#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include "util.h"
#include "argmax.cuh"

int main() {
  const int V = 128256, R = 8;
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> U(-5.f, 5.f);
  std::uniform_int_distribution<int> small(-8, 8);
  std::vector<__half> h((size_t)R * V);
  for (int r = 0; r < R; r++)
    for (int i = 0; i < V; i++) {
      float v;
      if (r < 4) v = small(rng) * 0.5f;          // many exact ties
      else if (r == 4) v = 1.0f;                 // every value equal: the answer must be index 0
      else v = U(rng);
      h[(size_t)r * V + i] = __float2half(v);
    }
  h[(size_t)5 * V + (V - 1)] = __float2half(100.f);                       // maximum at the very last index
  h[(size_t)6 * V + 0] = __float2half(100.f);                             // two equal maxima: index 0 must win
  h[(size_t)6 * V + 77777] = __float2half(100.f);

  __half* d; int* dout;
  CUDA_CHECK(cudaMalloc(&d, h.size() * 2));
  CUDA_CHECK(cudaMalloc(&dout, R * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  argmax_rows(d, dout, R, V);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> got(R);
  CUDA_CHECK(cudaMemcpy(got.data(), dout, R * sizeof(int), cudaMemcpyDeviceToHost));

  int failures = 0;
  for (int r = 0; r < R; r++) {
    int best = 0;
    for (int i = 1; i < V; i++)
      if (__half2float(h[(size_t)r * V + i]) > __half2float(h[(size_t)r * V + best])) best = i;
    const bool ok = got[r] == best;
    printf("row %d: gpu %6d  cpu %6d  %s\n", r, got[r], best, ok ? "PASS" : "FAIL");
    failures += !ok;
  }
  return failures ? 1 : 0;
}
