#include <cstdint>
#include <cstring>
#include "util.h"
#include "residual_add.cuh"

static uint32_t rng_state = 12345u;
static float rnd() {   // deterministic pseudo-random float in [-4, 4)
  rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
  return (rng_state & 0xFFFFFF) / 16777216.0f * 8.0f - 4.0f;
}

int main() {
  const size_t n = 1000003;     // deliberately not a multiple of any block size
  const size_t pad = 256;       // extra bytes after the array, to catch out-of-bounds writes
  std::vector<__half> a(n), b(n), expected(n);
  for (size_t i = 0; i < n; i++) {
    a[i] = __float2half(rnd());
    b[i] = __float2half(rnd());
    expected[i] = __float2half_rn(__half2float(a[i]) + __half2float(b[i]));
  }
  __half *da, *db, *dout;
  CUDA_CHECK(cudaMalloc(&da, n * 2));
  CUDA_CHECK(cudaMalloc(&db, n * 2));
  CUDA_CHECK(cudaMalloc(&dout, (n + pad) * 2));
  CUDA_CHECK(cudaMemcpy(da, a.data(), n * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), n * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dout, 0xAB, (n + pad) * 2));

  int failures = 0;

  // 1) out-of-place, and the memory after the array must stay untouched
  residual_add(da, db, dout, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> got(n + pad);
  CUDA_CHECK(cudaMemcpy(got.data(), dout, (n + pad) * 2, cudaMemcpyDeviceToHost));
  bool ok1 = memcmp(got.data(), expected.data(), n * 2) == 0;
  bool ok2 = true;
  for (size_t i = n; i < n + pad; i++) {
    uint16_t u; memcpy(&u, &got[i], 2);
    if (u != 0xABAB) { ok2 = false; break; }
  }
  printf("out-of-place, exact match:            %s\n", ok1 ? "PASS" : "FAIL");
  printf("bounds check (padding untouched):     %s\n", ok2 ? "PASS" : "FAIL");
  failures += !ok1 + !ok2;

  // 2) in-place: out is the same array as a
  residual_add(da, db, da, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> got2(n);
  CUDA_CHECK(cudaMemcpy(got2.data(), da, n * 2, cudaMemcpyDeviceToHost));
  bool ok3 = memcmp(got2.data(), expected.data(), n * 2) == 0;
  printf("in-place (out == a), exact match:     %s\n", ok3 ? "PASS" : "FAIL");
  failures += !ok3;

  return failures ? 1 : 0;
}
