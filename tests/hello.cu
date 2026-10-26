#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;
  std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n, 0.0f);
  float *da, *db, *dc;
  cudaMalloc(&da, n * sizeof(float));
  cudaMalloc(&db, n * sizeof(float));
  cudaMalloc(&dc, n * sizeof(float));
  cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    if (c[i] != 3.0f) { printf("FAIL at %d: got %f\n", i, c[i]); return 1; }
  }
  printf("PASS: all %d results correct\n", n);
  return 0;
}
