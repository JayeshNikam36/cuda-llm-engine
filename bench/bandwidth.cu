#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  exit(1); } } while (0)

__global__ void copy_vec4(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] = src[i];
}

int main() {
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  int memClockKHz = 0, busWidth = 0;
  CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0));
  CHECK(cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, 0));
  double specGBps = 2.0 * memClockKHz * 1e3 * (busWidth / 8.0) / 1e9;
  printf("GPU: %s | SMs: %d | L2: %d MB | bus: %d-bit | spec peak: %.0f GB/s\n",
         prop.name, prop.multiProcessorCount, prop.l2CacheSize / (1024 * 1024), busWidth, specGBps);

  const size_t bytes = 512ull << 20;   // 512 MiB per buffer, far bigger than the L2 cache
  const size_t n = bytes / sizeof(float4);
  float4 *src, *dst;
  CHECK(cudaMalloc(&src, bytes));
  CHECK(cudaMalloc(&dst, bytes));
  CHECK(cudaMemset(src, 1, bytes));

  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));

  double best = 0; int bestBlock = 0, bestMult = 0;
  int blocks[] = {128, 256, 512};
  int mults[] = {2, 4, 8, 16};
  printf("\nblock  grid_mult   median_ms   GB/s\n");
  for (int block : blocks) for (int mult : mults) {
    int grid = prop.multiProcessorCount * mult;
    for (int i = 0; i < 5; i++) copy_vec4<<<grid, block>>>(src, dst, n);   // warmup
    CHECK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < 30; i++) {
      CHECK(cudaEventRecord(t0));
      copy_vec4<<<grid, block>>>(src, dst, n);
      CHECK(cudaEventRecord(t1));
      CHECK(cudaEventSynchronize(t1));
      float m; CHECK(cudaEventElapsedTime(&m, t0, t1));
      ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    float med = ms[ms.size() / 2];
    double gbps = (2.0 * bytes) / (med * 1e-3) / 1e9;   // read + write
    printf("%5d  %9d  %10.3f  %7.1f\n", block, mult, med, gbps);
    if (gbps > best) { best = gbps; bestBlock = block; bestMult = mult; }
  }
  printf("\nBEST: %.1f GB/s (block %d, grid_mult %d) = %.1f%% of spec peak\n",
         best, bestBlock, bestMult, 100.0 * best / specGBps);
  return 0;
}
