#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include "util.h"

// Loads one tensor from weights.bin onto the GPU and returns the device pointer.
inline __half* load_weight(const std::string& dir, const std::string& name,
                           std::vector<long>* shape = nullptr) {
  TensorInfo ti = find_tensor(dir + "/index.tsv", name);
  auto host = read_file<uint16_t>(dir + "/weights.bin", ti.offset, ti.nbytes / 2);
  __half* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, ti.nbytes));
  CUDA_CHECK(cudaMemcpy(d, host.data(), ti.nbytes, cudaMemcpyHostToDevice));
  if (shape) *shape = ti.shape;
  return d;
}
