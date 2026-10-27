#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
  exit(1); } } while (0)

// Read `count` items of type T starting at `offset_bytes` (count == 0 means "to the end").
template <typename T>
std::vector<T> read_file(const std::string& path, size_t offset_bytes = 0, size_t count = 0) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  if (count == 0) {
    fseeko(f, 0, SEEK_END);
    count = (size_t)(ftello(f) - (off_t)offset_bytes) / sizeof(T);
  }
  fseeko(f, (off_t)offset_bytes, SEEK_SET);
  std::vector<T> v(count);
  size_t got = fread(v.data(), sizeof(T), count, f);
  fclose(f);
  if (got != count) { fprintf(stderr, "short read on %s\n", path.c_str()); exit(1); }
  return v;
}

struct TensorInfo { size_t offset = 0, nbytes = 0; std::vector<long> shape; };

// Looks up a tensor by name in index.tsv (name, offset, nbytes, shape).
inline TensorInfo find_tensor(const std::string& tsv_path, const std::string& name) {
  FILE* f = fopen(tsv_path.c_str(), "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", tsv_path.c_str()); exit(1); }
  char line[1024];
  while (fgets(line, sizeof(line), f)) {
    char nm[512], shp[256];
    unsigned long long off, nb;
    if (sscanf(line, "%511[^\t]\t%llu\t%llu\t%255s", nm, &off, &nb, shp) == 4 && name == nm) {
      TensorInfo ti; ti.offset = off; ti.nbytes = nb;
      char* p = shp;
      while (*p) { ti.shape.push_back(strtol(p, &p, 10)); if (*p == ',') p++; }
      fclose(f);
      return ti;
    }
  }
  fclose(f);
  fprintf(stderr, "tensor %s not found\n", name.c_str());
  exit(1);
}
