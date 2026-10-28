#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>

// Hands out page numbers from a fixed pool. Page numbers index into the K/V pools on the GPU.
struct PageAllocator {
  std::vector<int> free_list;
  int total = 0;

  explicit PageAllocator(int n) : total(n) {
    for (int i = n - 1; i >= 0; i--) free_list.push_back(i);   // page 0 is handed out first
  }
  int num_free() const { return (int)free_list.size(); }
  bool can_alloc(int n) const { return num_free() >= n; }
  int alloc() {
    if (free_list.empty()) { fprintf(stderr, "PageAllocator: out of pages\n"); exit(1); }
    int p = free_list.back();
    free_list.pop_back();
    return p;
  }
  void release(int p) { free_list.push_back(p); }
};
