#pragma once
#include <vector>
#include "model.h"
#include "paging.h"

// K and V live in pools of fixed-size pages, one pool per layer:
// [num_pages, page_size, n_kv_heads, head_dim]. The same page number is used in every layer.
struct PagedKVCache {
  std::vector<__half*> k_pool, v_pool;
  PageAllocator alloc;
  int page_size;
  PagedKVCache(int num_pages, int page_size_) : alloc(num_pages), page_size(page_size_) {}
};

// One request: the pages it owns (in order), its length, and its block table on the GPU.
struct Sequence {
  std::vector<int> pages;
  int len = 0;
  int* d_table = nullptr;
  int max_pages = 0;
};

PagedKVCache alloc_paged_cache(const Model& m, int num_pages, int page_size);
void free_paged_cache(PagedKVCache& c);

Sequence new_sequence(int max_pages);
void free_sequence(PagedKVCache& c, Sequence& s);   // returns its pages to the allocator

// Processes the whole prompt (sequence must be empty), stores K/V in pages, returns last-token logits.
void model_prefill_paged(Model& m, PagedKVCache& c, Sequence& seq, const std::vector<int>& ids,
                         std::vector<float>& logits);

// Processes ONE new token at position seq.len, appends its K/V, returns logits for the next token.
void model_decode_paged(Model& m, PagedKVCache& c, Sequence& seq, int token, std::vector<float>& logits);
