#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Writes T rows (each kvdim halves, kvdim a multiple of 8) from src [T, kvdim] into a paged pool.
// Row t goes to position start_pos + t: page block_table[pos / page_size], slot pos % page_size.
void kv_scatter(const __half* src, __half* pool, const int* block_table,
                int start_pos, int T, int page_size, int kvdim, cudaStream_t stream = 0);
