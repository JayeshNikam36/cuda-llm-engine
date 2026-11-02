#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// out[r] = index of the largest value in row r of logits [n_rows, vocab].
// The lowest index wins ties, exactly like a CPU scan that only replaces the best on a strictly greater value.
void argmax_rows(const __half* logits, int* out, int n_rows, int vocab, cudaStream_t stream = 0);
