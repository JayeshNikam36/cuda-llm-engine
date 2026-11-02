# An LLM inference engine from scratch in C++/CUDA

## Summary

An inference engine for Llama-3.2-1B (FP16) written from scratch: the kernels, a paged KV cache, batched decoding and a continuous-batching scheduler. No PyTorch, vLLM or TensorRT-LLM at runtime. Generated words match Hugging Face greedy decoding token for token on 8 reference prompts (6 to 402 tokens) in every configuration tested. Everything was built and measured on a laptop RTX 4050 (6 GB, Ada sm_89) under WSL2 and Docker, and compared with vLLM 0.32.0 on the same GPU.

## Results

| what | result |
|---|---|
| correctness | 32/32 words match Hugging Face on all 8 reference prompts: single sequence, shuffled KV pages, a batch of 8 mixed lengths, and a 24-request scheduler run |
| single-token decode pass | 14.06 ms (71 words/s), equal to the weight-read floor (2.47 GB at about 175 GB/s measured) |
| decode attention kernel | 12x faster than the naive paged kernel at 2048 context, 16 sequences (29.1 vs 346 us per sequence per layer); 82% of the measured ceiling in the machine's normal state, 57 to 72% during slow episodes |
| split-KV at small batch | batch 1: 41.0 us with 8 splits vs 82.9 us with 1 split (2.0x); no benefit at batch 16 |
| KV cache capacity | paged fits 134 to 197 concurrent requests vs 37 with contiguous 2048-token reservation (3.6 to 5.3x; about 2.5 GB KV budget, simulated lengths) |
| continuous batching | 244 to 249 words/s vs 60 one at a time (4.0 to 4.1x), 24 staggered requests |
| decode step at batch 8 | 15.0 to 16.5 ms with GPU argmax, 7 to 17% above the weight-read floor |
| GPU-side argmax | single stream 67 words/s in 5 of 5 rounds vs 60 to 66 with CPU argmax; each batched step 2.8 to 3.3 ms faster |
| vs vLLM 0.32.0, one at a time | 67 vs 67 to 68 words/s (level) |
| vs vLLM 0.32.0, 24 requests, 8 at once | vLLM median 533 vs 235 to 361 words/s for this engine (about 1.5 to 2.3x faster); vLLM measured in a single session |

## Design

- kernels/: embedding lookup, RMSNorm, RoPE (with the Llama-3 frequency scaling), SwiGLU, residual add, naive and fast paged decode attention, KV scatter, GPU argmax. Large matrix multiplies use cuBLAS.
- engine/: weight loading, the model and one transformer layer, contiguous and paged KV caches, page allocator, batched decode, scheduler.
- Model files are converted once to FP16 and read by a plain C++ loader; tokenization stays outside the engine.

### Paged KV cache

K and V live in pools of 16-token pages. Each sequence owns a list of pages (its block table); position p is on page table[p / 16], slot p % 16. A free-list allocator hands pages out as sequences grow and takes them back when they finish. The paged attention kernel replaces "row j" with a block-table lookup. Tests shuffle the page order and fill unused slots with NaN, so any wrong address poisons the output.

### Fast decode attention (split-KV, grouped-query)

- One thread block serves one KV head and the 4 query heads that share it, so each key and value is loaded once and used four times.
- Eight threads cooperate on a position, each loading 16 bytes (128-bit loads); the partial dot products are added with warp shuffles.
- Online softmax keeps a running maximum, sum and weighted value total, so no scores array is needed and there is no limit on context length.
- The positions of a sequence are split across blocks, which keeps the GPU busy at small batch sizes. A second tiny kernel merges the partial results by rescaling each with exp(m_split - M).
- The best number of splits depends on batch size; a heuristic of about 64 blocks in the grid fit the measured sweep.

### Scheduler

Each iteration admits at most one waiting request (reserving pages for its full length, so a running request can never run out of pages), runs one batched decode step for everything running, and retires finished requests immediately, returning their pages.

## How correctness was checked

Reference outputs come from Hugging Face on fixed prompts. Each kernel is tested against PyTorch on real layer-0 data, then one layer, then the whole 16-layer model must generate identical greedy words. The fast decode kernel is tested against the naive paged kernel at lengths 1 to 2048 with 1 to 32 splits. The scheduler test requires every request to match its reference while requests join and leave the batch, and checks that no pages leak.

A bug worth remembering: the RoPE rotation speeds computed in 64-bit math differed from Hugging Face's 32-bit values by about 2e-7. Invisible at short positions, but at position 400 it flipped fp16 rounding for about 0.2% of values and failed the 402-token test. Matching the 32-bit computation fixed it.

## Measurement method, and what went wrong

The GPU is a laptop part that sheds clocks under sustained load (sampled throttle code 0x24, SM clock down to 1110 MHz in 9 of 29 busy samples even when plugged in). The same code measured 29 to 42 us in different periods. Two claims were made and retracted: an unroll experiment first looked like a 12% regression, but reverting to identical code gave the same slow number; and an "argmax speeds up batching by 1.18x" ratio was an artifact of run order. Practices adopted: flush L2 before timed runs, compare variants in the same process, alternate their order, sample clocks and power during runs, report ranges and ratios, and keep every result and retraction in docs/optimization_log.md.

## Where the remaining gap to vLLM is

Batched decode steps are within about 7 to 17% of the weight-read floor. Prefill is the largest remaining cost: each prompt is its own forward pass over all the weights, and the 402-token prompts use the slow naive attention kernel. Prefill was about 30 to 40% of the batched run time and varied from 0.6 to 1.3 s for identical work, which fits an SM-clock limit on a compute-bound phase (not proven). vLLM's own prefill share was not measured.

## Limitations and next steps

Greedy decoding only; no end-of-text handling; FP16 only; the fast attention kernel is specialised to head dimension 64, 16-token pages and 4 query heads per KV head; single GPU; results from one laptop GPU; vLLM compared in one session.
Next: batched and chunked prefill with a variable-length attention kernel, CUDA Graphs for the decode step (about 1 ms per step of headroom), FP8 weights, and an OpenAI-compatible server.
