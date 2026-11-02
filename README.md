# cuda-llm-engine

An LLM inference engine written from scratch in C++/CUDA, without PyTorch, vLLM or TensorRT-LLM at runtime. It runs Llama-3.2-1B in FP16 with a paged KV cache, a split-KV grouped-query decode attention kernel, batched decoding and a continuous-batching scheduler. Each piece is tested against Hugging Face outputs.

Status: working engine, compared against vLLM 0.32.0 on the same GPU (see Results). Not done yet: FP8, an OpenAI-compatible server, CUDA Graphs, chunked prefill.

Development hardware: RTX 4050 Laptop GPU (6 GB, Ada sm_89, about 175 GB/s measured memory bandwidth), WSL2, Docker with CUDA 12.4.

## Results

| what | result | notes |
|---|---|---|
| correctness | 32/32 words match Hugging Face greedy output on all 8 reference prompts (6 to 402 tokens) | checked single-sequence, with shuffled KV pages, as a batch of 8 mixed lengths, and in a 24-request scheduler run |
| single-token decode pass | 14.06 ms, about 71 words/s | at the memory-bandwidth ceiling for 2.47 GB of FP16 weights |
| decode attention kernel | 12x faster than the naive paged kernel at 2048 context, 16 sequences (29.1 us vs 346 us per sequence per layer) | 82% of the 175 GB/s ceiling in the machine's normal state; 57 to 72% seen during slow episodes |
| KV cache capacity | paged fits 134 to 197 concurrent requests vs 37 with contiguous 2048-token reservation (3.6 to 5.3x) | about 2.5 GB KV budget; request lengths simulated |
| continuous batching | 244 to 249 words/s vs 60 one at a time (4.0 to 4.1x) | 24 requests, same process, interleaved runs, all outputs correct |
| GPU-side argmax | one request at a time 64 to 68 words/s (about +5%), batched unchanged at 291 words/s | per-step host overhead at batch 8 fell from about 3 ms to about 0.3 ms; the batched total did not change because prefill time rose by about the same amount in the same runs (cause not established); details in docs/step_time_split_results.txt |
| vs vLLM 0.32.0 | vLLM is about 1.9 to 2.1x faster with up to 8 requests at once (median 533 vs 255 to 288 words/s) and about 10 to 17% faster one request at a time (67 to 68 vs 58 to 61 words/s) | same GPU, same 24 prompts available at once, greedy, 32 new words each; both engines match Hugging Face on every request; vLLM run with its FlashInfer sampler disabled because WSL has no nvcc; details in docs/comparison_vllm.md |

Run-to-run variation on this laptop GPU is large (the same attention benchmark measured 29 to 42 us at different times), so ratios and ranges are more trustworthy than single absolute numbers. See docs/optimization_log.md for every experiment, including the ones that did not pan out.

## What is in the repo

- kernels/: embedding lookup, RMSNorm, RoPE with Llama-3 frequency scaling, SwiGLU, residual add, naive causal attention, naive and fast paged decode attention, KV scatter, cuBLAS linear wrapper
- engine/: weight loading, one transformer layer, the model, contiguous and paged KV caches, page allocator, batched decode, the scheduler
- tests/: one test per kernel, per layer and per engine stage, each compared with Hugging Face reference outputs
- bench/: bandwidth baseline, forward-pass and per-stage timing, KV capacity, decode attention comparison
- docs/: measured results and the optimization log

## How correctness is checked

Reference outputs come from Hugging Face on 8 fixed prompts. Each kernel is tested against PyTorch on real layer-0 data, then one layer, then the full 16-layer model must generate identical greedy words. Paging is tested with shuffled page order and NaN-filled unused slots, so a wrong address poisons the output. The scheduler test requires every request to match its reference while requests join and leave the batch.

## Reproduce

Needs an NVIDIA GPU, Docker with GPU support, and access to Llama-3.2-1B on Hugging Face.

    huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ~/models/llama-3.2-1b
    python scripts/make_reference.py
    python scripts/export_for_cpp.py
    python scripts/make_index_tsv.py
    python scripts/export_greedy.py
    python scripts/make_rmsnorm_reference.py
    python scripts/make_rope_reference.py
    python scripts/make_swiglu_reference.py
    python scripts/make_attention_reference.py
    bash scripts/docker.sh bash -c "cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --output-on-failure"
    bash scripts/docker.sh ./build/tests/test_scheduler

## Known limitations

Greedy decoding only; no end-of-text handling; prompts are processed whole with the slow naive attention kernel; each prompt is its own forward pass (no chunked or batched prefill); the fast attention kernel is specialised to head dim 64, 16-token pages and 4 query heads per KV head; single GPU; results come from one laptop GPU.
