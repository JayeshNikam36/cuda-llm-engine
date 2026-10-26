# cuda-llm-engine

An LLM inference engine written from scratch in C++/CUDA: paged KV cache, continuous batching, chunked prefill, fused kernels, a split-KV GQA decode attention kernel, CUDA Graphs, FP8, and an OpenAI-compatible server. Results are benchmarked against vLLM and TensorRT-LLM.

**Status:** Phase 0 (setup and baselines) complete. Phase 1 (correct forward pass) is next.

## Hardware (development)
NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB, Ada (sm_89). Measured memory bandwidth: about 175-179 GB/s (93% of the 192 GB/s spec). Details in [docs/bandwidth.md](docs/bandwidth.md).

## Target model
Llama-3.2-1B (16 layers, 32 query heads, 8 KV heads, head dim 64). Architecture and byte math in [docs/architecture.md](docs/architecture.md). Batch-1 decode ceiling on this GPU: about 71 tokens/s.

## Build and test
Requires Docker with GPU support.

    bash scripts/docker.sh bash -c "cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --output-on-failure"
    bash scripts/docker.sh ./build/bench/bandwidth

## Reference outputs
`scripts/make_reference.py` generates Hugging Face reference outputs (greedy decoding is deterministic) that every kernel is tested against.
