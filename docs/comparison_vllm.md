# Comparison with vLLM 0.32.0 (same GPU, same prompts)

GPU: RTX 4050 Laptop (6 GB). Greedy decoding, exactly 32 new tokens per request, the 8 reference prompts (6 to 402 tokens) repeated to 24 requests for the batched case. Both engines' outputs match Hugging Face on every request.
vLLM: dtype float16, max_num_seqs 8, max_model_len 1024, gpu_memory_utilization 0.70, CUDA graphs on, prompt token ids passed in, no detokenization. Run with VLLM_USE_FLASHINFER_SAMPLER=0 because WSL has no nvcc for FlashInfer's JIT; for greedy decoding the sampling path is a small cost.
Own engine: all requests available at once (SPACING=0), up to 8 running, scheduler test. Runs bracket the vLLM run (before, between failed attempt, after).

| workload | vLLM words/s | own engine words/s | notes |
|---|---|---|---|
| 24 requests, up to 8 at once | 482, 533, 544 (median 533) | 274, 288 (before); 255, 261; 259, 259 (after) | vLLM about 1.9 to 2.1x faster |
| one request at a time | 68, 68, 67 (8 requests) | 59, 60, 60, 61, 60, 61, 58, 61 (24 requests) | vLLM about 10 to 17% faster |

Rough per-step arithmetic (assuming about 100 steps for 24 requests of 32 words, 8 at a time): vLLM about 15 ms per step including prefill; own engine about 30 ms per step including prefill. The weight-read floor is 14 ms.

Suspected causes of the gap, none measured yet: CPU-side argmax and logits copy every step, no CUDA Graphs, slow naive-attention prefill for the 402-token prompts.
The one-at-a-time runs use 8 requests for vLLM and 24 for the own engine (same prompt mix).
