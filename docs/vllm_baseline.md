# vLLM baseline (same machine, same prompts)

vLLM 0.32.0, torch 2.13.0+cu130, dtype float16, max_num_seqs 8, gpu_memory_utilization 0.7, enforce_eager False. Greedy, exactly 32 new tokens, prompt token ids passed in, no detokenization.

| workload | words/s per repetition | median | outputs identical to Hugging Face |
|---|---|---:|---:|
| 24 requests, up to 8 at once | 482, 533, 544 | 533 | 24/24 |
| one request at a time | 68, 68, 67 | 68 | 8/8 |
