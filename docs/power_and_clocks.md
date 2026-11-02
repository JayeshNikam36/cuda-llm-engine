# Power and clock state during benchmarks

Laptop plugged in (battery status Full, adapter online). nvidia-smi sampled once per second inside the container during the CPU-vs-GPU argmax A/B run, busy samples only (29 of them):

- 20 samples: SM clock 2640 to 2655 MHz, memory clock 8000 MHz, no throttle flags
- 9 samples: SM clock 1110 to 2265 MHz, memory clock 7000 MHz, throttle code 0x24 (software power cap plus software thermal slowdown, per the NVML bit definitions as I read them)
- maximum power draw 69.6 W

So even plugged in, sustained load makes the GPU shed clocks during part of a run. The samples are not aligned with individual runs, so this does not prove the cause of any specific slow run. An earlier sampler reading of 13.3 W maximum came from a light memory-bound benchmark and should not be read as a power cap.

The CPU-vs-GPU argmax A/B reproduced across two sessions within about 3% (batched medians 291/291 and 288/282 words/s; one at a time 64/68 and 63/67 words/s). Round 1 of the batched runs is consistently slower (warm-up), and prefill time differed between argmax modes in both sessions (batched: about 230 to 260 ms longer with GPU argmax; one at a time: about 190 ms shorter), possibly an order effect from a fixed run order.

## Update after the order-balanced A/B (rounds 2 to 5, round 1 is warm-up)

- Batched whole-run throughput ranged from 235 to 361 words/s. The fastest runs were GPU-argmax runs placed right after one-at-a-time runs (329, 361); the slowest were CPU-argmax runs placed right after another batched run (236, 239). The printed median ratio of 1.18x is therefore confounded by position and is not a valid measure of the argmax change.
- Robust in every round: per decode step at batch 8, GPU argmax saves 2.8 to 3.3 ms (CPU path 18.3 to 19.9 ms, GPU path 15.0 to 16.5 ms).
- One at a time: GPU argmax 67 words/s in all five rounds; CPU argmax 60 to 66 (median 64).
- Prefill (compute-bound) varied from 623 to 1280 ms for identical work, while decode GPU time per step varied only about 7% (14.7 to 16.7 ms). That pattern fits an SM-clock limit under sustained compute load, consistent with the throttle samples above, but it is not proven.
- The earlier "prefill got slower with GPU argmax" observation did not hold up: in this balanced run GPU-argmax prefill was lower in every round.
