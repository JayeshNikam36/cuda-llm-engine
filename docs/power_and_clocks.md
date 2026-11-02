# Power and clock state during benchmarks

Laptop plugged in (battery status Full, adapter online). nvidia-smi sampled once per second inside the container during the CPU-vs-GPU argmax A/B run, busy samples only (29 of them):

- 20 samples: SM clock 2640 to 2655 MHz, memory clock 8000 MHz, no throttle flags
- 9 samples: SM clock 1110 to 2265 MHz, memory clock 7000 MHz, throttle code 0x24 (software power cap plus software thermal slowdown, per the NVML bit definitions as I read them)
- maximum power draw 69.6 W

So even plugged in, sustained load makes the GPU shed clocks during part of a run. The samples are not aligned with individual runs, so this does not prove the cause of any specific slow run. An earlier sampler reading of 13.3 W maximum came from a light memory-bound benchmark and should not be read as a power cap.

The CPU-vs-GPU argmax A/B reproduced across two sessions within about 3% (batched medians 291/291 and 288/282 words/s; one at a time 64/68 and 63/67 words/s). Round 1 of the batched runs is consistently slower (warm-up), and prefill time differed between argmax modes in both sessions (batched: about 230 to 260 ms longer with GPU argmax; one at a time: about 190 ms shorter), possibly an order effect from a fixed run order.
