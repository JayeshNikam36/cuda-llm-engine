# Optimization log: decode attention

Hardware: RTX 4050 Laptop GPU, practical memory ceiling about 175 GB/s (docs/bandwidth.md).
Method: L2 cache flushed before each timed run, median of 15 runs, useful bytes = unique K and V bytes per sequence per layer.
All rows at 2048 tokens of context.

| # | change | result | notes |
|---|---|---|---|
| 0 | naive paged kernel: one block per query head, serial softmax in one thread | 346 us per sequence, 6.9% of peak | baseline; time grows linearly with context |
| 1 | fast v1: 8 threads per position with 128-bit loads, one block per KV head so 4 query heads share each load, online softmax, split-KV with a merge kernel | batch 16: 29.1 us, 82.5% of peak (12x over naive) | 119 registers, no spills; matches naive within 1.22e-4 |
| 2 | split count sweep by batch size | batch 1: 82.9 us with 1 split, 41.0 us with 8; batch 2: 49.7 us, 36.4 us with 4; batch 4: about flat; batch 16: 1 split best | I predicted more splits would win at batch 16 and was wrong; best split count fits "about 64 blocks in the grid"; helper added as choose_num_splits |
| 3 | `#pragma unroll 2` on the main loop | inconclusive | first run showed batch 16 at 33.1 us vs 29.1 us before, but the same code without the pragma later measured 33.2 us, so GPU state drifted between runs and the comparison is invalid. The first attempt silently did not apply (caught by grepping for the pragma). Registers 119 to 128, no spills |

Not tried yet: cp.async prefetch, fusing the merge into the split kernel, CUDA Graphs for the launch overhead at short context.

Correction: an earlier version of row 3 said unroll made the kernel 12% slower. That was wrong. After reverting (git tree clean, identical code) the batch-16 time stayed at 33.2 us, so the slowdown came from the machine, not the pragma. Cross-run comparisons on this laptop GPU are unreliable; compare variants with interleaved A/B runs in one process and record clocks.
