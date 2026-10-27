# Baseline: full forward pass, no KV cache

GPU: NVIDIA GeForce RTX 4050 Laptop GPU | peak used for %: 175 GB/s | 3 warmup + 15 timed runs per row

| tokens in sequence | median ms | min ms | max ms | words/s | weight GB/s | % of peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.06 | 13.96 | 14.13 | 71.1 | 175.8 | 100.4% |
| 8 | 14.38 | 14.26 | 15.35 | 69.5 | 171.9 | 98.2% |
| 32 | 15.56 | 15.46 | 15.65 | 64.3 | 158.9 | 90.8% |
| 128 | 23.49 | 23.44 | 23.56 | 42.6 | 105.2 | 60.1% |
| 402 | 100.72 | 100.56 | 100.92 | 9.9 | 24.5 | 14.0% |

Notes: 'weight GB/s' = 2.472 GB of weights divided by the median time. It is only meaningful when the pass is memory-bound (few tokens); at long sequences compute and the naive attention dominate.
