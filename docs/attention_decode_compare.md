# Decode attention: naive paged kernel vs fast split-KV kernel (v1)

GPU: NVIDIA GeForce RTX 4050 Laptop GPU | peak used for %: 175 GB/s | 16 sequences per measurement | L2 flushed before every timed run
Naive: one launch per sequence. Fast: one launch for all 16 sequences.

| context S | kernel | splits | us per sequence | useful GB/s | % of peak | speedup vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 256 | naive | - | 57.3 | 9.2 | 5.2% | 1.0x |
| 256 | fast v1 | 1 | 4.9 | 107.8 | 61.6% | 11.8x |
| 256 | fast v1 | 2 | 5.2 | 99.9 | 57.1% | 10.9x |
| 256 | fast v1 | 4 | 5.2 | 99.9 | 57.1% | 10.9x |
| 256 | fast v1 | 8 | 5.6 | 93.2 | 53.2% | 10.2x |
| 256 | fast v1 | 16 | 6.9 | 75.9 | 43.3% | 8.3x |
| 256 | fast v1 | 32 | 8.2 | 64.0 | 36.6% | 7.0x |
| 256 | floor at peak | - | 3.0 | 175.0 | 100.0% | - |
| 1024 | naive | - | 196.8 | 10.7 | 6.1% | 1.0x |
| 1024 | fast v1 | 1 | 16.7 | 125.5 | 71.7% | 11.8x |
| 1024 | fast v1 | 2 | 17.1 | 122.7 | 70.1% | 11.5x |
| 1024 | fast v1 | 4 | 16.9 | 124.1 | 70.9% | 11.6x |
| 1024 | fast v1 | 8 | 16.8 | 124.6 | 71.2% | 11.7x |
| 1024 | fast v1 | 16 | 16.8 | 124.6 | 71.2% | 11.7x |
| 1024 | fast v1 | 32 | 18.6 | 112.6 | 64.3% | 10.6x |
| 1024 | floor at peak | - | 12.0 | 175.0 | 100.0% | - |
| 2048 | naive | - | 346.2 | 12.1 | 6.9% | 1.0x |
| 2048 | fast v1 | 1 | 29.1 | 144.4 | 82.5% | 11.9x |
| 2048 | fast v1 | 2 | 29.8 | 140.6 | 80.4% | 11.6x |
| 2048 | fast v1 | 4 | 29.2 | 143.4 | 81.9% | 11.8x |
| 2048 | fast v1 | 8 | 29.5 | 142.2 | 81.2% | 11.7x |
| 2048 | fast v1 | 16 | 30.2 | 138.8 | 79.3% | 11.5x |
| 2048 | fast v1 | 32 | 31.4 | 133.5 | 76.3% | 11.0x |
| 2048 | floor at peak | - | 24.0 | 175.0 | 100.0% | - |

useful bytes = unique K and V bytes for one sequence in one layer (S x 512 x 2 x 2). 'floor at peak' = time if those bytes streamed at the peak figure.
