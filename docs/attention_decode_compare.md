# Decode attention: naive paged kernel vs fast split-KV kernel (v1)

GPU: NVIDIA GeForce RTX 4050 Laptop GPU | peak used for %: 175 GB/s | 16 sequences per measurement | L2 flushed before every timed run
Naive: one launch per sequence. Fast: one launch for all 16 sequences.

| context S | kernel | splits | us per sequence | useful GB/s | % of peak | speedup vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 256 | naive | - | 124.0 | 4.2 | 2.4% | 1.0x |
| 256 | fast v1 | 1 | 6.3 | 83.6 | 47.8% | 19.8x |
| 256 | fast v1 | 2 | 6.7 | 78.0 | 44.6% | 18.5x |
| 256 | fast v1 | 4 | 7.2 | 73.1 | 41.8% | 17.3x |
| 256 | fast v1 | 8 | 9.3 | 56.1 | 32.1% | 13.3x |
| 256 | fast v1 | 16 | 13.8 | 37.9 | 21.7% | 9.0x |
| 256 | fast v1 | 32 | 17.0 | 30.8 | 17.6% | 7.3x |
| 256 | floor at peak | - | 3.0 | 175.0 | 100.0% | - |
| 1024 | naive | - | 471.8 | 4.4 | 2.5% | 1.0x |
| 1024 | fast v1 | 1 | 19.5 | 107.4 | 61.4% | 24.2x |
| 1024 | fast v1 | 2 | 19.8 | 105.7 | 60.4% | 23.8x |
| 1024 | fast v1 | 4 | 19.5 | 107.8 | 61.6% | 24.3x |
| 1024 | fast v1 | 8 | 20.6 | 101.8 | 58.2% | 22.9x |
| 1024 | fast v1 | 16 | 24.2 | 86.7 | 49.5% | 19.5x |
| 1024 | fast v1 | 32 | 33.7 | 62.2 | 35.5% | 14.0x |
| 1024 | floor at peak | - | 12.0 | 175.0 | 100.0% | - |
| 2048 | naive | - | 963.3 | 4.4 | 2.5% | 1.0x |
| 2048 | fast v1 | 1 | 33.4 | 125.5 | 71.7% | 28.8x |
| 2048 | fast v1 | 2 | 34.2 | 122.7 | 70.1% | 28.2x |
| 2048 | fast v1 | 4 | 32.4 | 129.3 | 73.9% | 29.7x |
| 2048 | fast v1 | 8 | 32.8 | 128.0 | 73.1% | 29.4x |
| 2048 | fast v1 | 16 | 34.8 | 120.7 | 69.0% | 27.7x |
| 2048 | fast v1 | 32 | 40.8 | 102.7 | 58.7% | 23.6x |
| 2048 | floor at peak | - | 24.0 | 175.0 | 100.0% | - |

useful bytes = unique K and V bytes for one sequence in one layer (S x 512 x 2 x 2). 'floor at peak' = time if those bytes streamed at the peak figure.
