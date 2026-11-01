# Decode attention: naive paged kernel vs fast split-KV kernel (v1)

GPU: NVIDIA GeForce RTX 4050 Laptop GPU | peak used for %: 175 GB/s | 16 sequences per measurement | L2 flushed before every timed run
Naive: one launch per sequence. Fast: one launch for all 16 sequences.

| context S | kernel | splits | us per sequence | useful GB/s | % of peak | speedup vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 256 | naive | - | 117.6 | 4.5 | 2.5% | 1.0x |
| 256 | fast v1 | 1 | 6.1 | 85.3 | 48.8% | 19.1x |
| 256 | fast v1 | 2 | 6.7 | 78.0 | 44.6% | 17.5x |
| 256 | fast v1 | 4 | 7.0 | 74.4 | 42.5% | 16.7x |
| 256 | fast v1 | 8 | 7.2 | 73.1 | 41.8% | 16.4x |
| 256 | fast v1 | 16 | 12.5 | 41.8 | 23.9% | 9.4x |
| 256 | fast v1 | 32 | 13.6 | 38.5 | 22.0% | 8.6x |
| 256 | floor at peak | - | 3.0 | 175.0 | 100.0% | - |
| 1024 | naive | - | 239.2 | 8.8 | 5.0% | 1.0x |
| 1024 | fast v1 | 1 | 18.7 | 112.2 | 64.1% | 12.8x |
| 1024 | fast v1 | 2 | 18.7 | 111.9 | 63.9% | 12.8x |
| 1024 | fast v1 | 4 | 18.8 | 111.5 | 63.7% | 12.7x |
| 1024 | fast v1 | 8 | 18.7 | 112.2 | 64.1% | 12.8x |
| 1024 | fast v1 | 16 | 18.9 | 110.7 | 63.3% | 12.6x |
| 1024 | fast v1 | 32 | 20.4 | 102.7 | 58.7% | 11.7x |
| 1024 | floor at peak | - | 12.0 | 175.0 | 100.0% | - |
| 2048 | naive | - | 363.8 | 11.5 | 6.6% | 1.0x |
| 2048 | fast v1 | 1 | 33.2 | 126.5 | 72.3% | 11.0x |
| 2048 | fast v1 | 2 | 33.7 | 124.4 | 71.1% | 10.8x |
| 2048 | fast v1 | 4 | 33.3 | 125.8 | 71.9% | 10.9x |
| 2048 | fast v1 | 8 | 33.7 | 124.6 | 71.2% | 10.8x |
| 2048 | fast v1 | 16 | 34.4 | 122.0 | 69.7% | 10.6x |
| 2048 | fast v1 | 32 | 35.7 | 117.4 | 67.1% | 10.2x |
| 2048 | floor at peak | - | 24.0 | 175.0 | 100.0% | - |

useful bytes = unique K and V bytes for one sequence in one layer (S x 512 x 2 x 2). 'floor at peak' = time if those bytes streamed at the peak figure.
