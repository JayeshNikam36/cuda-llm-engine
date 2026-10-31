# Decode attention: naive paged kernel vs fast split-KV kernel (v1)

GPU: NVIDIA GeForce RTX 4050 Laptop GPU | peak used for %: 175 GB/s | 16 sequences per measurement | L2 flushed before every timed run
Naive: one launch per sequence. Fast: one launch for all 16 sequences.

| context S | kernel | splits | us per sequence | useful GB/s | % of peak | speedup vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 256 | naive | - | 47.0 | 11.2 | 6.4% | 1.0x |
| 256 | fast v1 | 1 | 4.7 | 112.0 | 64.0% | 10.0x |
| 256 | fast v1 | 2 | 5.1 | 102.4 | 58.5% | 9.2x |
| 256 | fast v1 | 4 | 5.0 | 105.0 | 60.0% | 9.4x |
| 256 | fast v1 | 8 | 5.2 | 99.9 | 57.1% | 9.0x |
| 256 | fast v1 | 16 | 6.5 | 81.1 | 46.3% | 7.3x |
| 256 | fast v1 | 32 | 7.5 | 70.0 | 40.0% | 6.3x |
| 256 | floor at peak | - | 3.0 | 175.0 | 100.0% | - |
| 1024 | naive | - | 175.4 | 12.0 | 6.8% | 1.0x |
| 1024 | fast v1 | 1 | 16.6 | 126.5 | 72.3% | 10.6x |
| 1024 | fast v1 | 2 | 17.0 | 123.7 | 70.7% | 10.3x |
| 1024 | fast v1 | 4 | 16.7 | 125.5 | 71.7% | 10.5x |
| 1024 | fast v1 | 8 | 16.7 | 125.5 | 71.7% | 10.5x |
| 1024 | fast v1 | 16 | 16.7 | 125.5 | 71.7% | 10.5x |
| 1024 | fast v1 | 32 | 18.2 | 115.0 | 65.7% | 9.6x |
| 1024 | floor at peak | - | 12.0 | 175.0 | 100.0% | - |
| 2048 | naive | - | 350.1 | 12.0 | 6.8% | 1.0x |
| 2048 | fast v1 | 1 | 29.1 | 144.4 | 82.5% | 12.1x |
| 2048 | fast v1 | 2 | 29.8 | 140.9 | 80.5% | 11.8x |
| 2048 | fast v1 | 4 | 29.2 | 143.4 | 81.9% | 12.0x |
| 2048 | fast v1 | 8 | 29.5 | 142.2 | 81.2% | 11.9x |
| 2048 | fast v1 | 16 | 30.2 | 138.8 | 79.3% | 11.6x |
| 2048 | fast v1 | 32 | 31.4 | 133.7 | 76.4% | 11.2x |
| 2048 | floor at peak | - | 24.0 | 175.0 | 100.0% | - |

useful bytes = unique K and V bytes for one sequence in one layer (S x 512 x 2 x 2). 'floor at peak' = time if those bytes streamed at the peak figure.
