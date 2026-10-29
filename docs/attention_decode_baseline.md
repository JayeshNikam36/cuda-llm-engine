# Baseline: naive paged decode attention (one layer, one sequence per call)

GPU: NVIDIA GeForce RTX 4050 Laptop GPU | peak used for %: 175 GB/s | 16 sequences per measurement, launched back to back

| context S | us per call | ideal us at peak | useful MB per call | useful GB/s | % of peak |
|---:|---:|---:|---:|---:|---:|
| 256 | 45.4 | 3.0 | 0.52 | 11.5 | 6.6% |
| 1024 | 217.8 | 12.0 | 2.10 | 9.6 | 5.5% |
| 2048 | 434.9 | 24.0 | 4.19 | 9.6 | 5.5% |

useful bytes = unique K and V bytes for one sequence in one layer (S x 512 x 2 x 2). The kernel may request more than that, because 4 query heads share each KV head.
