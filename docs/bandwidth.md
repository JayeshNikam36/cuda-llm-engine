# Memory bandwidth baseline

GPU: NVIDIA GeForce RTX 4050 Laptop GPU (20 SMs, 24 MB L2, 96-bit bus, 6 GB)
Spec peak: 192 GB/s
Method: vectorized (128-bit) copy kernel, 512 MiB buffers, 5 warmup runs, median of 30 timed runs, read + write counted.

| block | grid_mult | median ms | GB/s |
|------:|----------:|----------:|-----:|
| 128 | 2 | 6.002 | 178.9 |
| 128 | 4 | 6.225 | 172.5 |
| 128 | 8 | 6.166 | 174.2 |
| 128 | 16 | 6.177 | 173.8 |
| 256 | 2 | 6.218 | 172.7 |
| 256 | 4 | 6.184 | 173.6 |
| 256 | 8 | 6.199 | 173.2 |
| 256 | 16 | 6.159 | 174.3 |
| 512 | 2 | 6.169 | 174.1 |
| 512 | 4 | 6.162 | 174.2 |
| 512 | 8 | 6.159 | 174.3 |
| 512 | 16 | 6.137 | 175.0 |

Best: 178.9 GB/s (93.2% of spec peak). Most configurations land at about 172-175 GB/s, so I use about 175 GB/s as the practical ceiling.
