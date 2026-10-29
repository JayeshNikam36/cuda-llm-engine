# KV cache capacity: paged vs contiguous reservation

Measured after loading the model: 2.80 GB free of 6.44 GB total; KV budget 2.53 GB (256 MB headroom).
KV cost: 32768 bytes per token (K and V, 16 layers). Budget holds 77216 tokens = 4826 pages of 16 tokens.

Contiguous = every request reserves max_seq tokens up front. Paged = pages taken as the sequence grows.
The request lengths are simulated (seeded); only the memory numbers are measured.

| workload (final length per request) | paged: requests | paged: memory used | contiguous (reserve 2048): requests | used | contiguous (reserve 4096): requests | used | paged vs 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed 512 | 150 | 100% | 37 | 25% | 18 | 12% | 4.1x |
| uniform 128-1024 | 134 | 99% | 37 | 24% | 18 | 11% | 3.6x |
| chat-like (median 300, tail to 2048) | 197 | 98% | 37 | 19% | 18 | 8% | 5.3x |

'used' = share of the reserved KV memory that holds real tokens. The rest is wasted: reserved but empty.
