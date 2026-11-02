# Continuous batching: 24 requests, one at a time vs up to 8 at once

Workload: the 8 reference prompts three times each (6 to 402 tokens, 32 new words each), one request becomes eligible every 2 scheduler iterations.
Same process, interleaved: one at a time, up to 8, one at a time, up to 8. Outputs checked against Hugging Face greedy output.

| round | mode | decode steps | mean batch | total | words/s | outputs correct |
|---:|---|---:|---:|---:|---:|---:|
| 1 | one at a time | 744 | 1.0 | 12.73 s | 60 | 24/24 |
| 1 | up to 8 at once | 107 | 7.0 | 3.15 s | 244 | 24/24 |
| 2 | one at a time | 744 | 1.0 | 12.70 s | 60 | 24/24 |
| 2 | up to 8 at once | 107 | 7.0 | 3.09 s | 249 | 24/24 |

Throughput gain: 4.04x (round 1) and 4.12x (round 2).

Caveat on latency: arrivals are defined in scheduler iterations, not wall-clock time, so the one-at-a-time mode (faster iterations) sees a higher effective arrival rate and builds a long queue. Time to first word (median 6.0 s vs 0.37 s) and latency numbers therefore exaggerate the benefit; throughput is the fair comparison.
