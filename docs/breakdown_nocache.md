# Per-stage time of one layer (layer 0), scaled to 16 layers

## 1 tokens (estimated total 15.0 ms)

| stage | ms per layer | ms x 16 layers | share of pass |
|---|---:|---:|---:|
| attn rmsnorm | 0.023 | 0.36 | 2.4% |
| q/k/v projections | 0.082 | 1.31 | 8.7% |
| rope | 0.007 | 0.11 | 0.8% |
| attention (naive) | 0.007 | 0.11 | 0.8% |
| o_proj | 0.051 | 0.82 | 5.5% |
| residual add 1 | 0.006 | 0.10 | 0.6% |
| mlp rmsnorm | 0.005 | 0.08 | 0.5% |
| gate/up projections | 0.368 | 5.90 | 39.3% |
| swiglu | 0.015 | 0.25 | 1.6% |
| down_proj | 0.186 | 2.98 | 19.9% |
| residual add 2 | 0.005 | 0.08 | 0.5% |
| final norm + lm_head | n/a | 2.89 | 19.3% |

## 32 tokens (estimated total 17.1 ms)

| stage | ms per layer | ms x 16 layers | share of pass |
|---|---:|---:|---:|
| attn rmsnorm | 0.020 | 0.33 | 1.9% |
| q/k/v projections | 0.102 | 1.64 | 9.6% |
| rope | 0.007 | 0.11 | 0.7% |
| attention (naive) | 0.042 | 0.67 | 3.9% |
| o_proj | 0.056 | 0.90 | 5.3% |
| residual add 1 | 0.006 | 0.10 | 0.6% |
| mlp rmsnorm | 0.007 | 0.11 | 0.7% |
| gate/up projections | 0.424 | 6.78 | 39.7% |
| swiglu | 0.011 | 0.18 | 1.1% |
| down_proj | 0.204 | 3.26 | 19.1% |
| residual add 2 | 0.007 | 0.11 | 0.7% |
| final norm + lm_head | n/a | 2.90 | 16.9% |

## 402 tokens (estimated total 103.3 ms)

| stage | ms per layer | ms x 16 layers | share of pass |
|---|---:|---:|---:|
| attn rmsnorm | 0.014 | 0.23 | 0.2% |
| q/k/v projections | 0.277 | 4.44 | 4.3% |
| rope | 0.016 | 0.26 | 0.3% |
| attention (naive) | 3.643 | 58.29 | 56.4% |
| o_proj | 0.206 | 3.29 | 3.2% |
| residual add 1 | 0.018 | 0.29 | 0.3% |
| mlp rmsnorm | 0.010 | 0.16 | 0.2% |
| gate/up projections | 1.255 | 20.07 | 19.4% |
| swiglu | 0.088 | 1.41 | 1.4% |
| down_proj | 0.702 | 11.24 | 10.9% |
| residual add 2 | 0.044 | 0.70 | 0.7% |
| final norm + lm_head | n/a | 2.90 | 2.8% |

Each stage is timed with its own events and a sync, so the estimated total runs a little above the measured pass in docs/baseline_nocache.md.
