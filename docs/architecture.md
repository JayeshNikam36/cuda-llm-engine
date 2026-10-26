# Model architecture: Llama-3.2-1B

All numbers below are read from `config.json` and `model.safetensors` by `scripts/inspect_model.py`.

## Config

| item | value |
|---|---|
| layers | 16 |
| hidden size | 2048 |
| query heads | 32 |
| KV heads (GQA) | 8 (4 query heads share each KV head) |
| head dimension | 64 |
| MLP intermediate size | 8192 |
| vocab size | 128256 |
| RMSNorm epsilon | 1e-05 |
| RoPE theta | 500000.0 |
| RoPE scaling | `{"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"}` |
| max positions | 131072 |
| tied embedding and output head | True (separate lm_head.weight tensor present: False) |
| stored dtype(s) | BF16 |

## Byte math

- Total parameters: 1,235,814,400 (1.236 B)
- Size in FP16: 2.472 GB
- KV cache per token (FP16): 32,768 bytes (32.0 KB)
- Decode ceiling at batch 1: about 175 GB/s / 2.47 GB = **71 tokens/s** (every weight is read once per token)

## Tensors in layer 0 (every layer has the same set)

| name | shape | dtype |
|---|---|---|
| model.layers.0.input_layernorm.weight | [2048] | BF16 |
| model.layers.0.mlp.down_proj.weight | [2048, 8192] | BF16 |
| model.layers.0.mlp.gate_proj.weight | [8192, 2048] | BF16 |
| model.layers.0.mlp.up_proj.weight | [8192, 2048] | BF16 |
| model.layers.0.post_attention_layernorm.weight | [2048] | BF16 |
| model.layers.0.self_attn.k_proj.weight | [512, 2048] | BF16 |
| model.layers.0.self_attn.o_proj.weight | [2048, 2048] | BF16 |
| model.layers.0.self_attn.q_proj.weight | [2048, 2048] | BF16 |
| model.layers.0.self_attn.v_proj.weight | [512, 2048] | BF16 |

## Tensors outside the layers

| name | shape | dtype |
|---|---|---|
| model.embed_tokens.weight | [128256, 2048] | BF16 |
| model.norm.weight | [2048] | BF16 |

## Forward pass (what the CUDA code must implement)

1. Embedding lookup: token ids to vectors of size hidden
2. For each layer:
   1. RMSNorm (input_layernorm)
   2. Q, K, V projections (GEMMs)
   3. RoPE applied to Q and K
   4. Write K, V into the KV cache
   5. Attention (causal softmax over cached K, V), GQA
   6. Output projection (o_proj), then residual add
   7. RMSNorm (post_attention_layernorm)
   8. MLP: gate and up projections, SiLU(gate) * up, down projection, then residual add
3. Final RMSNorm
4. Output projection to vocab logits (tied to the embedding matrix if tied)
5. Sampling (greedy first)
