import json, math, os
from safetensors import safe_open

MODEL = os.path.expanduser("~/models/llama-3.2-1b")
BW = 175e9  # measured practical bandwidth in bytes/s (see docs/bandwidth.md)

cfg = json.load(open(f"{MODEL}/config.json"))
shapes = {}
with safe_open(f"{MODEL}/model.safetensors", "pt") as f:
    for k in f.keys():
        sl = f.get_slice(k)
        shapes[k] = (tuple(sl.get_shape()), sl.get_dtype())

L = cfg["num_hidden_layers"]
H = cfg["hidden_size"]
nh = cfg["num_attention_heads"]
nkv = cfg["num_key_value_heads"]
hd = cfg.get("head_dim", H // nh)
inter = cfg["intermediate_size"]
vocab = cfg["vocab_size"]
tied = cfg.get("tie_word_embeddings", False)
has_lm_head = "lm_head.weight" in shapes

params = sum(math.prod(s) for s, _ in shapes.values())
dtypes = sorted({d for _, d in shapes.values()})
fp16_bytes = params * 2
kv_per_token = 2 * L * nkv * hd * 2
ceiling = BW / fp16_bytes

lines = []
p = lines.append
p("# Model architecture: Llama-3.2-1B\n")
p("All numbers below are read from `config.json` and `model.safetensors` by `scripts/inspect_model.py`.\n")
p("## Config\n")
p("| item | value |\n|---|---|")
p(f"| layers | {L} |")
p(f"| hidden size | {H} |")
p(f"| query heads | {nh} |")
p(f"| KV heads (GQA) | {nkv} ({nh // nkv} query heads share each KV head) |")
p(f"| head dimension | {hd} |")
p(f"| MLP intermediate size | {inter} |")
p(f"| vocab size | {vocab} |")
p(f"| RMSNorm epsilon | {cfg.get('rms_norm_eps')} |")
p(f"| RoPE theta | {cfg.get('rope_theta')} |")
p(f"| RoPE scaling | `{json.dumps(cfg.get('rope_scaling'))}` |")
p(f"| max positions | {cfg.get('max_position_embeddings')} |")
p(f"| tied embedding and output head | {tied} (separate lm_head.weight tensor present: {has_lm_head}) |")
p(f"| stored dtype(s) | {', '.join(dtypes)} |")
p("")
p("## Byte math\n")
p(f"- Total parameters: {params:,} ({params / 1e9:.3f} B)")
p(f"- Size in FP16: {fp16_bytes / 1e9:.3f} GB")
p(f"- KV cache per token (FP16): {kv_per_token:,} bytes ({kv_per_token / 1024:.1f} KB)")
p(f"- Decode ceiling at batch 1: about {BW / 1e9:.0f} GB/s / {fp16_bytes / 1e9:.2f} GB = **{ceiling:.0f} tokens/s** (every weight is read once per token)")
p("")
p("## Tensors in layer 0 (every layer has the same set)\n")
p("| name | shape | dtype |\n|---|---|---|")
for k, (s, d) in sorted(shapes.items()):
    if k.startswith("model.layers.0."):
        p(f"| {k} | {list(s)} | {d} |")
p("")
p("## Tensors outside the layers\n")
p("| name | shape | dtype |\n|---|---|---|")
for k, (s, d) in sorted(shapes.items()):
    if not k.startswith("model.layers."):
        p(f"| {k} | {list(s)} | {d} |")
p("")
p("## Forward pass (what the CUDA code must implement)\n")
p("1. Embedding lookup: token ids to vectors of size hidden")
p("2. For each layer:")
p("   1. RMSNorm (input_layernorm)")
p("   2. Q, K, V projections (GEMMs)")
p("   3. RoPE applied to Q and K")
p("   4. Write K, V into the KV cache")
p("   5. Attention (causal softmax over cached K, V), GQA")
p("   6. Output projection (o_proj), then residual add")
p("   7. RMSNorm (post_attention_layernorm)")
p("   8. MLP: gate and up projections, SiLU(gate) * up, down projection, then residual add")
p("3. Final RMSNorm")
p("4. Output projection to vocab logits (tied to the embedding matrix if tied)")
p("5. Sampling (greedy first)")

open("docs/architecture.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:32]))
print("\nwrote docs/architecture.md")
