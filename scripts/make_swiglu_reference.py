import os, torch
from transformers import AutoModelForCausalLM

MODEL = os.path.expanduser("~/models/llama-3.2-1b")
REF = "tests/reference"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
layer = model.model.layers[0]
for i in range(8):
    d = torch.load(f"{REF}/prompt_{i:02d}.pt")
    x = d["after_embedding"].cuda()
    with torch.no_grad():
        h = layer.post_attention_layernorm(x)
        gate = layer.mlp.gate_proj(h)
        up = layer.mlp.up_proj(h)
        out = layer.mlp.act_fn(gate) * up
    for name, t in (("gate", gate), ("up", up), ("swiglu", out)):
        t[0].contiguous().cpu().numpy().tofile(f"{REF}/bin/prompt_{i:02d}_{name}.f16")
    print(i, tuple(out.shape))
print("saved swiglu references")
