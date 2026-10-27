import os, torch
from transformers import AutoModelForCausalLM

MODEL = os.path.expanduser("~/models/llama-3.2-1b")
REF = "tests/reference"
NH, NKV, HD = 32, 8, 64

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
layer = model.model.layers[0]
rot = model.model.rotary_emb
rot.inv_freq.detach().float().cpu().numpy().tofile(f"{REF}/bin/rope_inv_freq.f32")

def rotate_half(t):
    h = t.shape[-1] // 2
    return torch.cat((-t[..., h:], t[..., :h]), dim=-1)

for i in range(8):
    d = torch.load(f"{REF}/prompt_{i:02d}.pt")
    x = d["after_embedding"].cuda()
    T = x.shape[1]
    with torch.no_grad():
        h = layer.input_layernorm(x)
        q = layer.self_attn.q_proj(h).view(1, T, NH, HD)
        k = layer.self_attn.k_proj(h).view(1, T, NKV, HD)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        cos, sin = rot(x, pos)
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        q_out = q * cos + rotate_half(q) * sin
        k_out = k * cos + rotate_half(k) * sin
    for name, t in (("q_in", q), ("k_in", k), ("q_rope", q_out), ("k_rope", k_out)):
        t[0].contiguous().cpu().numpy().tofile(f"{REF}/bin/prompt_{i:02d}_{name}.f16")
    print(i, T, "tokens")
print("saved rope references")
