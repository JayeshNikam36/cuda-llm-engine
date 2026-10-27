import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = os.path.expanduser("~/models/llama-3.2-1b")
REF = "tests/reference"
NH, NKV, HD = 32, 8, 64
G = NH // NKV

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
layer = model.model.layers[0]

def load(i, name, shape):
    a = np.fromfile(f"{REF}/bin/prompt_{i:02d}_{name}.f16", dtype=np.float16)
    return torch.from_numpy(a).view(*shape).cuda()

for i in range(8):
    d = torch.load(f"{REF}/prompt_{i:02d}.pt")
    x = d["after_embedding"].cuda()
    T = x.shape[1]
    with torch.no_grad():
        v = layer.self_attn.v_proj(layer.input_layernorm(x)).view(T, NKV, HD)
    q = load(i, "q_rope", (T, NH, HD)).float()
    k = load(i, "k_rope", (T, NKV, HD)).float().repeat_interleave(G, dim=1)   # head h uses kv head h // 4
    vr = v.float().repeat_interleave(G, dim=1)
    qh, kh, vh = q.permute(1, 0, 2), k.permute(1, 0, 2), vr.permute(1, 0, 2)   # [NH, T, HD]
    scores = qh @ kh.transpose(1, 2) * (HD ** -0.5)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device="cuda"), diagonal=1)
    p = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    out = (p @ vh).permute(1, 0, 2).contiguous()                                # [T, NH, HD]
    out.half().cpu().numpy().tofile(f"{REF}/bin/prompt_{i:02d}_attn.f16")
    v.contiguous().cpu().numpy().tofile(f"{REF}/bin/prompt_{i:02d}_v.f16")
    print(i, T, "tokens")
print("saved attention references (float32 math, rounded to fp16)")
