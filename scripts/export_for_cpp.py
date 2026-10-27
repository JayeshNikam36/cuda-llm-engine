import json, os
import numpy as np
import torch
from safetensors import safe_open

MODEL = os.path.expanduser("~/models/llama-3.2-1b")
OUT_W = os.path.expanduser("~/models/llama-3.2-1b-fp16")
REF = "tests/reference"
OUT_R = f"{REF}/bin"
os.makedirs(OUT_W, exist_ok=True)
os.makedirs(OUT_R, exist_ok=True)

# 1) Weights: one flat FP16 file plus an index of (name, shape, offset)
index, offset = [], 0
with safe_open(f"{MODEL}/model.safetensors", "pt") as f, open(f"{OUT_W}/weights.bin", "wb") as out:
    for name in sorted(f.keys()):
        t = f.get_tensor(name).to(torch.float16).contiguous()
        b = t.numpy().tobytes()
        out.write(b)
        index.append({"name": name, "shape": list(t.shape), "dtype": "f16",
                      "offset": offset, "nbytes": len(b)})
        offset += len(b)
json.dump(index, open(f"{OUT_W}/index.json", "w"), indent=1)
print(f"weights: {len(index)} tensors, {offset / 1e9:.3f} GB -> {OUT_W}")

# 2) Reference tensors: raw binaries plus a meta file
meta = []
for i in range(8):
    d = torch.load(f"{REF}/prompt_{i:02d}.pt")
    ids = d["input_ids"].to(torch.int32).numpy()
    ids.tofile(f"{OUT_R}/prompt_{i:02d}_ids.i32")
    d["after_embedding"].to(torch.float16).numpy().tofile(f"{OUT_R}/prompt_{i:02d}_embed.f16")
    d["after_layer0"].to(torch.float16).numpy().tofile(f"{OUT_R}/prompt_{i:02d}_layer0.f16")
    d["last_logits"].to(torch.float32).numpy().tofile(f"{OUT_R}/prompt_{i:02d}_logits.f32")
    meta.append({"id": i, "num_tokens": int(ids.shape[1]),
                 "hidden": int(d["after_embedding"].shape[-1]),
                 "vocab": int(d["last_logits"].shape[-1])})
json.dump(meta, open(f"{OUT_R}/meta.json", "w"), indent=1)
print("reference binaries:", len(meta), "prompts ->", OUT_R)
