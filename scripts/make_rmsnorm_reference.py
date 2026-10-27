import os, torch
from transformers import AutoModelForCausalLM

MODEL = os.path.expanduser("~/models/llama-3.2-1b")
REF = "tests/reference"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
ln = model.model.layers[0].input_layernorm
for i in range(8):
    d = torch.load(f"{REF}/prompt_{i:02d}.pt")
    x = d["after_embedding"].cuda()
    with torch.no_grad():
        y = ln(x)
    y.cpu().to(torch.float16).numpy().tofile(f"{REF}/bin/prompt_{i:02d}_norm0.f16")
    print(i, tuple(y.shape))
print("saved layer-0 input_layernorm references")
