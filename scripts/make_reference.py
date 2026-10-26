import json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.path.expanduser("~/models/llama-3.2-1b")
OUT = "tests/reference"
os.makedirs(OUT, exist_ok=True)

prompts = [
    "The capital of France is",
    "Once upon a time",
    "def fibonacci(n):",
    "The three primary colors are red, blue, and",
    "Explain why the sky is blue in one sentence:",
    "1, 2, 3, 4, 5,",
    "Water boils at 100 degrees Celsius at sea level because",
    "The quick brown fox jumps over the lazy dog. " * 40,
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

results = []
with torch.no_grad():
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        out = model(ids, output_hidden_states=True)
        torch.save({
            "input_ids": ids.cpu(),
            "last_logits": out.logits[0, -1].float().cpu(),
            "after_embedding": out.hidden_states[0].cpu(),
            "after_layer0": out.hidden_states[1].cpu(),
        }, f"{OUT}/prompt_{i:02d}.pt")
        gen = model.generate(ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        new = gen[0, ids.shape[1]:].cpu().tolist()
        results.append({
            "id": i,
            "prompt_tokens": ids[0].cpu().tolist()[:16],
            "num_prompt_tokens": ids.shape[1],
            "greedy_tokens": new,
            "greedy_text": tok.decode(new),
        })
        print(i, ids.shape[1], "tokens ->", repr(tok.decode(new)[:60]))

with open(f"{OUT}/reference.json", "w") as f:
    json.dump(results, f, indent=1)
print("saved", len(results), "references")
