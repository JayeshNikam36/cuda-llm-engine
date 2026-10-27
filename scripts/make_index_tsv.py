import json, os
d = os.path.expanduser("~/models/llama-3.2-1b-fp16")
idx = json.load(open(d + "/index.json"))
with open(d + "/index.tsv", "w") as f:
    for t in idx:
        f.write(f"{t['name']}\t{t['offset']}\t{t['nbytes']}\t{','.join(map(str, t['shape']))}\n")
print(len(idx), "rows written")
