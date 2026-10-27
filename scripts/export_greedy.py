import json
import numpy as np
ref = json.load(open("tests/reference/reference.json"))
for r in ref:
    np.array(r["greedy_tokens"], dtype=np.int32).tofile(
        f"tests/reference/bin/prompt_{r['id']:02d}_greedy.i32")
print("saved", len(ref), "greedy token files")
