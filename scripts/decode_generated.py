import os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/llama-3.2-1b"))
for line in open("generated.txt"):
    i, rest = line.split(":")
    print(i, repr(tok.decode([int(x) for x in rest.split()])))
