import os
import time
import statistics
import numpy as np


def main():
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    model = os.path.expanduser("~/models/llama-3.2-1b")
    ref = "tests/reference/bin"
    NPROMPT, NREQ = 8, 24
    util = float(os.environ.get("VLLM_UTIL", "0.70"))
    eager = os.environ.get("EAGER", "0") == "1"
    max_seqs = int(os.environ.get("MAX_SEQS", "8"))

    prompts = [np.fromfile(f"{ref}/prompt_{i:02d}_ids.i32", dtype=np.int32).tolist() for i in range(NPROMPT)]
    expected = [np.fromfile(f"{ref}/prompt_{i:02d}_greedy.i32", dtype=np.int32).tolist() for i in range(NPROMPT)]
    batch = [prompts[i % NPROMPT] for i in range(NREQ)]
    batch_order = [i % NPROMPT for i in range(NREQ)]
    single = [prompts[i] for i in range(NPROMPT)]

    llm = LLM(model=model, dtype="float16", max_model_len=1024, max_num_seqs=max_seqs,
              gpu_memory_utilization=util, enforce_eager=eager)
    sp = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True, detokenize=False)

    def gen(plist):
        return llm.generate([{"prompt_token_ids": p} for p in plist], sp, use_tqdm=False)

    def n_words(outs):
        return sum(len(o.outputs[0].token_ids) for o in outs)

    def n_correct(outs, order):
        return sum(list(o.outputs[0].token_ids) == expected[k] for o, k in zip(outs, order))

    gen(batch[:2])      # warm-up
    gen(single[:1])

    batch_wps, single_wps = [], []
    batch_ok = single_ok = 0
    for rep in range(3):
        t0 = time.perf_counter()
        outs = gen(batch)
        dt = time.perf_counter() - t0
        wps = n_words(outs) / dt
        batch_wps.append(wps)
        batch_ok = n_correct(outs, batch_order)
        print(f"RESULT all {NREQ} at once (max {max_seqs} running), rep {rep + 1}: {wps:.0f} words/s, {dt:.2f} s, identical to HF: {batch_ok}/{NREQ}")
    for rep in range(3):
        t0 = time.perf_counter()
        outs = []
        for p in single:
            outs += gen([p])
        dt = time.perf_counter() - t0
        wps = n_words(outs) / dt
        single_wps.append(wps)
        single_ok = n_correct(outs, list(range(NPROMPT)))
        print(f"RESULT one request at a time ({NPROMPT} requests), rep {rep + 1}: {wps:.0f} words/s, {dt:.2f} s, identical to HF: {single_ok}/{NPROMPT}")

    md = (
        "# vLLM baseline (same machine, same prompts)\n\n"
        f"vLLM {vllm.__version__}, torch {torch.__version__}, dtype float16, max_num_seqs {max_seqs}, "
        f"gpu_memory_utilization {util}, enforce_eager {eager}. Greedy, exactly 32 new tokens, "
        "prompt token ids passed in, no detokenization.\n\n"
        "| workload | words/s per repetition | median | outputs identical to Hugging Face |\n|---|---|---:|---:|\n"
        f"| {NREQ} requests, up to {max_seqs} at once | {', '.join(f'{w:.0f}' for w in batch_wps)} | {statistics.median(batch_wps):.0f} | {batch_ok}/{NREQ} |\n"
        f"| one request at a time | {', '.join(f'{w:.0f}' for w in single_wps)} | {statistics.median(single_wps):.0f} | {single_ok}/{NPROMPT} |\n"
    )
    with open("docs/vllm_baseline.md", "w") as f:
        f.write(md)
    print("RESULT wrote docs/vllm_baseline.md")


if __name__ == "__main__":
    main()
