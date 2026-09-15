"""Measure real compute costs of Ling-3.0-tiny (MLX 4-bit) on this Mac, as inputs for SwarmLab.

1. End-to-end: prefill tok/s at several prompt lengths; decode tok/s.
2. Per-component decode cost per layer (attention / router / routed experts / shared expert), with forced evals.
3. Routed-expert cost vs number of experts k on one token (what a remote expert peer computes).
4. Eval/sync overhead of an empty op (to correct 2).
Output: results/compute_v1.json
Run: ../../.venv/bin/python bench_compute.py
"""
import json, platform, statistics, subprocess, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import bailing_moe_v3 as bm

MODEL = "../../models/Ling-3.0-tiny-MLX-4bit-kvbfix"  # made by fix_kv_b_proj.py
model, tok = load(MODEL)
L = model.layers
args = model.args
mx.random.seed(0)
TEXT = open("prompts.py", encoding="utf-8").read() * 20
BASE_IDS = tok.encode(TEXT)


def timed(fn, reps=5, warmup=2):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); xs.append(time.perf_counter() - t)
    return statistics.median(xs)


res = {"model": "rapid-mlx/Ling-3.0-tiny-MLX-4bit@328a497 + fix_kv_b_proj.py", "mlx": mx.__version__, "python": platform.python_version(),
       "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()}

# 1. end-to-end prefill
pre = {}
for P in [128, 512, 2048, 4096]:
    ids = mx.array(BASE_IDS[:P])[None]
    def run():
        c = model.make_cache(); mx.eval(model(ids, cache=c))
    s = timed(run, reps=3)
    pre[P] = {"seconds": s, "tok_per_s": P / s}
res["prefill"] = pre

# 1b. end-to-end decode after a 256-token prompt (greedy, no sampling cost)
def decode_run(n=128):
    c = model.make_cache()
    logits = model(mx.array(BASE_IDS[:256])[None], cache=c)[:, -1, :]
    mx.eval(logits)
    t = time.perf_counter()
    for _ in range(n):
        nxt = mx.argmax(logits, axis=-1).reshape(1, 1)
        logits = model(nxt, cache=c)[:, -1, :]
        mx.eval(logits)
    return n / (time.perf_counter() - t)
decode_run(16)
res["decode_tok_per_s"] = statistics.median([decode_run() for _ in range(3)])

# 4. sync overhead
z = mx.zeros((1,))
res["eval_overhead_s"] = timed(lambda: mx.eval(z + 1), reps=200, warmup=20)

# 2. per-component decode timing
c = model.make_cache()
logits = model(mx.array(BASE_IDS[:256])[None], cache=c)[:, -1, :]
mx.eval(logits)
comp = {"attention_kda": [], "attention_mla": [], "dense_mlp": [], "router": [], "routed_experts": [], "shared_expert": [], "head": []}
STEPS = 48
for step in range(STEPS):
    nxt = mx.argmax(logits, axis=-1).reshape(1, 1)
    h = model.model.word_embeddings(nxt); mx.eval(h)
    for layer, lc in zip(L, c):
        t0 = time.perf_counter()
        h = h + layer.attention(layer.input_layernorm(h), None, lc); mx.eval(h)
        t1 = time.perf_counter()
        comp["attention_kda" if layer.is_linear else "attention_mla"].append(t1 - t0)
        x = layer.post_attention_layernorm(h)
        if isinstance(layer.mlp, bm.BailingSparseMoE):
            t2 = time.perf_counter(); idx, sc = layer.mlp.gate(x); mx.eval(idx, sc)
            t3 = time.perf_counter(); r = (layer.mlp.switch_mlp(x, idx) * sc[..., None]).sum(axis=-2).astype(x.dtype); mx.eval(r)
            t4 = time.perf_counter(); s = layer.mlp.shared_experts(x); mx.eval(s)
            t5 = time.perf_counter()
            comp["router"].append(t3 - t2); comp["routed_experts"].append(t4 - t3); comp["shared_expert"].append(t5 - t4)
            h = h + r + s
        else:
            t2 = time.perf_counter(); h = h + layer.mlp(x); mx.eval(h); comp["dense_mlp"].append(time.perf_counter() - t2)
    t6 = time.perf_counter(); logits = model.lm_head(model.model.norm(h))[:, -1, :]; mx.eval(logits)
    comp["head"].append(time.perf_counter() - t6)
res["decode_components_median_s"] = {k: statistics.median(v) for k, v in comp.items() if v}
res["decode_components_count_per_token"] = {k: len(v) // STEPS for k, v in comp.items() if v}

# 3. routed-expert cost vs k for a single token (a remote expert peer's work)
moe = next(l for l in L if isinstance(l.mlp, bm.BailingSparseMoE)).mlp
x = mx.random.normal((1, 1, args.hidden_size)).astype(mx.float16)
vs_k = {}
for k in [1, 2, 4, 8]:
    idx = mx.array(list(range(k)), dtype=mx.int32).reshape(1, 1, k)
    vs_k[k] = timed(lambda: mx.eval(moe.switch_mlp(x, idx)), reps=50, warmup=10)
res["routed_experts_vs_k_s"] = vs_k

json.dump(res, open("results/compute_v1.json", "w"), indent=2)
print(json.dumps(res, indent=2))
