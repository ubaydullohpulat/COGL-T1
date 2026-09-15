"""Log real MoE routing (top-k expert ids per token per layer) for Ling-3.0-tiny on the fixed prompt set.

Prefill routing = the prompt; decode routing = the model's own sampled continuation (what serving sees).
Output: traces/routing_v1.npz + results/generations_v1.jsonl
Run: ../../.venv/bin/python trace_ling.py [--max-new 256]
"""
import argparse, json, platform, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models import bailing_moe_v3 as bm

from prompts import PROMPTS

MODEL = "../../models/Ling-3.0-tiny-MLX-4bit-kvbfix"  # made by fix_kv_b_proj.py
REVISION = "rapid-mlx/Ling-3.0-tiny-MLX-4bit@328a497f3eb2cdd3d064f1da2b1084a2791387d5 + fix_kv_b_proj.py"

ap = argparse.ArgumentParser()
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--top-p", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=20260916)
args = ap.parse_args()

RECORD = []  # list of (layer_idx, indices array) appended during a forward pass
_orig_gate = bm.BailingGate.__call__


def _gate(self, x):
    idx, sc = _orig_gate(self, x)
    RECORD.append((self._layer_idx, idx))
    return idx, sc


bm.BailingGate.__call__ = _gate

model, tok = load(MODEL)
moe_layers = [i for i, l in enumerate(model.layers) if isinstance(l.mlp, bm.BailingSparseMoE)]
for i in moe_layers:
    model.layers[i].mlp.gate._layer_idx = i
sampler = make_sampler(temp=args.temp, top_p=args.top_p)
mx.random.seed(args.seed)


def collect():
    """Stack RECORD into [n_moe_layers, T, k] int16 and clear it."""
    by_layer = {}
    for li, idx in RECORD:
        by_layer.setdefault(li, []).append(np.array(idx.reshape(-1, idx.shape[-1])))
    RECORD.clear()
    return np.stack([np.concatenate(by_layer[li], 0) for li in moe_layers]).astype(np.int16)


out = {}
gens = []
t_start = time.time()
for pi, (domain, text) in enumerate(PROMPTS):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
    prompt = mx.array(ids)[None]
    cache = model.make_cache()
    logits = model(prompt, cache=cache)[:, -1, :]
    mx.eval(logits)
    prefill = collect()
    new = []
    for _ in range(args.max_new):
        nxt = sampler(logits)
        mx.eval(nxt)
        t = int(nxt.item())
        if t in tok.eos_token_ids:  # no forward pass for EOS, so nothing extra was recorded
            break
        new.append(t)
        logits = model(nxt.reshape(1, 1), cache=cache)[:, -1, :]
        mx.eval(logits)
    decode = collect() if new else np.zeros((len(moe_layers), 0, 8), np.int16)
    decode = decode[:, : len(new)]
    out[f"p{pi:02d}_prefill"] = prefill
    out[f"p{pi:02d}_decode"] = decode
    gens.append({"i": pi, "domain": domain, "prompt": text, "prompt_tokens": len(ids),
                 "new_tokens": len(new), "output": tok.decode(new)})
    print(f"[{pi:02d}] {domain:8s} prompt={len(ids):4d} new={len(new):4d}  {tok.decode(new)[:70]!r}", flush=True)

np.savez_compressed("traces/routing_v1.npz", moe_layers=np.array(moe_layers),
                    domains=np.array([d for d, _ in PROMPTS]), **out)
meta = {"model": REVISION, "mlx": mx.__version__, "python": platform.python_version(), "seed": args.seed,
        "temp": args.temp, "top_p": args.top_p, "max_new": args.max_new, "prompts": "prompts.py v1",
        "runtime_s": round(time.time() - t_start, 1), "n_experts": int(model.args.num_experts),
        "top_k": int(model.args.num_experts_per_tok), "moe_layers": moe_layers}
with open("results/generations_v1.jsonl", "w") as f:
    f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
    for g in gens:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")
print(json.dumps(meta))
