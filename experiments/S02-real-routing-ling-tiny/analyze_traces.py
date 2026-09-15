"""Routing-structure metrics (T1-lite) and hybrid-cache locality from routing_v1.npz.

Metrics (decode tokens unless noted):
  1. usage skew per layer: normalized entropy, Gini, traffic share of top 10% / 25% experts
  2. per-sequence concentration: traffic share of a sequence's own top-25% experts
  3. temporal locality: overlap |S_t ∩ S_t+1| / k, and P(expert reused within last w tokens) vs random
  4. domain clustering: cosine similarity of per-prompt usage vectors, within vs across domains
  5. hybrid cache locality: stage keeps fraction f of each layer's experts; P(all k selected experts local)
     for caches chosen by (a) random, (b) global frequency (leave-one-prompt-out), (c) same-domain frequency
     (leave-one-out), (d) the sequence's own frequency (oracle upper bound)
Output: results/locality_v1.json, results/locality_v1.md
"""
import json
import numpy as np

Z = np.load("traces/routing_v1.npz")
E, K = 128, 8
domains = list(Z["domains"])
n_prompts = len(domains)
dec = [Z[f"p{i:02d}_decode"].astype(np.int64) for i in range(n_prompts)]   # [layers, T, k]
pre = [Z[f"p{i:02d}_prefill"].astype(np.int64) for i in range(n_prompts)]
n_layers = dec[0].shape[0]
rng = np.random.default_rng(0)


def counts(tr):  # tr [layers, T, k] -> [layers, E]
    c = np.zeros((tr.shape[0], E))
    for l in range(tr.shape[0]):
        c[l] = np.bincount(tr[l].ravel(), minlength=E)
    return c


def gini(x):
    x = np.sort(x); n = len(x); cum = np.cumsum(x)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n


def onehot(tr):  # [layers, T, k] -> bool [layers, T, E]
    m = np.zeros((tr.shape[0], tr.shape[1], E), bool)
    np.put_along_axis(m, tr, True, axis=2)
    return m


out = {"n_prompts": n_prompts, "decode_tokens": int(sum(d.shape[1] for d in dec)),
       "prefill_tokens": int(sum(p.shape[1] for p in pre)), "n_moe_layers": n_layers, "E": E, "k": K}

# 1. usage skew
C = sum(counts(d) for d in dec)
P = C / C.sum(1, keepdims=True)
with np.errstate(divide="ignore", invalid="ignore"):
    ent = -(np.where(P > 0, P * np.log(P), 0)).sum(1) / np.log(E)
top = np.sort(P, 1)[:, ::-1]
out["skew"] = {
    "norm_entropy_mean": float(ent.mean()), "norm_entropy_min": float(ent.min()),
    "gini_mean": float(np.mean([gini(c) for c in C])),
    "top10pct_share_mean": float(top[:, : E // 10].sum(1).mean()),
    "top25pct_share_mean": float(top[:, : E // 4].sum(1).mean()),
    "uniform_top10pct_share": 0.1, "uniform_top25pct_share": 0.25,
    "per_layer_top25pct_share": [round(float(x), 3) for x in top[:, : E // 4].sum(1)],
}

# 2. per-sequence concentration
seq_share = []
for d in dec:
    if d.shape[1] < 32: continue
    c = counts(d); s = np.sort(c, 1)[:, ::-1]
    seq_share.append((s[:, : E // 4].sum(1) / c.sum(1)).mean())
out["per_sequence_top25pct_share_mean"] = float(np.mean(seq_share))

# 3. temporal locality
ov, reuse = [], {1: [], 4: [], 16: []}
for d in dec:
    if d.shape[1] < 32: continue
    m = onehot(d)                                  # [layers, T, E]
    ov.append(((m[:, 1:] & m[:, :-1]).sum(2) / K).mean())
    for w in reuse:
        cs = np.cumsum(m, axis=1)
        prev = cs[:, w - 1 : -1] - np.concatenate([np.zeros_like(cs[:, :1]), cs[:, : -w - 1]], 1) if w > 1 else m[:, :-1]
        hit = (m[:, w:] & (prev[:, -m[:, w:].shape[1]:] > 0)).sum(2) / K
        reuse[w].append(hit.mean())
out["temporal"] = {"adjacent_overlap": float(np.mean(ov)), "random_baseline": K / E,
                   **{f"reuse_within_{w}": float(np.mean(v)) for w, v in reuse.items()},
                   **{f"random_reuse_within_{w}": float(1 - (1 - K / E) ** w) for w in reuse}}

# 4. domain clustering (per-prompt usage vectors, all layers concatenated)
V = np.stack([(counts(d) / max(d.shape[1], 1)).ravel() for d in dec])
V = V / np.linalg.norm(V, axis=1, keepdims=True)
S = V @ V.T
same = [S[i, j] for i in range(n_prompts) for j in range(i + 1, n_prompts) if domains[i] == domains[j]]
diff = [S[i, j] for i in range(n_prompts) for j in range(i + 1, n_prompts) if domains[i] != domains[j]]
out["domain"] = {"cosine_within_domain": float(np.mean(same)), "cosine_across_domain": float(np.mean(diff))}

# 5. hybrid cache locality
fracs = [0.10, 0.25, 0.50, 0.75]
loc = {}
for f in fracs:
    m_keep = int(round(f * E))
    res = {"random": [], "global_freq": [], "domain_freq": [], "oracle_self": []}
    miss = {k: [] for k in res}
    for i, d in enumerate(dec):
        if d.shape[1] == 0: continue
        others = [j for j in range(n_prompts) if j != i]
        same_dom = [j for j in others if domains[j] == domains[i]] or others
        caches = {
            "random": np.stack([rng.permutation(E)[:m_keep] for _ in range(n_layers)]),
            "global_freq": np.argsort(-sum(counts(dec[j]) for j in others), 1)[:, :m_keep],
            "domain_freq": np.argsort(-sum(counts(dec[j]) for j in same_dom), 1)[:, :m_keep],
            "oracle_self": np.argsort(-counts(d), 1)[:, :m_keep],
        }
        for name, cache in caches.items():
            local = np.zeros((n_layers, E), bool)
            np.put_along_axis(local, cache, True, axis=1)
            hits = np.take_along_axis(local[:, None, :].repeat(d.shape[1], 1), d, axis=2)  # [layers, T, k]
            res[name].append(hits.all(2).mean())
            miss[name].append((~hits).sum(2).mean())
    loc[f] = {name: {"p_all_local": float(np.mean(v)), "mean_missed_experts": float(np.mean(miss[name]))} for name, v in res.items()}
out["hybrid_cache"] = loc

json.dump(out, open("results/locality_v1.json", "w"), indent=2)

md = ["# Routing structure & locality — Ling-3.0-tiny (v1)\n",
      f"{n_prompts} prompts, {out['decode_tokens']} decode tokens, {out['prefill_tokens']} prefill tokens, {n_layers} MoE layers, E={E}, k={K}. Source: traces/routing_v1.npz.\n",
      "## Usage skew (decode, all prompts pooled)\n",
      f"- Normalized entropy: mean {out['skew']['norm_entropy_mean']:.3f} (1.0 = uniform), min layer {out['skew']['norm_entropy_min']:.3f}",
      f"- Gini: mean {out['skew']['gini_mean']:.3f} (0 = uniform)",
      f"- Traffic share of top 10% experts: {out['skew']['top10pct_share_mean']:.1%} (uniform 10%); top 25%: {out['skew']['top25pct_share_mean']:.1%} (uniform 25%)",
      f"- Within one sequence, its own top 25% experts carry {out['per_sequence_top25pct_share_mean']:.1%} of its traffic\n",
      "## Temporal locality (decode)\n",
      f"- Adjacent-token overlap |S_t ∩ S_t+1|/k: {out['temporal']['adjacent_overlap']:.1%} (random {K/E:.1%})",
      *[f"- Selected expert already used within last {w} tokens: {out['temporal'][f'reuse_within_{w}']:.1%} (random {out['temporal'][f'random_reuse_within_{w}']:.1%})" for w in (1, 4, 16)],
      "\n## Domain clustering\n",
      f"- Cosine similarity of per-prompt expert usage: within domain {out['domain']['cosine_within_domain']:.3f}, across domains {out['domain']['cosine_across_domain']:.3f}\n",
      "## Hybrid cache: P(all 8 selected experts are local) per token-layer\n",
      "| fraction of experts held locally | random | global frequency (LOO) | same-domain frequency (LOO) | sequence's own (oracle) |", "|---|---|---|---|---|"]
for f in fracs:
    r = loc[f]
    md.append(f"| {f:.0%} | {r['random']['p_all_local']:.3f} | {r['global_freq']['p_all_local']:.3f} | {r['domain_freq']['p_all_local']:.3f} | {r['oracle_self']['p_all_local']:.3f} |")
md += ["\nMean missed experts per token-layer (of 8):\n", "| fraction | random | global | domain | oracle |", "|---|---|---|---|---|"]
for f in fracs:
    r = loc[f]
    md.append(f"| {f:.0%} | {r['random']['mean_missed_experts']:.2f} | {r['global_freq']['mean_missed_experts']:.2f} | {r['domain_freq']['mean_missed_experts']:.2f} | {r['oracle_self']['mean_missed_experts']:.2f} |")
open("results/locality_v1.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
