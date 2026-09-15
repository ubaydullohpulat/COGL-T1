"""SwarmLab v1 — replay REAL Ling-3.0-tiny routing with MEASURED M4 Max compute over emulated home links.

Topologies (N stage peers; the client = user's device holds embeddings + head):
  A_chain      layers split contiguously over N peers; every expert local; activations forwarded peer->peer.
  B_star       client holds all non-routed weights; routed experts split by id range over client + N peers;
               per MoE layer fan out in parallel to the holders of the selected experts.
  C_hybrid(f)  A_chain placement, but each stage keeps only a fraction f of each layer's experts (the most
               frequently used ones, chosen leave-one-prompt-out); the other experts live on a cold pool of
               M = N peers (split by id). A miss costs a parallel round trip from the stage to the cold holders.
Network model reused from S01 (`swarmlab.network`). Remote peers are assumed to compute as fast as this Mac.
Output: results/replay_v1.json, results/replay_v1.md
Run: python3 swarm_replay.py   (numpy only)
"""
import json, sys
import numpy as np

sys.path.insert(0, "../S01-swarmlab-v0")
from swarmlab.network import PROFILES, transit_s, serial_s  # noqa: E402

SEED = 20260916
rng = np.random.default_rng(SEED)
CFG = json.load(open("../P00-proxy-selection/configs/inclusionAI__Ling-3.0-tiny.json"))
E, K, H = CFG["num_experts"], CFG["num_experts_per_tok"], CFG["hidden_size"]
N_LAYERS, FIRST_MOE = CFG["num_hidden_layers"], CFG["first_k_dense_replace"]
GROUP = CFG["layer_group_size"]
IS_KDA = [not ((i + 1) % GROUP == 0 or i >= N_LAYERS // GROUP * GROUP) for i in range(N_LAYERS)]
REMOTE_CALL_OVERHEAD_S = 0.002        # ASSUMED: (de)serialize + dispatch per remote expert call
MSG_BYTES = H * 2 + 64                # one bf16 hidden vector + header

# ---------- measured compute ----------
M = json.load(open("results/compute_v1.json"))
ov = M["eval_overhead_s"]
c = {k: max(v - ov, 0.0) for k, v in M["decode_components_median_s"].items()}
vs_k = {int(k): max(v - ov, 0.0) for k, v in M["routed_experts_vs_k_s"].items()}
def experts_s(k):
    if k <= 0: return 0.0
    xs = sorted(vs_k); return float(np.interp(k, xs, [vs_k[x] for x in xs]))
def layer_local_s(i, k_local):
    attn = c["attention_kda"] if IS_KDA[i] else c["attention_mla"]
    if i < FIRST_MOE:
        return attn + c["dense_mlp"]
    return attn + c["router"] + c["shared_expert"] + experts_s(k_local)
raw = sum(layer_local_s(i, K) for i in range(N_LAYERS)) + c["head"]
SCALE = (1 / M["decode_tok_per_s"]) / raw     # calibrate component sum to the measured end-to-end decode
def S(x): return x * SCALE

# ---------- traces ----------
Z = np.load("traces/routing_v1.npz")
domains = list(Z["domains"])
dec = [Z[f"p{i:02d}_decode"].astype(np.int64) for i in range(len(domains))]
moe_layers = list(Z["moe_layers"])


def counts(tr):
    out = np.zeros((tr.shape[0], E))
    for l in range(tr.shape[0]):
        out[l] = np.bincount(tr[l].ravel(), minlength=E)
    return out


def rtt(profile, n):
    p = PROFILES[profile]
    return (transit_s(p, MSG_BYTES, rng, n) * 2 + serial_s(MSG_BYTES, p.up_mbps) + serial_s(MSG_BYTES, p.up_mbps))


def stages(n_peers):
    return np.array_split(np.arange(N_LAYERS), n_peers)


def replay(topology, profile, n_peers, frac=1.0, cache_kind="global_freq"):
    lat = []
    for pi, tr in enumerate(dec):
        T = tr.shape[1]
        if T == 0: continue
        t = np.zeros(T)
        p = PROFILES[profile]
        if topology in ("A_chain", "C_hybrid"):
            hops = n_peers + 1
            for _ in range(hops):
                t += transit_s(p, MSG_BYTES, rng, T) + serial_s(MSG_BYTES, p.up_mbps)
            t += S(c["head"]) + REMOTE_CALL_OVERHEAD_S * n_peers
            if topology == "C_hybrid":
                others = [j for j in range(len(dec)) if j != pi]
                pool = [j for j in others if domains[j] == domains[pi]] if cache_kind == "domain_freq" else others
                freq = sum(counts(dec[j]) for j in pool)
                keep = int(round(frac * E))
            for li in range(N_LAYERS):
                if li < FIRST_MOE:
                    t += S(layer_local_s(li, 0)); continue
                sel = tr[moe_layers.index(li)]                           # [T, k]
                if topology == "A_chain":
                    t += S(layer_local_s(li, K)); continue
                order = np.argsort(-freq[moe_layers.index(li)])
                local = np.zeros(E, bool); local[order[:keep]] = True
                cold_ids = order[keep:]                                    # cold experts, split by rank over N holders
                holder = np.full(E, -1); holder[cold_ids] = np.arange(len(cold_ids)) * n_peers // max(len(cold_ids), 1)
                is_local = local[sel]                                      # [T, k]
                k_loc = is_local.sum(1)
                t += S(np.array([layer_local_s(li, k) for k in range(K + 1)]))[k_loc]
                wait = np.zeros(T)
                for h in range(n_peers):
                    k_h = ((~is_local) & (holder[sel] == h)).sum(1)
                    hit = k_h > 0
                    if not hit.any(): continue
                    w = rtt(profile, T) + S(np.array([experts_s(k) for k in range(K + 1)]))[k_h] + REMOTE_CALL_OVERHEAD_S
                    wait = np.maximum(wait, hit * w)
                t += wait
        elif topology == "B_star":
            holders = n_peers + 1
            t += S(c["head"])
            for li in range(N_LAYERS):
                if li < FIRST_MOE:
                    t += S(layer_local_s(li, 0)); continue
                sel = tr[moe_layers.index(li)]
                owner = sel * holders // E
                k0 = (owner == 0).sum(1)
                t += S(np.array([layer_local_s(li, k) for k in range(K + 1)]))[k0]
                wait = np.zeros(T)
                for h in range(1, holders):
                    k_h = (owner == h).sum(1)
                    w = rtt(profile, T) + S(np.array([experts_s(k) for k in range(K + 1)]))[k_h] + REMOTE_CALL_OVERHEAD_S
                    wait = np.maximum(wait, (k_h > 0) * w)
                t += wait
        lat.append(t)
    lat = np.concatenate(lat)
    return {"tok_per_s": float(1 / lat.mean()), "mean_s": float(lat.mean()), "p95_s": float(np.percentile(lat, 95))}


def memory_gb(topology, n_peers, frac=1.0):
    """4.5-bit weights. Returns (per stage/expert peer GB, cold pool peer GB)."""
    params_total = 7_893_392_800                                     # CITED: HF safetensors total
    per_expert = 3 * H * CFG["moe_intermediate_size"]
    routed = len(moe_layers) * E * per_expert
    gb = lambda n: n * 4.5 / 8 / 1e9
    if topology == "A_chain": return gb(params_total) / n_peers, 0.0
    if topology == "B_star": return gb(routed) / (n_peers + 1), 0.0
    return gb(params_total - (1 - frac) * routed) / n_peers, gb((1 - frac) * routed) / n_peers


profiles = ["lan", "metro", "national", "researcher", "intercontinental"]
N = 4
rows = []
for prof in profiles:
    rows.append({"topology": "A_chain", "profile": prof, "frac": 1.0, **replay("A_chain", prof, N)})
    rows.append({"topology": "B_star", "profile": prof, "frac": None, **replay("B_star", prof, N)})
    for f in [0.25, 0.5, 0.75, 0.9]:
        rows.append({"topology": "C_hybrid", "profile": prof, "frac": f, "cache": "global_freq", **replay("C_hybrid", prof, N, f)})
    rows.append({"topology": "C_hybrid", "profile": prof, "frac": 0.5, "cache": "domain_freq", **replay("C_hybrid", prof, N, 0.5, "domain_freq")})

meta = {"seed": SEED, "n_peers": N, "calibration_scale": SCALE, "measured_decode_tok_per_s": M["decode_tok_per_s"],
        "remote_call_overhead_s": REMOTE_CALL_OVERHEAD_S, "decode_tokens": int(sum(d.shape[1] for d in dec))}
json.dump({"meta": meta, "rows": rows}, open("results/replay_v1.json", "w"), indent=2)

def row(topo, prof, f=None, cache="global_freq"):
    return next(r for r in rows if r["topology"] == topo and r["profile"] == prof and (f is None or r["frac"] == f)
                and (topo != "C_hybrid" or r.get("cache") == cache))
md = ["# SwarmLab v1 replay — Ling-3.0-tiny, real routing + measured compute, 4 stage peers\n",
      f"Seed {SEED}; {meta['decode_tokens']} real decode tokens replayed; single-machine decode {M['decode_tok_per_s']:.1f} tok/s (MEASURED); "
      f"component calibration ×{SCALE:.2f}; remote call overhead {REMOTE_CALL_OVERHEAD_S*1e3:.0f} ms (ASSUMED). Links from S01 (ASSUMED except researcher).\n",
      "## Decode tok/s (mean)\n",
      "| setup | memory per stage peer | cold-pool peer | " + " | ".join(f"{p} ({PROFILES[p].rtt_ms:.0f} ms)" for p in profiles) + " |",
      "|---|---|---|" + "---|" * len(profiles)]
setups = [("A_chain (all experts local)", "A_chain", None, "global_freq"), ("B_star (experts spread)", "B_star", None, "global_freq")]
setups += [(f"C_hybrid, keep {int(f*100)}% hot experts", "C_hybrid", f, "global_freq") for f in [0.9, 0.75, 0.5, 0.25]]
setups += [("C_hybrid, keep 50%, domain-aware cache", "C_hybrid", 0.5, "domain_freq")]
for label, topo, f, cache in setups:
    mem = memory_gb(topo, N, f if f is not None else 1.0)
    md.append(f"| {label} | {mem[0]:.2f} GB | {mem[1]:.2f} GB | " + " | ".join(f"{row(topo, p, f, cache)['tok_per_s']:.2f}" for p in profiles) + " |")
md += ["\n## p95 seconds per token at `researcher` (137 ms)\n", "| setup | mean s | p95 s |", "|---|---|---|"]
for label, topo, f, cache in setups:
    r = row(topo, "researcher", f, cache); md.append(f"| {label} | {r['mean_s']:.3f} | {r['p95_s']:.3f} |")
open("results/replay_v1.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
