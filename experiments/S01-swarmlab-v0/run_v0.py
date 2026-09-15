"""SwarmLab v0 sweep for Ling-3.0-flash. Writes results/v0_*.csv and results/v0_summary.md.
Run: python3 run_v0.py   (seed fixed; ~1 min on M4 Max)
"""
import csv, json, platform, time
import numpy as np

from swarmlab import arch, sim
from swarmlab.compute import COMPUTE
from swarmlab.network import PROFILES, LinkProfile

SEED = 20260916
N_TOKENS = 2000
CFG = "../P00-proxy-selection/configs/inclusionAI__Ling-3.0-flash.json"
LING = arch.load(CFG, params_total=127_486_405_600, name="Ling-3.0-flash")  # CITED: HF safetensors total @ e0dfe7c
TOPOS = ["A_chain", "A_relay", "B_star"]
PEERS = [2, 3, 4, 8]
CLIENT = COMPUTE["m4max"]
PEER = COMPUTE["consumer_gpu"]
rng = np.random.default_rng(SEED)
t0 = time.time()

rows = []
zero = LinkProfile("none", 0, 0, 0, 0, 0, 1e9, 1e9, "no network")
local = sim.decode_latency(sim.Scenario(LING, "A_chain", 1, zero, [CLIENT], CLIENT), N_TOKENS, rng)
rows.append({"topology": "single_machine", "n_peers": 0, "profile": "none", **local})

for prof in PROFILES.values():
    for topo in TOPOS:
        for n in PEERS:
            s = sim.Scenario(LING, topo, n, prof, [PEER] * n, CLIENT)
            r = sim.decode_latency(s, N_TOKENS, rng)
            rows.append({"topology": topo, "n_peers": n, "profile": prof.name, **r, **sim.memory_per_holder_gb(s)})

with open("results/v0_decode.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader(); w.writerows(rows)

# straggler: one slow peer among 4, researcher-like links
strag = []
for topo in TOPOS:
    for label, peers in [("4x consumer_gpu", [PEER] * 4), ("3x consumer_gpu + 1 slow_laptop", [PEER] * 3 + [COMPUTE["slow_laptop"]])]:
        r = sim.decode_latency(sim.Scenario(LING, topo, 4, PROFILES["researcher"], peers, CLIENT), N_TOKENS, rng)
        strag.append({"topology": topo, "peers": label, **r})

# prefill
pre = []
for pname in ["metro", "researcher", "intercontinental"]:
    for topo in TOPOS:
        for P in [1000, 4000]:
            for bpv, prec in [(2.0, "bf16"), (1.0, "int8")]:
                s = sim.Scenario(LING, topo, 4, PROFILES[pname], [PEER] * 4, CLIENT, bytes_per_value=bpv)
                pre.append({"profile": pname, "topology": topo, "prompt": P, "wire": prec, **sim.prefill_latency(s, P, rng)})
with open("results/v0_prefill.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pre[0])); w.writeheader(); w.writerows(pre)

# summary markdown
prof_names = list(PROFILES)
def cell(topo, n, p):
    r = next(r for r in rows if r["topology"] == topo and r["n_peers"] == n and r["profile"] == p)
    return f'{r["tok_per_s"]:.2f}'
md = [f"# SwarmLab v0 results — Ling-3.0-flash\n",
      f"Seed {SEED}, {N_TOKENS} sampled tokens per cell, python {platform.python_version()}, numpy {np.__version__}, runtime {time.time()-t0:.0f}s.",
      "Routing SYNTHETIC (uniform random = no locality). Compute ASSUMED (client m4max, peers consumer_gpu). Links ASSUMED except `researcher` (partly MEASURED).\n",
      f"Single machine, no network (ASSUMED compute): **{local['tok_per_s']:.1f} tok/s**\n",
      "## Decode throughput (tok/s, mean) — rows: topology × remote peers; columns: link profile (median RTT)\n",
      "| topology | peers | " + " | ".join(f"{p} ({PROFILES[p].rtt_ms:.0f} ms)" for p in prof_names) + " |",
      "|---|---|" + "---|" * len(prof_names)]
for topo in TOPOS:
    for n in PEERS:
        md.append(f"| {topo} | {n} | " + " | ".join(cell(topo, n, p) for p in prof_names) + " |")
md += ["\n## p95 token latency (s) at `researcher` profile\n", "| topology | peers | mean s | p95 s | compute share |", "|---|---|---|---|---|"]
for topo in TOPOS:
    for n in PEERS:
        r = next(r for r in rows if r["topology"] == topo and r["n_peers"] == n and r["profile"] == "researcher")
        md.append(f'| {topo} | {n} | {r["mean_s"]:.3f} | {r["p95_s"]:.3f} | {r["compute_share"]:.1%} |')
md += ["\n## Memory per holder (GB, 4.5-bit weights)\n", "| topology | peers | client GB | each peer GB |", "|---|---|---|---|"]
for topo in TOPOS:
    for n in PEERS:
        r = next(r for r in rows if r["topology"] == topo and r["n_peers"] == n and r["profile"] == "lan")
        md.append(f'| {topo} | {n} | {r["client_gb"]:.1f} | {r["peer_gb"]:.1f} |')
md += ["\n## Straggler: 4 peers, `researcher` links\n", "| topology | peers | tok/s | p95 s |", "|---|---|---|---|"]
md += [f'| {r["topology"]} | {r["peers"]} | {r["tok_per_s"]:.2f} | {r["p95_s"]:.3f} |' for r in strag]
md += ["\n## Prefill, 4 peers — time to first token (s, mean)\n", "| profile | topology | prompt | bf16 wire | int8 wire |", "|---|---|---|---|---|"]
for pname in ["metro", "researcher", "intercontinental"]:
    for topo in TOPOS:
        for P in [1000, 4000]:
            b = next(r for r in pre if r["profile"] == pname and r["topology"] == topo and r["prompt"] == P and r["wire"] == "bf16")
            i = next(r for r in pre if r["profile"] == pname and r["topology"] == topo and r["prompt"] == P and r["wire"] == "int8")
            md.append(f'| {pname} | {topo} | {P} | {b["ttft_mean_s"]:.1f} | {i["ttft_mean_s"]:.1f} |')
md += ["\n## Link profiles\n", "| name | RTT ms | jitter σ | loss | up/down Mbps | evidence |", "|---|---|---|---|---|---|"]
md += [f"| {p.name} | {p.rtt_ms} | {p.jitter_sigma} | {p.loss} | {p.up_mbps}/{p.down_mbps} | {p.evidence} |" for p in PROFILES.values()]
open("results/v0_summary.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
