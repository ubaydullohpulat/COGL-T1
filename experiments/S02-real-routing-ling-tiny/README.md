# S02 — Real routing + measured compute on Ling-3.0-tiny, replayed over home-internet links (SwarmLab v1)

**Date:** 2026-09-16 · **Status:** DONE · **Includes:** T1-lite (routing structure on a proxy) + SwarmLab v1 replay with hybrid topology

## Questions
1. Does Ling-3.0 routing have exploitable structure (skew, temporal locality, domain clustering)? (T1-lite, kill gate)
2. With *real* routing and *measured* compute, can a hybrid split (stages keep hot experts, cold experts remote) approach
   a pipeline's speed while cutting memory per peer?

## Setup (reproducibility)
| Item | Value |
|---|---|
| Model | `rapid-mlx/Ling-3.0-tiny-MLX-4bit` @ `328a497f…` (base `inclusionAI/Ling-3.0-tiny`, 7.9B, 128 experts top-8, 23 MoE layers, 4-bit affine g64); sha256 `be523d3d…` |
| Model fix | `fix_kv_b_proj.py`: the community build used a different MLX port. Renamed 261 tensors (`mlp.experts.*`→`mlp.switch_mlp.*`, `*_conv1d.conv.weight`→`*_conv1d.weight`) and dequantized 6 `kv_b_proj` tensors to bf16. Output `models/Ling-3.0-tiny-MLX-4bit-kvbfix` (git-ignored) |
| Software | mlx 0.32.2, mlx-lm git `872ae88d` (PyPI 0.31.3 lacks `bailing_moe_v3`), Python 3.11.13; `requirements.txt` |
| Hardware | Apple M4 Max, 128 GB |
| Prompts | `prompts.py` v1: 24 prompts, 9 domains (code, en_tech, en_prose, ru, uz, zh, math, chat) |
| Sampling | temp 0.7, top-p 0.9, seed 20260916, max 256 new tokens, chat template |
| Scripts | `bench_compute.py` → `results/compute_v1.json`; `trace_ling.py` → `traces/routing_v1.npz` (git-ignored, 1.3 MB) + `results/generations_v1.jsonl`; `analyze_traces.py` → `results/locality_v1.{json,md}`; `swarm_replay.py` → `results/replay_v1.{json,md}` |

Smoke test: coherent answers in English and Russian (the model opens with a reasoning preamble). `mlx_lm.generate` reports 189–205 tok/s, peak memory 4.6 GB.

## 1. Measured compute (M4 Max, MLX 4-bit)
- Prefill 2.5k tok/s (128-token prompt) → ~4.0–4.2k tok/s (512–4096 tokens)
- Decode 163 tok/s in a plain Python loop (≈6.1 ms/token)
- Per component per layer, decode: KDA attention 0.26 ms, MLA 0.30 ms, router 0.22 ms, routed experts 0.20 ms, shared expert 0.14 ms, head 0.40 ms. These include one synced eval each, so the sum is ×1.85 the end-to-end time; the replay calibrates by ×0.54.
- Routed-expert cost for one token is flat in k (0.29–0.32 ms for k = 2…8): kernel dispatch dominates, not FFN math.

## 2. Routing structure (T1-lite) — 6,136 decode tokens
| Metric | Ling-3.0-tiny | Uniform-random baseline |
|---|---|---|
| Traffic share of top 10% / 25% experts (pooled) | 29.0% / 56.9% | 10% / 25% |
| Normalized entropy (mean over layers) | 0.928 | 1.0 |
| Gini (mean) | 0.458 | 0 |
| Within one sequence: its own top 25% experts carry | 80.0% | 25% |
| Adjacent-token expert overlap | 43.8% | 6.2% |
| Selected expert already used within last 4 / 16 tokens | 65.0% / 86.2% | 22.8% / 64.4% |
| Usage cosine similarity: within domain vs across | 0.698 vs 0.382 | — |

**Gate verdict (T1 kill criterion, proxy, preliminary): PASS.** Usage is clearly skewed and temporal locality is ~7× random. Neither kill condition holds.
Limits: one small proxy (128 experts), 24 prompts, decode dominated by the model's reasoning preamble (often English even for ru/uz prompts).

## 3. Hybrid cache locality — P(all 8 selected experts local) per token-layer
| experts kept locally | random | global frequency (leave-one-out) | same-domain frequency | sequence's own (oracle) |
|---|---|---|---|---|
| 25% | 0.000 | 0.047 | 0.101 | 0.282 |
| 50% | 0.003 | 0.338 | 0.493 | 0.808 |
| 75% | 0.090 | 0.641 | 0.823 | 0.990 |

## 4. SwarmLab v1 replay — 4 stage peers, decode tok/s
| setup | GB per stage | lan | metro 20 ms | national 60 ms | researcher 137 ms | intercont. 250 ms |
|---|---|---|---|---|---|---|
| A_chain (all experts local) | 1.11 | 59.6 | 13.3 | 5.15 | **2.48** | 1.32 |
| B_star (experts spread) | 0.78 | 12.4 | 1.27 | 0.44 | **0.21** | 0.11 |
| C_hybrid keep 90% | 1.01 | 33.1 | 5.22 | 1.93 | **0.91** | 0.48 |
| C_hybrid keep 75% | 0.87 | 23.0 | 3.15 | 1.15 | 0.55 | 0.28 |
| C_hybrid keep 50% | 0.62 | 15.1 | 1.88 | 0.67 | 0.32 | 0.16 |
| C_hybrid keep 50%, domain-aware | 0.62 | 18.4 | 2.45 | 0.89 | 0.42 | 0.22 |

p95 at 137 ms: A_chain 0.58 s/token; C_hybrid-90% 3.1 s; B_star 5.8 s.

## What it means
1. **Routing structure is real** (skew, temporal locality, domain clustering): the thesis survives its first gate on a proxy.
2. **But static hot-expert placement does not rescue expert sharding over home internet.** Even keeping 90% of experts on
   the stage (saving only 10% memory) costs 2.7× throughput at 137 ms, and p95 jumps from 0.6 s to 3.1 s.
   Every layer that misses even one of its 8 experts pays a full round trip, and with 23 layers most tokens hit several.
3. **Budget rule** (`COMPUTED` from the replay): at 137 ms a layer with ≥1 remote miss costs 0.167–0.180 s, while the chain
   costs 0.403 s/token. Losing ≤ 25% of pipeline throughput allows 0.134 s of misses, ≈ 0.8 miss-layers per token.
   That means P(layer miss) ≤ ~3.5% for 23 MoE layers, ≤ ~2.0% for Ling-3.0-flash (40), ≤ ~0.9% for K3 (92), assuming similar chain cost.
   Static caches are nowhere near that:

   | experts kept (global-frequency cache) | P(layer has a miss) | miss-layers / token | extra s / token @137 ms |
   |---|---|---|---|
   | 90% | 18.2% | 4.2 | 0.70 |
   | 75% | 35.9% | 8.3 | 1.43 |
   | 50% | 66.2% | 15.2 | 2.74 |
4. **Where the headroom is:** sequence-specific caches are far better than global ones (50% kept: 0.81 vs 0.34 all-local).
   With 86% of experts reused within 16 tokens, placement should adapt *per session*, not per model.

## Ideas this points to (not yet tested)
- **Expert paging:** on a miss, pull the expert's *weights* to the stage (Ling-tiny ≈ 1.3 MB, flash ≈ 3.3 MB at 4-bit) and keep them via LRU. Temporal locality amortizes the fetch.
- **Routing-aware session scheduling:** pick, per session, the replica whose expert subset best matches the prefill routing.
- **Miss-tolerant execution:** substitute or skip a missing low-weight expert. Lossy, so quality impact must be measured.
- **Batch misses across speculative tokens:** one round trip covers several drafted tokens.

## Caveats
- Proxy is tiny (128 experts); larger expert counts may behave differently (the granularity ladder is still to do).
- Remote peers assumed as fast as the M4 Max; remote call overhead 2 ms assumed; links assumed (S01).
- Cold pool = separate peers holding cold experts split by frequency rank; misses wait for the slowest holder.
- The oracle column uses the sequence's own future routing, so it's an upper bound, not a policy.
