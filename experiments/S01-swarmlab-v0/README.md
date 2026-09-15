# S01 — SwarmLab v0: Ling-3.0-flash over a few home-internet peers

**Dates:** 2026-09-15/16 · **Status:** DONE (v0) · **Code:** `swarmlab/`, `run_v0.py`, `test_sanity.py` · **Results:** `results/v0_summary.md`, `results/*.csv`

## Question
If Ling-3.0-flash (127B, 512 experts top-8, 42 layers) is split across 2–8 home machines, how does it behave
(decode tok/s, tail latency, time-to-first-token, memory per machine) under home-internet latency, and which split works?

## Method (v0)
Monte Carlo emulator, numpy only, seed 20260916, 2,000 sampled tokens per cell.
- **Architecture:** `CITED` from Ling-3.0-flash config @ e0dfe7c and HF parameter total (127,486,405,600). Routed experts = 41 layers (40 MoE + 1 MTP) × 512 × 5.9M = 123.8B; non-routed = 3.7B (`COMPUTED`).
- **Links:** RTT with lognormal jitter + Pareto spikes + per-packet loss → TCP RTO stall + uplink serialization. Profiles `ASSUMED`; `researcher` uses the measured 137 ms / 30 up / 72 down (to an Apple server, not peer-to-peer).
- **Compute:** memory-bandwidth-bound decode + per-call overhead, FLOPs-bound prefill, all `ASSUMED` (client M4 Max, peers RTX-4070-class).
- **Routing:** `SYNTHETIC` uniform random top-8. That is the *worst case* for expert sharding (no locality).
- **Topologies:**
  - `A_chain`: pipeline, peer→peer forwarding
  - `A_relay`: pipeline, client relays each hop (Petals-style)
  - `B_star`: client holds attention/shared/router; routed experts split over peers + client; fan-out to peers every MoE layer
- Sanity checks against hand-computed cases: `python3 test_sanity.py` (6/6 pass).

## Results (headline; full tables in `results/v0_summary.md`)
Decode tok/s, remote peers = 4 (single machine, no network: 109 tok/s ASSUMED):

| topology | LAN 1 ms | metro 20 ms | national 60 ms | researcher 137 ms | intercontinental 250 ms |
|---|---|---|---|---|---|
| A_chain | 41.5 | 11.6 | 4.8 | **2.4** | 1.3 |
| A_relay | 38.9 | 8.0 | 3.1 | 1.5 | 0.8 |
| B_star | 5.8 | 0.66 | 0.24 | **0.12** | 0.06 |

- 2 peers at 137 ms: A_chain 3.9 tok/s (35 GB per peer); 8 peers: 1.3 tok/s (8.8 GB per peer).
- Time to first token, 4 peers at 137 ms, 1k-token prompt: A_chain 8.9 s (bf16 wire) → 5.4 s (int8); B_star 252 s.
- p95 token latency ≈ 1.4–1.7 × mean for A_chain (jitter + loss stalls).
- One slow laptop among 4 peers: no visible effect (compute is ~5% of token time; network dominates).

## What it means
1. **Over home internet, decode time = hops × one-way latency.** Compute is ~5% of token time, so fewer sequential network crossings beats faster GPUs.
2. **Naive expert sharding is 20–30× slower than a pipeline.** B_star pays a round trip in all 40 MoE layers per token, so ~0.12 tok/s at 137 ms. This is the same wall T0 predicted for K3, now shown concretely for Ling. **The project's core bet only works if most layers need no remote round trip**: experts must be co-located by layer and hot experts cached, so cross-peer fan-out is rare. That is exactly what real routing traces (Phase 1) must show.
3. **A few-peer pipeline is usable today** for a Ling-flash-sized model: 2–4 tok/s with 2–4 peers at ~137 ms, ~5–12 tok/s within a country/city.
4. **Prefill is bandwidth-bound:** a 30 Mbps uplink makes each hop cost ~1.4 s per 1k tokens at bf16. int8 activations nearly halve it (T3 matters). Star prefill is hopeless.

## Caveats (do not over-read)
- Routing synthetic (worst case for B); no hot-expert caching; no hybrid topology yet.
- Compute and link profiles assumed; the `researcher` RTT is to a CDN, not to another home peer.
- TCP model is simple (one stall per lost message, no congestion window); single user, no batching; KDA/MLA state transfer not modelled.
- No speculative decoding. Ling-3.0-flash ships an MTP layer that could draft tokens and verify them in one pass, multiplying pipeline tok/s.

## Next (v1)
1. Measure real compute per layer and real routing traces on **Ling-3.0-tiny** (same architecture, MLX 4-bit 4.5 GB), then replay them in SwarmLab.
2. Add topology **C_hybrid**: pipeline stages own whole layers and keep hot experts local; fetch cold experts remotely only on a miss. Measure P(all top-k local) from real traces.
3. Add **MTP speculative decoding** to A_chain / C_hybrid.
4. Validate the emulator against a real multi-process run on localhost with injected delays (mini T9 gate: error < 30%).
