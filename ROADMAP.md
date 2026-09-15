# Roadmap

Source of truth for *status*. The frozen plan (`docs/plan/swarm-moe-research-plan.v1.md`) holds
the detailed method for each test; amendments live in `DECISIONS.md`.

---

## ▶ Current step

**S01 — SwarmLab: few-peer swarm emulator on the Ling-3.0 family** · v0 ✅ done 2026-09-16 · v1 next (D-006)
Folder: `experiments/S01-swarmlab-v0/` · Headline (simulated): 4-peer pipeline @137 ms ≈ 2.4 tok/s; naive expert-sharded star ≈ 0.12 tok/s
Next action: **S01 v1**. Install MLX, get Ling-3.0-tiny (needs P-10 approval), measure real per-layer compute + routing traces, replay in SwarmLab, add hybrid topology + MTP speculative decoding.
Paused: T0 step 1.2 (K3) and K3S spike (D-006).

---

## The project in one picture

```
 Phase 0  Arithmetic ──► GATE: is it physically possible?
    │
 Phase 1  Routing structure (proxy MoE) ──► GATE: does routing have exploitable structure?
    │         └─ produces: routing-trace dataset  (publishable asset)
 Phase 2  Simulator ──► GATE: any path below 5 s/token?
    │         └─ produces: simulator + placement paper  (paper #1)
 Phase 3  Real prototype, small model ──► GATE: simulator within 30% of reality
    │
 Phase 4  Scale to K3  ◄──► Phase 5  Trust / privacy / abuse (overlaps)
    │         └─ produces: K3 routing results  (paper #2)          └─ ETHICAL GATE: Test 15
 Phase 6  Client + launch ──► GATE: real volunteer capacity sufficient (Test 18)

 Parallel long-lead tracks (start early, run in background):
   L1 prior-art sweep · L2 consumer RTT measurements · L3 license + legal · L4 draft-model availability
```

## Workstreams and tests

### Phase 0 — Arithmetic (≈1 week)
| ID | Test | Question | Kill gate | Status |
|---|---|---|---|---|
| T0 | Napkin feasibility | Best-case tok/s from architecture + RTT | < 0.1 tok/s optimistic → stop / batch-only | IN PROGRESS |

### Phase 1 — Routing structure on a proxy MoE (6–8 weeks)
| ID | Test | Question | Kill gate | Status |
|---|---|---|---|---|
| T1 | Routing structure | Skewed usage? temporal locality? domain clustering? | near-uniform AND no autocorrelation → thesis dead | NOT STARTED |
| T2 | Next-layer predictability | Probe recall of layer L+1 experts from layer L state | recall ≈ chance | NOT STARTED |
| T3 | Boundary precision | fp8 / int8 / int4 activations on the wire | — (sets bandwidth) | NOT STARTED |

Proxies per D-005: dev on Ling-3.0-tiny → primary Ling-3.0-flash, LatentMoE check on Nemotron-3-Super, family check on Kimi Linear, transfer check vs streamed K3 layers. Peers simulated as logical partitions of one loaded model. Open: P-6, P-7, P-8.

### Phase 2 — Simulator (6–8 weeks)
| ID | Test | Question | Kill gate | Status |
|---|---|---|---|---|
| T4 | Latency surface | tok/s over hops × RTT with jitter | — | NOT STARTED |
| T5 | Placement vs random | Co-activation partitioning; sweeps slice size (answers §0) | < 10% gain over random | NOT STARTED |
| T6 | Churn resilience | Replication factor R for 99% completion | — (sets launch peer count) | NOT STARTED |
| T7 | Speculative decoding | Effective tok/s vs draft length & acceptance | no path < 5 s/token | NOT STARTED |

### Phase 3 — Real prototype, small model (8–10 weeks)
| ID | Test | Question | Kill gate | Status |
|---|---|---|---|---|
| T8 | Two-process correctness | Matches single-machine output | must pass | NOT STARTED |
| T9 | 10-node LAN | Throughput; simulator error < 30% | sim badly wrong → fix sim first | NOT STARTED |
| T10 | 10-node WAN (VPS) | Degradation matches RTT model | — | NOT STARTED |

### Phase 4 — Scale to K3 (10–14 weeks)
| ID | Test | Question | Kill gate | Status |
|---|---|---|---|---|
| T11 | First correct token | Shard ~1.4 TB, match reference | milestone | NOT STARTED |
| T12 | Re-run T1, T2, T5 on K3 | Does proxy structure transfer? | retune if not | NOT STARTED |
| T13 | Heterogeneity / stragglers | MLX + CUDA + CPU mix; re-dispatch tradeoff | — | NOT STARTED |

### Phase 5 — Trust, privacy, abuse (8 weeks, overlaps Phase 4)
| ID | Test | Question | Kill gate | Status |
|---|---|---|---|---|
| T14 | Malicious peers | Detection rate vs spot-check overhead | — | NOT STARTED |
| T15 | Activation inversion | Prompt recovery from boundary activations | **ethical gate** | NOT STARTED |
| T16 | Sybil resistance | Cost to fake vs earn contribution | — | NOT STARTED |

### Phase 6 — Client and launch (6+ months)
| ID | Test | Question | Kill gate | Status |
|---|---|---|---|---|
| T17 | NAT traversal | Direct-connect success under CGNAT | relay economics | NOT STARTED |
| T18 | Closed alpha (50 volunteers) | Real capacity, sessions, churn | insufficient capacity → network can't exist | NOT STARTED |
| T19 | Public launch + seed cluster | Useful at zero peers | — | NOT STARTED |

### Parallel long-lead tracks
| ID | Track | Why start now | Status |
|---|---|---|---|
| L1 | Prior-art sweep (`references/reading-list.md`) | T1/T2/T5 may be partly answered; protects novelty claim | NOT STARTED |
| L2 | Consumer RTT measurements, target geography | T4 input; needs calendar time | NOT STARTED |
| L3 | K3 license + legal questions | Can change target model; legal advice before launch | IN PROGRESS — license read 2026-09-15 (no blocker for non-commercial); legal review still due before launch |
| P0 | Proxy selection (how to test without serving K3) | Every Phase 1–3 result depends on it | DONE — recommendation 2026-09-15, awaiting P-8 |
| S01 | SwarmLab emulator (few peers, home latency, Ling-3.0) | Concrete behaviour now; feeds T4/T5/T7/T9 | IN PROGRESS — v0 done 2026-09-16 |
| K3S | K3 layer-streaming spike (load + run real K3 layers one shard at a time) | Only local source of real K3 routing/activations | PAUSED (D-006) |
| L4 | Same-tokenizer draft model availability | Spec decoding (T7) depends on it | IN PROGRESS — same tokenizer: Kimi Linear 48B-A3B, Moonlight 16B-A3B (no 1–3B dense); chat-token ID alignment unchecked |

---

## Near-term sequence (proposed)
1. **Step 1 — T0 Napkin feasibility** (this week) + L3 license read + L4 tokenizer check (both tiny, folded into 1.1)
2. **Step 2 — L1 prior-art sweep** (3–5 days, can overlap end of Step 1)
3. **Step 3 — Phase 1 setup**: pick proxy MoE + hardware, build routing-trace logger, assemble prompt corpus
4. Start **L2 RTT measurements** in the background as soon as Step 1 closes
5. **K3S spike** (after P-7 approval): stream K3 layers 0–3, verify MLX correctness, measure per-layer compute time (also feeds T0)
