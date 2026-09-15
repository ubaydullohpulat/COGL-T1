# Roadmap

Source of truth for *status*. The frozen plan (`docs/plan/swarm-moe-research-plan.v1.md`) holds
the detailed method for each test; amendments live in `DECISIONS.md`.

---

## ▶ Current step

**Step 1 — Test 0: Napkin feasibility** (Phase 0) · `IN PROGRESS` (defined 2026-09-15, not yet executed)
Folder: `experiments/T00-feasibility/` · Target: ~1 week
Next action: **1.1 — pin the exact target model (K3), revision, tech report, config.json, license.**

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

Prerequisite: choose proxy model + hardware (open question, see DECISIONS backlog).

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
| L3 | K3 license + legal questions | Can change target model; legal advice before launch | NOT STARTED |
| L4 | Same-tokenizer draft model availability | Spec decoding (T7) depends on it | NOT STARTED |

---

## Near-term sequence (proposed)
1. **Step 1 — T0 Napkin feasibility** (this week) + L3 license read + L4 tokenizer check (both tiny, folded into 1.1)
2. **Step 2 — L1 prior-art sweep** (3–5 days, can overlap end of Step 1)
3. **Step 3 — Phase 1 setup**: pick proxy MoE + hardware, build routing-trace logger, assemble prompt corpus
4. Start **L2 RTT measurements** in the background as soon as Step 1 closes
