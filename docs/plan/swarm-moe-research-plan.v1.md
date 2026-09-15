# P2P Swarm Inference for Frontier MoE Models — Research & Build Plan

**End goal:** open-source client. Any user with a consumer GPU or Apple M-series chip downloads it, joins a network, holds a slice of a ~2.8T-parameter MoE model, and gets to use the full model for as long as they are contributing.

**Core bet:** sparse MoE + expert-level sharding makes frontier-scale models tractable on a volunteer network, in a way that dense models never were.

---

## 0. Design parameter you need to settle first

You said ~1B params per peer. Work through what that implies:

| Slice size / peer | MXFP4 footprint | Peers per full replica | Peers at 4x redundancy |
|---|---|---|---|
| 1B | ~0.5 GB | ~2,800 | ~11,200 |
| 4B | ~2 GB | ~700 | ~2,800 |
| 16B | ~8 GB | ~175 | ~700 |
| 32B | ~16 GB | ~88 | ~350 |

Smaller slices = lower barrier to joining, but more peers touched per token, so more network hops and worse latency. Larger slices = fast, but excludes most consumer hardware.

**This tradeoff is the central design decision of the project** and it should be an output of Test 5, not an assumption. My prior is that the answer is tiered: peers declare capacity, the scheduler assigns 1–32B accordingly, and small peers get assigned cold/rare experts while large peers get hot ones.

---

## PHASE 0 — Arithmetic (1 week, no code)

### Test 0 — Napkin feasibility
**Question:** Is the target physically possible before writing anything?

**Method:** From the K3 technical report, extract: layer count, expert count per layer, top-k, expert dimensions, shared-expert structure, attention param count, hidden dim. Compute by hand:
- active params per token
- bytes on the wire per layer boundary (hidden_dim × precision)
- minimum sequential hops per token
- theoretical tok/s at 50ms, 150ms, 300ms per-hop RTT

**Should show:** a best-case tok/s figure. If best case at 150ms RTT is above ~0.3 tok/s, continue.

**Kill criterion:** if even the optimistic number is below 0.1 tok/s, the project is a research curiosity, not a usable product. Stop or redefine the goal to batch-only.

**What it cannot tell you:** anything about real-world behaviour. This is a lower bound on badness, not a prediction.

---

## PHASE 1 — Structure of MoE routing (6–8 weeks)

This phase decides whether the entire thesis is valid. Do it before anything else.

### Test 1 — Does expert routing have exploitable structure?
**Question:** Is expert selection near-random, or clustered?

**Method:** Run an accessible MoE (Qwen3-MoE class, or DeepSeek-V3-class if budget allows) over a diverse prompt corpus — code, Russian prose, English technical text, math, long-context documents. Log per-token, per-layer expert IDs. Then measure:
- expert usage frequency distribution (Gini / entropy)
- temporal autocorrelation: P(expert e at token t+1 | expert e at token t)
- cross-layer co-activation mutual information
- domain separation: do code prompts and prose prompts use distinguishable expert sets?

**Should show:** a heavy-tailed usage distribution (a minority of experts carrying most traffic), strong temporal locality within a sequence, and measurable domain clustering.

**Kill criterion:** if usage is near-uniform *and* temporal autocorrelation is near zero, every token touches a random peer set. Placement optimisation becomes worthless and your only remaining lever is brute redundancy. The paper becomes much weaker. Consider pivoting to the speculative-decoding angle alone.

**What it cannot tell you:** whether K3 behaves like the proxy model. K3 uses a different routing design (LatentMoE), and its behaviour may differ substantially. **Everything in Phase 1 must be re-validated on K3 in Phase 4.** Treat Phase 1 as hypothesis generation, not proof.

### Test 2 — Is next-layer routing predictable?
**Question:** Can you know which peers to warm up before you need them?

**Method:** Train a small probe (linear, then 2-layer MLP) mapping layer-L hidden state → layer L+1 top-k expert set. Evaluate top-k recall on held-out data.

**Should show:** recall meaningfully above the frequency prior. Even 50–70% recall is enough to prefetch and hide most of one RTT.

**Kill criterion:** recall at chance level. Then every layer costs a full round trip with no overlap, and your latency floor is hop_count × RTT with no mitigation.

**What it cannot tell you:** whether prefetch is worth it economically — warming a peer you don't use wastes their bandwidth. That's a Phase 2 simulation question.

### Test 3 — Boundary precision tolerance
**Question:** How few bits can cross the wire?

**Method:** Insert fake-quantization at every layer boundary (fp8, int8 per-channel, int4 per-group). Measure perplexity delta and a small benchmark suite.

**Should show:** fp8 essentially free; int8 with per-channel scaling near-free; int4 probably degrades.

**Why it matters:** prefill of a long context is bandwidth-bound, not latency-bound. A 100k-token prefill at hidden dim 8192 in bf16 is ~1.6 GB per hop. At int8 that's 800 MB. This is the difference between "works on a home connection" and "doesn't."

**What it cannot tell you:** whether errors compound differently over the much greater depth of K3.

---

## PHASE 2 — Simulator (6–8 weeks)

Build a discrete-event simulator that wraps a real model but fakes the network. This is the highest-leverage artifact in the project — it lets you test 5,000-peer topologies without owning five machines.

### Test 4 — Baseline latency curve
**Question:** Where exactly is the cliff?

**Method:** Simulate hop count from 2 to 200, RTT from 20ms to 500ms, with realistic jitter. Plot tok/s surface.

**Should show:** a clear usable region and a clear dead region, and the maximum hop count you can tolerate at each RTT.

**Input you need first:** real peer-to-peer RTT distributions for consumer connections in your target geography. Do not use datacenter numbers. Measure them.

### Test 5 — Placement algorithm vs random
**Question:** Does co-activation-aware expert placement beat random assignment?

**Method:** Build the expert co-activation graph from Test 1 traces. Partition it (METIS or spectral) under per-peer capacity constraints. Compare cross-peer hops per token against random placement and against a frequency-only heuristic.

**Should show:** a meaningful reduction in hops per token — I'd hope for 30–50%, but I genuinely don't know.

**This test also answers the slice-size question from Section 0.** Sweep slice size as a parameter and find where total latency is minimised.

**Kill criterion:** if optimised placement beats random by under 10%, the scheduling paper isn't worth writing.

**What it cannot tell you:** how placement degrades when peers join and leave constantly. Static partitioning of a moving graph is a different and harder problem.

### Test 6 — Churn resilience
**Question:** What replication factor do you actually need?

**Method:** Model peer lifetime as a heavy-tailed distribution (most peers leave within minutes, a few stay for weeks). Sweep replication factor R. Measure request completion rate and mid-generation failure rate.

**Should show:** the R needed for ~99% completion. My guess is R between 3 and 5, but this is a guess.

**This output directly sets your launch target:** peers needed = (replica size ÷ slice size) × R.

### Test 7 — Speculative decoding gain
**Question:** Does drafting locally rescue the latency problem?

**Method:** Local small dense model drafts k tokens; swarm verifies in one batched pass. Sweep k and simulated acceptance rate.

**Should show:** 3–6x effective tok/s at 150–300ms RTT.

**Major unknown:** the acceptance rate of a ~1–3B draft model against a frontier 2.8T verifier. The capability gap is enormous and acceptance may be much worse than the published numbers for same-family draft/verify pairs. If acceptance is under 40%, the gain mostly evaporates. **Measure this on real models, do not assume it.**

---

## PHASE 3 — Real distributed prototype, small model (8–10 weeks)

### Test 8 — Two-process correctness
**Should show:** bit-identical (or within float tolerance) outputs vs single-machine inference. Unglamorous and non-negotiable.

### Test 9 — 10-node LAN
**Should show:** real throughput, and — critically — **simulator prediction error under 30%**.

**If the simulator is badly wrong here, every conclusion in Phase 2 is suspect and you go back and fix it.** This is the most important validation gate in the whole plan.

### Test 10 — 10-node WAN
**Method:** rent cheap VPS instances across three continents. Real internet, real jitter, real packet loss.

**Should show:** degradation vs LAN that matches the simulator's RTT model.

**What it cannot tell you:** anything about consumer NAT, residential ISP behaviour, or asymmetric uplinks. VPS networking is far better than home networking. That gap is tested in Phase 6.

---

## PHASE 4 — Scale to K3 (10–14 weeks)

### Test 11 — First correct token
**Question:** Can you shard 1.4TB across a rented cluster and produce one correct token?

**Should show:** output matching a reference implementation. Purely a milestone — no performance claim.

**Expect this to take much longer than estimated.** Weight loading, sharding format, MXFP4 handling, and the attention/KDA kernels are all places where undocumented details will cost weeks.

### Test 12 — Re-run Tests 1, 2, 5 on K3
**This is the real validation.** Everything from Phase 1 was on a proxy model. Now check whether K3's routing has the same structure.

**Should show:** qualitatively similar locality and predictability.

**If it doesn't:** the scheduler is retuned for K3's actual behaviour. Budget time for this; do not assume transfer.

### Test 13 — Heterogeneity and stragglers
**Method:** mix M-series (MLX), CUDA, and CPU-only peers with deliberately mismatched speeds.

**Should show:** the straggler penalty, and whether capacity-aware assignment recovers it.

**Likely finding:** the slowest peer in the active path dominates. You will need speculative re-dispatch — send the same work to two peers and take whoever answers first. That costs redundant compute; quantify the tradeoff.

---

## PHASE 5 — Trust, privacy, abuse (8 weeks, overlaps Phase 4)

### Test 14 — Malicious peer detection
**Method:** inject peers returning subtly corrupted activations. Detect via redundant spot-checking at sampling rate p.

**Should show:** detection rate vs overhead curve. Find the p that catches a cheater within N requests at acceptable cost.

**Unsolved in general:** you cannot cryptographically verify a matmul cheaply. ZK proofs are orders of magnitude too expensive. Accept statistical detection plus reputation, and be honest about that limitation in the paper.

### Test 15 — Activation inversion
**Question:** How much of a user's prompt can a hostile peer reconstruct from the activations it sees?

**Method:** implement a published inversion attack against your own boundaries.

**Should show:** the recovery rate. Expect it to be uncomfortably high at early layers.

**This is an ethical gate, not an optional extra.** If a peer can read user prompts, you must either disclose it prominently, keep early layers client-side, or add noise. Do not ship without answering this.

### Test 16 — Sybil resistance of the contribution ledger
**Question:** Can someone fake contribution to get free usage?

**Should show:** the cost of faking credit vs the cost of earning it honestly.

**Design note:** resist the urge to solve this with a token or blockchain. It attracts speculators rather than contributors and will distort the network's incentives badly. Proof-of-useful-work via redundant verification plus simple reputation is less elegant and more likely to work.

---

## PHASE 6 — Client and launch (6+ months)

### Test 17 — NAT traversal in the wild
**Should show:** direct-connection success rate across real residential networks. Under CGNAT, which is common in many markets, this may be poor.

**Contingency:** relay infrastructure, which you pay for, which changes the economics.

### Test 18 — Closed alpha, 50 real volunteers
**Should show:** actual peer capacity distribution, actual session lengths, actual churn. **Every number in Phase 2 was a guess until this test.**

**Expect the biggest surprise here.** Specifically: I expect far fewer volunteers than you hope have 16GB+ of free accelerator memory and are willing to leave a machine running.

### Test 19 — Public launch with seed cluster
**Requirement:** you must run enough seed capacity that the network is useful at peer count zero. Budget for this from day one.

---

## What cannot be predicted

Listed honestly, because these are the things that will actually determine the outcome.

1. **Whether K3's routing has exploitable locality.** All of Phase 1 is on proxy models. This is the single biggest technical unknown.
2. **Real peer capacity distribution.** How many people have 16GB+ free and will leave a machine on? Unknowable until Test 18.
3. **Churn distribution.** Everything in Phase 2 depends on a guessed lifetime curve.
4. **Draft model acceptance rate** against a 2.8T verifier.
5. **NAT/CGNAT reality** in your target markets.
6. **Licensing.** K3's weights are open but under a custom license with revenue-sharing obligations above certain thresholds. A free non-commercial network is probably fine. Read the actual text before you write the README. And the next model's license may be worse.
7. **Field velocity.** Prime Intellect, Nous, Pluralis and others are well-funded and fast. You may be scooped on any individual result within months.
8. **Hardware trajectory.** If consumer devices ship with 256GB of cheap unified memory in three years, the problem partly dissolves.

## What can go wrong, ranked by how likely it is to kill the project

1. **Cold start.** A network with 30 peers is useless and nobody joins a useless network. This is the number one killer of P2P projects, above all technical risks. Mitigation is a self-funded seed cluster, which is a real recurring cost.
2. **Solo-maintainer burnout.** This plan is 18–24 months of sustained work. Open-source infrastructure projects die of this constantly.
3. **Routing has no locality.** Test 1 kills the thesis cheaply — which is exactly why it's Test 1.
4. **Latency floor too high even for batch use.** If you land at 30s/token, there is no user.
5. **Abuse and legal exposure.** You will be operating a network that runs an uncensored frontier model for anonymous users, and some of them will attempt things that are seriously illegal. Peers may be running that computation on their machines in their jurisdictions without knowing what it is. This is a genuine liability question and it deserves legal advice before launch, not after — not a paragraph in a terms-of-service file.
6. **Privacy turns out to be unfixable** and Test 15 shows peers can read prompts. This constrains the product to non-sensitive use.
7. **Scooped.** Mitigated by publishing early and often rather than building in silence.

## Kill gates — cheapest places to stop

- After **Test 0**: arithmetic says impossible.
- After **Test 1**: no routing structure → thesis dead, ~2 months spent.
- After **Test 7**: no path below 5s/token in simulation → not a product.
- After **Test 9**: simulator doesn't match reality → rebuild before scaling.
- After **Test 18**: real volunteer capacity insufficient → the network cannot exist regardless of how good the software is.

## Rough timeline

| Phase | Duration | Cumulative |
|---|---|---|
| 0 — arithmetic | 1 week | 1 wk |
| 1 — routing structure | 6–8 weeks | ~2 mo |
| 2 — simulator | 6–8 weeks | ~4 mo |
| 3 — real prototype | 8–10 weeks | ~6.5 mo |
| 4 — K3 scale | 10–14 weeks | ~10 mo |
| 5 — trust/privacy | 8 weeks (overlaps) | ~10 mo |
| 6 — client and launch | 6+ months | 16–24 mo |

Publishable papers fall out at the end of Phase 2 (placement + simulator) and the end of Phase 4 (K3 results). The simulator and the routing-trace dataset are probably the most valuable things you will produce, and both come early.
