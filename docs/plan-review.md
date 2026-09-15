# Plan review — 2026-09-15

Review of `docs/plan/swarm-moe-research-plan.v1.md`. The plan is strong: kill gates are cheap
and early, unknowns are listed honestly, and the simulator-validation gate (Test 9) is the right
backbone. Issues below are proposed amendments; accepted ones get logged in `DECISIONS.md`.

## A. Issues that affect Step 1 (Test 0) directly

### A1. The Test 0 kill line is likely to land in the gray zone — define it precisely first
Frontier MoEs in this class (DeepSeek-V3, Kimi K2) have ~61 layers. Decode must visit every layer
sequentially. Rough numbers for ~60 remote MoE layers, one network round trip each, zero compute:

| Per-layer latency | s/token | tok/s |
|---|---|---|
| 50 ms  | 3.0 | 0.33 |
| 150 ms | 9.0 | 0.11 |
| 300 ms | 18  | 0.06 |

At 150 ms that is ~0.11 tok/s — below the 0.3 "continue" line and just above the 0.1 "kill" line.
(These are `COMPUTED` from an `ASSUMED` layer count; the real number comes from the K3 config.)

So Test 0 must, before computing anything:
- **Define "hop" and "RTT".** Star topology (client → k expert peers in parallel → back) costs one
  RTT per MoE layer. Chain topology (peer → peer) costs one *one-way* latency (≈ RTT/2) per hop.
  These differ by 2×, which is the whole distance between PASS and GRAY.
- **Compute several topologies, not one number:** (a) Petals-style layer pipeline,
  (b) expert-sharded star with attention on the client, (c) expert-sharded chain,
  (d) (b)/(c) + speculative decoding at a range of acceptance rates.
- **State the gate for each**, since the plan's kill criterion implicitly assumes no speculative
  decoding while Test 7 assumes it rescues latency.

### A2. Which model is "K3", exactly?
Pin the exact model, revision, tech report and `config.json` before extracting numbers. Nothing in
Test 0 should be taken from memory; every architecture number is `CITED` with file@revision.
Sanity check: recomputing total and active parameters from the extracted dims must reproduce the
reported totals (within ~2%), or the extraction is wrong.

### A3. Read the license during Step 1, not at launch
It costs an hour and could change the target model. Record the relevant clauses in the notebook.

## B. Issues for later phases (worth deciding early)

### B1. Speculative decoding requires a same-tokenizer draft model
Standard speculative decoding needs the draft and verifier to share a tokenizer/vocabulary.
If no small model in the target's family exists, a draft must be distilled/trained — a real
work item not in the plan. Check availability in Step 1 (5 minutes) and log it.

### B2. Tests 1–2 have prior art — do a literature sweep before building
Expert temporal locality, inter-layer expert affinity and next-layer expert prediction have been
studied in the MoE offloading literature (see `references/reading-list.md`). Phase 1 can likely be
reframed as "reproduce known findings on our proxy + measure the metrics that matter for
*network* placement", which is shorter and makes the paper's novelty claim safer.
Decentralized MoE over volunteers also has direct precedent (Learning@home / Hivemind).

### B3. Start long-lead measurements now, in parallel
- **Consumer RTT distribution in the target geography** (Test 4 input). Needs real data and
  calendar time; cheap to start (RIPE Atlas-style public data, or friends' home connections).
- **Draft-model acceptance proxy** (Unknown #4): can be partially estimated early with the largest
  model you can access + a same-family small model.

### B4. Test 1 needs a hardware/budget decision
Which proxy MoE is feasible depends on available hardware (local Mac memory, rented GPUs).
This is an open question for the researcher and gates Phase 1 setup.

### B5. Minor
- Section 0 table: MXFP4 is ~4.25 bits/param incl. scales, so 1B ≈ 0.53 GB; fine as rounded.
- Test 0 also needs compute-time-per-peer on consumer hardware as an input (even a rough
  `ASSUMED` value), otherwise the tok/s figure is network-only.
- The project is not yet under version control; a git repo makes the notebook auditable.
