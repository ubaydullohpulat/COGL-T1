# T0 — Napkin feasibility

**Phase:** 0 · **Status:** IN PROGRESS (defined 2026-09-15) · **Budget:** ~1 week, arithmetic only

## Question
Is the target physically possible? What is the *best-case* decode tok/s for a ~2.8T MoE served
by expert-sharded peers, at 50 / 150 / 300 ms network latency?

## Gate (from plan v1)
- Best case at 150 ms > **0.3 tok/s** → continue
- Optimistic case < **0.1 tok/s** → stop, or redefine as batch-only
- In between → **GRAY**: record it, and decide whether spec decoding (T7) is a hard requirement from day one

Proposed amendment (pending P-1): evaluate the gate per topology, with and without speculative decoding.

## What this cannot tell us
Anything about real-world behaviour. This is a lower bound on badness, not a prediction.

---

## Sub-steps

### 1.1 Pin the target  *(~half day)*
- [ ] Exact model name, org, release, weights revision/hash → `sources.md`
- [ ] Tech report (URL, version) and `config.json` @ revision saved locally
- [ ] License text: read, quote the clauses on non-commercial use / revenue thresholds / redistribution (L3)
- [ ] Is there a small same-tokenizer model in the family? (L4 — needed for spec decoding)

### 1.2 Extract architecture  *(~1 day)* → `arch.md`
Every value `CITED` with file + field / report section:
- [ ] total layers; dense vs MoE layers
- [ ] hidden dim; vocab size; embedding tying
- [ ] routed experts per layer; top-k; shared experts
- [ ] expert FFN intermediate dim; activation type (gated → 3 matrices)
- [ ] attention type(s) and layer mix (e.g. linear/KDA vs full attention ratio), attention params per layer, KV/state size
- [ ] native precision / quantization format (MXFP4 details: block size, scale format)
- [ ] **Sanity check:** recompute total and active params from dims → must match reported within ~2%

### 1.3 Define the latency model  *(~half day)* → `model.md`
- [ ] Define terms: RTT, one-way latency, "hop", compute time per stage
- [ ] Topologies to evaluate:
  - **A** Layer pipeline (Petals-style): whole layers per peer, chain
  - **B** Expert-sharded **star**: attention + router on client, top-k experts fetched in parallel, 1 RTT per MoE layer
  - **C** Expert-sharded **chain**: peers forward to next layer's peers, ≈ one-way latency per hop
  - **D** B/C + speculative decoding: draft length k ∈ {2,4,8}, acceptance α ∈ {0.3,0.5,0.7}
- [ ] Assumed per-peer compute time for one expert on consumer hardware (`ASSUMED`, stated range)
- [ ] Slice sizes from plan §0: 1B / 4B / 16B / 32B → how many distinct peers per layer, per token

### 1.4 Compute  *(~1 day)* → `calc.py` + `results.md`
A single small arithmetic script (not a system) so sweeps are reproducible:
- [ ] active params / token
- [ ] bytes per layer boundary: hidden_dim × {bf16, fp8, int8, int4}; decode (1 token) and prefill (4k, 32k, 100k)
- [ ] minimum sequential network waits per token, per topology
- [ ] tok/s table: topology × latency {50, 150, 300 ms} × spec-decoding setting
- [ ] Calibration: run the same model on a published system's setup (e.g. Petals reported throughput) and compare — if our calc is wildly optimistic vs their measured numbers, find out why

### 1.5 Verdict  *(~half day)*
- [ ] Gate verdict per topology (PASS / GRAY / FAIL) with numbers → notebook + `DECISIONS.md`
- [ ] Row(s) in `FINDINGS.md`
- [ ] What T0 implies for later tests (e.g. "spec decoding is mandatory", "star beats chain by X")

## Files (to be created)
- `sources.md` — exact provenance of every input
- `arch.md` — sourced architecture table
- `model.md` — latency model definitions and assumptions
- `calc.py` — arithmetic
- `results.md` — tables + gate verdict
