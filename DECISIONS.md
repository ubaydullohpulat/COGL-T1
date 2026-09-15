# Decision log

Plan amendments, scope changes, method choices, and gate verdicts. Newest first.
Format: **D-NNN · date · title** — Context / Decision / Alternatives considered / Consequences.

---

## D-008 · 2026-09-16 · Gate verdict T1-lite (proxy): PASS — routing has exploitable structure
- **Numbers vs threshold:** kill = near-uniform usage AND ~zero temporal autocorrelation. Measured on Ling-3.0-tiny (6,136 decode tokens):
  top-25% experts carry 56.9% of traffic (uniform 25%); adjacent-token overlap 43.8% (random 6.2%). Both conditions fail → PASS.
- **Caveat:** preliminary; one small proxy, 24 prompts. Full T1 on Ling-3.0-flash / ladder still required.
- **Paired risk surfaced (not a plan gate, but serious):** static hot-expert placement over home links loses 2.7× throughput even
  when keeping 90% of experts local (S02 §4). Expert-level sharding needs per-session adaptive placement or expert paging to be viable.

## D-007 · 2026-09-16 · Method: MLX environment and patched community weights
- **Decision:** project `.venv` (Python 3.11) with mlx 0.32.2 + mlx-lm pinned to git `872ae88d` (PyPI release lacks Ling-3.0 support); `requirements.txt` records it.
  Use `rapid-mlx/Ling-3.0-tiny-MLX-4bit@328a497` made loadable by `fix_kv_b_proj.py` (key renames + dequantized kv_b_proj), instead of a 16 GB download of the official bf16 weights.
- **Risk:** community quant ≠ official weights. If a result hinges on exact quality, re-run on a self-converted official checkpoint.

---

## D-006 · 2026-09-16 · Pivot: pause K3 work, focus on a few-peer swarm emulator with the Ling-3.0 family
- **Context:** Researcher: stop focusing on K3 for now; simulate a few instances with home-internet latency for Ling-3.0-flash and see how it behaves.
- **Decision:** T0 (K3 feasibility) and K3S (layer streaming) paused, not cancelled. New active track S01 SwarmLab:
  v0 = numpy Monte Carlo emulator with Ling-3.0-flash architecture, assumed compute, synthetic routing (done);
  v1 = real compute + routing traces from Ling-3.0-tiny (MLX) replayed in the emulator; then hybrid topology, MTP speculative decoding, and a localhost multi-process validation.
- **Why emulate before running real processes:** several peer processes on one M4 Max would compete for the same GPU and distort timings.
  Measure compute once, then add network time virtually. Keep a real multi-process run only as a validation gate.
- **Consequences:** ROADMAP current step moves to S01 v1. Needs approval for installing mlx / mlx-lm and downloading Ling-3.0-tiny (P-10).

---

## D-005 · 2026-09-15 · Testing strategy: proxy portfolio + layer-streamed real K3 (supersedes D-004's single-proxy choice)
- **Context:** Researcher: K3 can't be run here; find the best-suited stand-in, prove the concept, then validate on K3. Disk ~300 GB.
- **Decision (recommended, awaiting confirmation):** no single proxy matches K3, so each test uses the model that matches the property it depends on:
  - Primary routing proxy **Ling-3.0-flash** (sigmoid+noaux_tc router, 512/top-8, KDA + gated MLA, MIT)
  - LatentMoE proxy **Nemotron-3-Super-120B-A12B** (MLX 4-bit)
  - Family/draft proxy **Kimi Linear 48B-A3B** (kept from D-004)
  - Dev proxy **Ling-3.0-tiny**
  - Code-path correctness **mini-K3** (random-init shrunk K3 config)
  - Partial ground truth: **K3 layer streaming** (one ~17 GB shard at a time)
  - A granularity ladder (128 → 256 → 512 → 896 experts) plus a transfer rule decide whether a proxy result counts as evidence about K3.
- **Alternatives considered:** Kimi Linear alone (D-004; too coarse at 256/8, no LatentMoE); Qwen3.5/Qwen3-Next (softmax router, kept as optional control);
  gpt-oss (softmax, 128/4, no shared experts); REAP-pruned 2-bit K3 (pruning changes routing, 181 GB).
- **Consequences:** Phase 1 starts with a dev pipeline on Ling-3.0-tiny and a K3 streaming spike; Ling-3.0-flash needs a shard-by-shard MLX conversion tool.
  Full details: `experiments/P00-proxy-selection/README.md`.

---

## D-004 · 2026-09-15 · Target = Kimi K3; proxy for on-Mac work = Kimi Linear 48B-A3B
- **Context:** All near-term work runs on one Apple M4 Max (128 GB). K3 (1.56 TB) can't run locally.
- **Decision:** Final target `moonshotai/Kimi-K3` @ `f831ab66…`. Proxy for Phase 1–3 experiments:
  `moonshotai/Kimi-Linear-48B-A3B-Instruct` @ `e1df551a…` (MIT). Multiple "peers" are simulated on the one machine
  as logical partitions of a single loaded model, with a fake network in between (weights are not duplicated per peer).
- **Why this proxy:** same code family as K3 (K3 ships `modeling_kimi_linear.py` with LatentMoE/AttnRes flags on),
  same KDA:MLA 3:1 attention pattern, same sigmoid router type, identical tokenizer; fits in memory with headroom; MLX support exists.
- **Known gaps (validity threats):** no LatentMoE, no AttnRes, 256/top-8 experts vs 896/top-16, 27 vs 93 layers,
  hidden 2304 vs 7168. See `experiments/T00-feasibility/sources.md`. Phase 4 re-validation on K3 (T12) stays mandatory.
- **Alternatives considered:** Moonlight-16B-A3B (smaller, same tokenizer, but DeepSeek-V3-style MLA-only, further from K3);
  Qwen3-MoE / DeepSeek-class (plan v1 suggestion; different family).

## D-003 · 2026-09-15 · Project under git
- **Decision:** `git init` on `main`. Weights, large index files and raw traces are git-ignored; provenance via pinned revisions + `SHA256SUMS`.

## D-002 · 2026-09-15 · Adopt plan-review amendments (resolves P-1)
- **Decision:** All amendments in `docs/plan-review.md` adopted. For T0 specifically: define hop/RTT precisely;
  evaluate layer-pipeline, expert-star, expert-chain, and each with speculative decoding; give a gate verdict per topology.
  Also: literature sweep before Phase 1 build; start RTT measurements (L2) early; check same-tokenizer draft availability (L4).
- **Consequences:** T0 brief updated; ROADMAP gained long-lead tracks L1–L4.

---

## D-001 · 2026-09-15 · Set up research notebook & project structure
- **Context:** Project start. Master plan received (`docs/plan/swarm-moe-research-plan.v1.md`).
- **Decision:** Freeze the plan as v1; track status in `ROADMAP.md`, results in `FINDINGS.md`,
  daily log in `notebook/`, amendments here. Notebook protocol defined in `CLAUDE.md`.
- **Consequences:** every session starts/ends with the notebook protocol.

---

## Pending decisions (backlog — need the researcher)
- ~~P-1~~ resolved by D-002 · ~~P-2~~ resolved by D-004 · ~~P-5~~ resolved by D-003
- **P-3** (partly resolved by D-004) Any budget for rented GPUs / VPS later (Phase 3 WAN, Phase 4)?
- **P-4** Target geography for RTT measurements (L2)
- ~~P-10~~ approved 2026-09-16 (D-007): create `.venv`, `pip install mlx mlx-lm` (Apple, PyPI), download `rapid-mlx/Ling-3.0-tiny-MLX-4bit` (4.5 GB) or convert `inclusionAI/Ling-3.0-tiny` (bf16 ~16 GB) ourselves?
- **P-7** (paused by D-006) Approve the K3 layer-streaming spike (first 4 layers ≈ 53 GB download ≈ 1.6 h, ~30 GB peak disk)?
- **P-8** Confirm D-005 proxy portfolio (esp. Ling-3.0-flash as primary despite needing our own MLX conversion)?
- **P-9** Budget for K3 hosted API (T7 draft acceptance) — and does the API expose logprobs?
- **P-6** Trace precision for Phase 1 on the Mac: MLX 8-bit (52 GB, lots of headroom) for bulk traces + bf16 (98 GB, tight) spot-check that quantization doesn't flip top-k routing? Or bf16 only?
