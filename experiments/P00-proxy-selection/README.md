# P0 — Proxy selection: how to test a model we can't run

**Date:** 2026-09-15 · **Status:** DONE (recommendation) — pending researcher confirmation of D-005 and the K3 streaming spike (P-7)
**Constraints:** Apple M4 Max, 128 GB unified memory, **~300 GB free disk** (researcher, 2026-09-15), 72 Mbps down / 30 Mbps up (`networkQuality`, MEASURED).
**Raw data:** `configs/*.json` (pinned revisions in `candidates_meta.json`), produced by `fetch_configs.py`.

---

## 1. Key reframing: which tests need what

"We can't run K3" is true for **serving** it. But each test needs something different:

| Test needs… | Tests | How we get it without serving K3 |
|---|---|---|
| **Trained routing behaviour** (which experts fire) | T1, T2, T5, T12 | Proxy ladder (§3) **+ real K3 traces via layer streaming (§4)** |
| **Trained activations + depth** (quant error compounding) | T3 | Proxy for full perplexity; K3 streaming for per-layer error on real K3 |
| **Exact architecture only** (shapes, code paths, compute cost) | T0 inputs, T4, T8, T9, T10, T13 | **mini-K3**: `mlx-lm` `kimi_k3.py` with a shrunk config and random weights (exercises LatentMoE, AttnRes, KDA, MLA with no download); real per-layer compute timing from one streamed K3 layer |
| **K3's output distribution** | T7 (draft acceptance) | K3 via hosted API + a same-tokenizer local draft (Kimi Linear / Moonlight) |
| **Real early-layer activations** | T15 (inversion) | K3 streaming of the first few layers only (~2.3 + 17 GB per layer) |

So the plan is not a single proxy but **a proxy portfolio plus partial real-K3 measurement**, with a transfer check between them.

---

## 2. What makes a model "K3-like" for this research

All K3 values `CITED` from `moonshotai/Kimi-K3` config @ f831ab6.

| # | K3 property | Why it matters for us | Weight |
|---|---|---|---|
| a | Router: sigmoid scores + aux-loss-free correction bias (`noaux_tc`), renormalized top-k | Shapes usage skew and locality (T1, T5) | high |
| b | Fine-grained experts: 896, top-16 (1.8% active) + 2 shared | Granularity drives placement/partitioning (T5) and hops | high |
| c | Attention: KDA (linear, recurrent state) + gated MLA, ~3:1 | Where per-sequence state lives; long-context behaviour | high |
| d | LatentMoE: experts in a 3584-d latent (7168 → 3584 → 7168) | Wire payload per expert call (T3, T4); may change expert specialization | medium |
| e | AttnRes: block residual snapshots (block 12) | Boundary state size in chains; error propagation (T3) | medium |
| f | Depth: 93 layers (92 MoE) | Hop count, error compounding | medium |
| g | Fits on this Mac (memory + ~300 GB disk) and runs in MLX | Practicality | gate |

---

## 3. Candidates surveyed (19 models; configs pinned in `candidates_meta.json`)

✓ = matches K3 · ~ = partial · ✗ = differs. All values `CITED` from each model's config.json; router type confirmed in `mlx-lm` model code.

| Model | Params | Experts / top-k (active %) | Shared | Router (a) | Attention (c) | LatentMoE (d) | AttnRes (e) | MoE layers | Fits Mac? (g) | License |
|---|---|---|---|---|---|---|---|---|---|---|
| **Kimi K3 (target)** | 2.8T | 896 / 16 (1.8%) | 2 | sigmoid + bias | 69 KDA + 24 gated MLA | ✓ 2× | ✓ | 92 | ✗ (1.56 TB) | Kimi K3 |
| **Ling-3.0-flash** (inclusionAI) | 127B | 512 / 8 (1.6%) ✓ | 1 | ✓ sigmoid + `noaux_tc` | ✓ 35 KDA + 7 gated MLA (5:1) | ✗ | ✗ | 40 | ~ 4-bit ≈ 70 GB; **no MLX build yet** | MIT |
| **Nemotron-3-Super-120B-A12B** (NVIDIA) | 124B | 512 / 22 (4.3%) | 1 | ✓ sigmoid + correction bias | ✗ 40 Mamba2 + 8 attn | ✓ **4×** (4096→1024) | ✗ | 40 | ✓ MLX 4-bit 68 GB | NVIDIA Open Model |
| **Kimi Linear 48B-A3B** (Moonshot) | 49B | 256 / 8 (3.1%) | 1 | ✓ sigmoid, grouped | ✓ 20 KDA + 7 MLA (3:1) | ✗ | ✗ (trained variant in AttnRes paper, **not released**) | 26 | ✓ MLX 8-bit 52 GB | MIT |
| **Ling-3.0-tiny** (inclusionAI) | 7.9B | 128 / 8 (6.3%) | 1 | ✓ sigmoid + `noaux_tc` | ✓ KDA + gated MLA | ✗ | ✗ | 23 | ✓ MLX 4-bit 4.5 GB | MIT |
| Moonlight-16B-A3B | 16B | 64 / 6 | 2 | ✓ sigmoid + `noaux_tc` | ✗ MLA only | ✗ | ✗ | 26 | ✓ | MIT |
| Ling-mini-2.0 | 16B | 256 / 8 | 1 | ✓ sigmoid | ✗ full attn | ✗ | ✗ | 19 | ✓ | MIT |
| Qwen3.5-35B-A3B / Qwen3.6-35B-A3B | 36B | 256 / 8 | 1 | ✗ softmax | ~ Gated DeltaNet + gated attn 3:1 | ✗ | ✗ | 40 | ✓ | Apache-2.0 |
| Qwen3.5-122B-A10B | 125B | 256 / 8 | 1 | ✗ softmax | ~ GDN + gated attn 3:1 | ✗ | ✗ | 48 | ✓ 4-bit | Apache-2.0 |
| Qwen3-Next-80B-A3B | 81B | 512 / 10 | 1 | ✗ softmax | ~ GDN + gated attn 3:1 | ✗ | ✗ | 48 | ✓ 4-bit | Apache-2.0 |
| Nemotron-3-Nano / 3.5-Lightning 30B-A3B | 32B | 128 / 6 | 1 | ✓ sigmoid + bias | ✗ Mamba2 hybrid | ✗ | ✗ | 23 | ✓ | NVIDIA Open |
| gpt-oss-120b / 20b | 117B / 21B | 128 / 4, 32 / 4 | 0 | ✗ softmax | ✗ sliding/full | ✗ | ✗ | 36 / 24 | ✓ | Apache-2.0 |
| GLM-4.5-Air / GLM-4.7-Flash | 110B / 31B | 128 / 8, 64 / 4 | 1 | ✓ sigmoid + bias | ✗ full attn | ✗ | ✗ | 45 / 46 | ✓ 4-bit | MIT |
| Qwen3.8-Flash-Next | 180B | 512 / 10 | 1 | ✗ softmax | ~ | ✗ | ✗ | 48 | ✗ disk/memory tight; experimental arch | — |
| open-attnres-0.6b-block | 0.5B dense | — | — | — | full attn | — | ✓ block AttnRes | — | ✓ | — |
| mlx-community/Kimi-K3-mlx-reap160-2bit | pruned K3 | 160 experts (REAP-pruned), 2-bit | | | | | | | ✗ 181 GB, and pruning changes routing, so unusable for routing research | |

**No available model has all of K3's properties.** Nothing that fits has 896 experts or AttnRes; only Nemotron-3-Super has LatentMoE.

---

## 4. Real K3 without holding K3: layer streaming (proposed spike, needs approval)

Facts (`MEASURED`/`CITED`, 2026-09-15):
- K3 checkpoint = 96 shards; **each decoder layer lives in exactly one shard** (~17.0 GB; layer 0 = 2.3 GB; embeddings/head/vision = 5.6 GB in 3 shards).
- `mlx-lm` ships `kimi_k3.py` that loads K3's native MXFP4 (`weight_packed` / `weight_scale`) and implements KDA, gated MLA, LatentMoE and AttnRes (with a Metal kernel). Its pipeline mode already sends AttnRes blocks + hidden state across ranks, consistent with F-004.

Method (prefill / teacher-forced traces over a fixed corpus):
1. Tokenize corpus → embeddings (non-layer shards, 5.6 GB, kept).
2. For L = 0…92: download shard L (17 GB), load layer L, push all corpus hidden states (+ AttnRes blocks, + per-sequence attention state) through it, **log router top-k IDs, scores, pre-/post-gate hidden states**, save the new hidden states, delete shard L.
3. Result: real K3 routing traces for every layer, without ever holding more than ~1 layer on disk.

Budget (mostly `COMPUTED`, compute time a `GUESS` until the spike measures it):
| Item | Value |
|---|---|
| Peak disk | ~17 GB layer + 5.6 GB non-layer + hidden-state checkpoint (≈ tokens × 7168 × up to 9 × 2 B ≈ 129 KB/token → 50k tokens ≈ 6.5 GB) ≈ **30 GB**. Fits the 300 GB budget. |
| Download, full pass | 1.56 TB at 72 Mbps ≈ **48 h** (one layer ≈ 31 min) |
| Download, first 4 layers (spike) | ~53 GB ≈ 1.6 h |
| Compute per layer | GUESS: minutes for ~50k tokens. Measure in spike. |
| Re-runs | Each full re-run re-downloads 1.56 TB unless shards are cached on an external drive |

Limits: traces are prefill on real text, not decode on the model's own samples (fine for T1/T2; note it). Vision tower unused.
Spike question: does one real K3 layer load and run correctly in MLX on this Mac, and how long does it take per 1k tokens?

---

## 5. Recommendation — the proxy portfolio

| Role | Model | Used for | Why this one |
|---|---|---|---|
| **Ground truth (partial)** | **Kimi K3, layer-streamed** | T1/T2 transfer check, per-layer T3 error, T0 compute timing, T15 early layers | The only real K3 data we can get locally |
| **Primary routing proxy** | **Ling-3.0-flash** (MIT) | T1, T2, T3, T5 full runs; simulator traces | Closest match on (a) router, (b) sparsity 1.6% vs 1.8%, (c) KDA + gated MLA. Cost: no MLX build, so we convert it ourselves shard by shard (bf16 is 255 GB; can't be fully stored) |
| **LatentMoE proxy** | **Nemotron-3-Super-120B-A12B** (MLX 4-bit, 68 GB) | Repeat T1/T2/T5 to test whether a latent expert space changes routing structure; LatentMoE payload for T3/T4 | Only fitting model with LatentMoE; same sigmoid + bias router; 512 experts. Ready to run today |
| **Family + draft proxy** | **Kimi Linear 48B-A3B** (MLX 8-bit) | Same code/tokenizer as K3; T7 draft candidate; T8 correctness on a real Kimi model | Same code path as K3 with flags off |
| **Dev / fast iteration** | **Ling-3.0-tiny** (MLX 4-bit, 4.5 GB) | Build and debug the trace logger, probes, simulator before spending hours on big models | Same architecture as Ling-3.0-flash at 1/16 size |
| **Code-path correctness** | **mini-K3** (random init, shrunk K3 config) | T8 two-process correctness incl. AttnRes + LatentMoE boundaries | Every K3 code path, zero download |

**Granularity ladder** (router type held constant, sigmoid + bias): Ling-3.0-tiny 128/8 → Kimi Linear 256/8 (or Ling-mini-2.0 256/8) → Ling-3.0-flash 512/8 → Nemotron-3-Super 512/22 → K3 streamed 896/16. Measure every T1 metric across this ladder, then check where real K3 layers fall. The trend matters more than any one proxy.

**Control (optional):** Qwen3.5-35B-A3B, which has a softmax router. If structure holds on both router families, the finding is more robust.

### Transfer rule (how proxies earn trust)
A proxy result counts as evidence about K3 only if the same metric, measured on streamed K3 layers,
falls within the proxy ladder's range or trend. Otherwise the proxy result is labelled "proxy-only".

### Disk plan (≤ 300 GB, staged, never all at once)
| Stage | Holds | Approx. GB |
|---|---|---|
| Dev | Ling-3.0-tiny 4-bit | 5 |
| K3 spike | 4 K3 layers streamed one at a time + non-layer shards + checkpoints | ~30 peak |
| Proxy runs | Ling-3.0-flash 4-bit (after streamed conversion) **or** Nemotron-3-Super 4-bit, one at a time | ~70 |
| Always | Kimi Linear 8-bit | 52 |
| Traces | routing traces + hidden-state samples | budget 50 |
