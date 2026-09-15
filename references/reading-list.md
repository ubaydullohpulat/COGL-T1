# Reading list

Status: `TO READ` · `SKIMMED` · `READ` · `UNVERIFIED` (citation details not yet checked by us).
All entries below were suggested from background knowledge on 2026-09-15 and are **UNVERIFIED** —
confirm title/authors/venue/link before citing anything.

## Decentralized / volunteer inference & training (closest prior art)
| Work | Why it matters | Status |
|---|---|---|
| Learning@home / Hivemind (Ryabinin & Gusev, 2020) | Decentralized mixture-of-experts over volunteer hardware — direct precedent | TO READ, UNVERIFIED |
| Petals (Borzunov et al., 2022–23) | Layer-pipeline inference of 100B+ models over the internet; latency baselines for calibration | TO READ, UNVERIFIED |
| Distributed Inference and Fine-tuning of LLMs Over the Internet (Borzunov et al., 2023) | Fault tolerance + load balancing in Petals | TO READ, UNVERIFIED |
| SWARM Parallelism (Ryabinin et al., 2023) | Randomized pipelines over unreliable heterogeneous devices | TO READ, UNVERIFIED |
| exo (open-source project) | Heterogeneous home-cluster inference incl. Apple silicon | TO READ, UNVERIFIED |
| Prime Intellect, Nous (Psyche/DisTrO), Pluralis, Gensyn | Competitive landscape / scoop risk | TO READ, UNVERIFIED |

## Expert routing structure & prediction (Tests 1–2, 5)
| Work | Why it matters | Status |
|---|---|---|
| Towards MoE Deployment: Mitigating Inefficiencies in MoE Inference (Huang et al., 2023) | Expert activation patterns, temporal locality, caching | TO READ, UNVERIFIED |
| Fast Inference of MoE Language Models with Offloading (Eliseev & Mazur, 2023) | LRU expert caching + speculative expert loading on Mixtral | TO READ, UNVERIFIED |
| Pre-gated MoE (Hwang et al., 2024) | Predicting next-layer experts from current layer | TO READ, UNVERIFIED |
| MoE-Infinity (Xue et al., 2024) | Activation tracing for offloading decisions | TO READ, UNVERIFIED |
| ExFlow — inter-layer expert affinity (Yao et al., 2024) | Cross-layer co-activation used for placement — closest to Test 5 | TO READ, UNVERIFIED |
| EdgeMoE (Yi et al., 2023) | MoE on edge devices, expert preloading | TO READ, UNVERIFIED |

## Speculative decoding (Test 7)
| Work | Why it matters | Status |
|---|---|---|
| Leviathan et al., 2023; Chen et al., 2023 | Core speculative sampling, acceptance-rate math | TO READ, UNVERIFIED |
| Medusa; EAGLE | Draft-head approaches that avoid a separate tokenizer-matched model | TO READ, UNVERIFIED |

## Target model
| Item | Why it matters | Status |
|---|---|---|
| K3 technical report | Test 0 architecture numbers | TO FIND — exact model/revision not yet pinned |
| K3 `config.json` @ revision | Ground truth for dims | TO FIND |
| K3 license text | Unknown #6 | TO FIND |
