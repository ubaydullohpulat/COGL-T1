# Findings ledger

Durable results only — numbers or conclusions that later work builds on.
Every row links to the notebook entry and the evidence file. Superseded rows are struck through, not deleted.

| ID | Date | Test | Finding | Value | Evidence type | Evidence | Notebook |
|---|---|---|---|---|---|---|---|
| F-001 | 2026-09-15 | T0 | K3 depth: 93 layers (1 dense + 92 MoE), 69 KDA + 24 gated MLA, hidden 7168 | 93 | CITED | `experiments/T00-feasibility/sources/kimi-k3/config.json` @ f831ab6 | [2026-09-15](notebook/2026/2026-09-15.md) |
| F-002 | 2026-09-15 | T0 | K3 MoE: 896 routed experts, top-16, 2 shared, LatentMoE dim 3584, expert FFN 3072 | 896/16 | CITED | same | same |
| F-003 | 2026-09-15 | T0 | K3 checkpoint size (index total_size, incl. vision) | 1.56 TB | CITED | `kimi-k3/model.safetensors.index.json` @ f831ab6 | same |
| F-004 | 2026-09-15 | T0 | AttnRes: residual snapshot every 12 layers → chain-topology boundary state up to 9 × 7168 values/token; expert calls need only 3584-d latent | 9× vs 0.5× hidden | CITED (code reading) | `kimi-k3/modeling_kimi_linear.py` L877–1048 | same |
| F-005 | 2026-09-15 | L4 | K3, Kimi Linear 48B-A3B, Moonlight 16B-A3B, Kimi-VL-A3B share an identical `tiktoken.model` | sha256 b6c497a7… | MEASURED | `sources/SHA256SUMS` | same |
| F-006 | 2026-09-15 | P0 | K3 checkpoint is laid out one decoder layer per shard (~17.0 GB; layer 0 2.3 GB; non-layer 5.6 GB) → layer streaming needs ~30 GB peak disk | 1 shard/layer | CITED (HF tree + index) | `experiments/T00-feasibility/sources` + P0 README §4 | [2026-09-15](notebook/2026/2026-09-15.md) |
| F-007 | 2026-09-15 | P0 | `mlx-lm` `kimi_k3.py` loads native MXFP4 K3 weights and implements KDA, MLA, LatentMoE, AttnRes; its pipeline mode ships AttnRes blocks + hidden across ranks | — | CITED (code) | mlx-lm main, `mlx_lm/models/kimi_k3.py` L821–843, L903–961 | same |
| F-008 | 2026-09-15 | P0 | No model that fits the Mac has 896 experts or AttnRes; only Nemotron-3-Super has LatentMoE; closest router+attention+sparsity match is Ling-3.0-flash | — | CITED (19 configs) | `experiments/P00-proxy-selection/configs/`, README §3 | same |
| F-009 | 2026-09-15 | L2 | Researcher's connection: 72 Mbps down / 30 Mbps up; base RTT 137 ms to Apple's measurement server | 72/30 Mbps | MEASURED | `networkQuality -s -c` | same |

