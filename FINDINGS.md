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
