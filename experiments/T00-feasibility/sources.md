# T0 sources — pinned provenance

Retrieved 2026-09-15 from Hugging Face, at pinned revisions. Local copies in `sources/`, hashes in `sources/SHA256SUMS`.
`model.safetensors.index.json` files are git-ignored (60 MB for K3); re-download them from the pinned revision.

## Target — Kimi K3 (final product)
| Item | Value |
|---|---|
| Repo | `moonshotai/Kimi-K3` |
| Revision | `f831ab66814297da540d832a5235f8e904f29d06` (lastModified 2026-09-02) |
| Files kept | config.json, LICENSE, README.md, configuration_kimi_k3.py, modeling_kimi_k3.py, modeling_kimi_linear.py, generation_config.json, tokenizer_config.json, tiktoken.model, (index) |
| Weights | 96 safetensors shards; index `total_size` = 1,560,860,324,864 bytes (1.56 TB / 1.42 TiB), 497,220 tensors, incl. vision tower |
| License | Custom "Kimi K3 License" (`sources/kimi-k3/LICENSE`) |
| Tech report | Not yet located as a primary document. README architecture table used; secondary blog posts are NOT used as sources |

## Proxy — Kimi Linear 48B-A3B (all experiments on this Mac)
| Item | Value |
|---|---|
| Repo | `moonshotai/Kimi-Linear-48B-A3B-Instruct` |
| Revision | `e1df551a447157d4658b573f9a695d57658590e9` (lastModified 2025-12-16) |
| Base model | `moonshotai/Kimi-Linear-48B-A3B-Base` @ `3b171c17bfc4ee348599b6781a2ca8715c21c8dc` |
| Weights | 20 shards, bf16, `total_size` = 98,245,528,576 bytes (98.2 GB) |
| License | MIT |
| MLX builds (mlx-lm has `kimi_linear.py` and `kimi_k3.py`) | `mlx-community/…-Instruct-mlx-bf16` @ `c318f02…` 98.2 GB · `…-8bit` @ `553a7a5…` 52.2 GB · `…-4bit` @ `919175d…` 27.6 GB |

## Hardware (experiment host)
Apple M4 Max, 16 cores, 128 GB unified memory, macOS 26.5.1, 281 GB free disk (2026-09-15).

---

## Proxy fidelity — what Kimi Linear does and does NOT share with K3
All values `CITED` from the two config.json files above (and code in K3's `modeling_kimi_linear.py`).

| Property | Kimi K3 | Kimi Linear 48B | Same? |
|---|---|---|---|
| Model code | `KimiLinearForCausalLM` (text part), flags on | `KimiLinearForCausalLM`, flags off | same code family |
| Layers (dense first) | 93 (1 dense) | 27 (1 dense) | ✗ depth 3.4× |
| Attention mix | 69 KDA + 24 gated MLA (every 4th + last) | 20 KDA + 7 MLA (every 4th + last) | ✓ same 3:1 pattern |
| MLA output gate / full-rank KDA gate | yes / yes | no / no | ✗ |
| Hidden size | 7168 | 2304 | ✗ |
| Routed experts / top-k | 896 / 16 (1.8% active) | 256 / 8 (3.1% active) | ✗ finer-grained in K3 |
| Shared experts | 2 | 1 | ✗ |
| Router | sigmoid, `noaux_tc` correction bias, renormalize | sigmoid, grouped top-k (1 group), renormalize | ~ similar |
| **LatentMoE** | routed experts run in a 3584-d latent: shared down-proj 7168→3584 (+norm), experts 3584→3072→3584, shared up-proj | none: experts run at hidden size | ✗ **key difference** |
| **AttnRes** | block size 12: residual snapshot stored at layer_idx 0,12,…,84; each layer mixes attention and MLP inputs over all snapshots + current stream | none | ✗ **key difference** |
| Activation | SiTU-GLU | SiLU-GLU | ✗ |
| Weights | MXFP4 (group 32) on routed experts; attention, shared experts, dense MLP, lm_head, vision kept unquantized (per `ignore` list) | bf16 | ✗ |
| Tokenizer | `tiktoken.model` sha256 `b6c497a7…` | identical hash | ✓ (chat-format special tokens renamed at a few IDs) |

**Implication for Phase 1 (validity threat, record in paper):** Kimi Linear shares K3's attention design,
router type and tokenizer, but not its expert granularity (256/8 vs 896/16), LatentMoE, or AttnRes.
Routing-structure results on Kimi Linear are hypothesis generation for K3, exactly as plan v1 already warns.

---

## Structural observations relevant to T0 / networking (from reading K3 code)
1. **Wire payload at a layer boundary depends on topology.** With AttnRes, the full per-token state passed
   between layers is the current stream plus up to 8 snapshots, i.e. up to 9 × 7168 values at deep layers
   (chain topology). A remote routed-expert call only needs the 3584-d latent vector each way
   (gate + down/up-proj stay with the layer host), i.e. half of one hidden vector.
2. **Routing is decided centrally per layer.** The gate runs on the 7168-d stream before down-projection,
   so a layer host naturally fans out to its top-16 expert peers (star within the layer).
3. **Open for 1.2:** how large is the non-routed part (attention + shared experts + dense + embeddings, unquantized)?
   That decides whether any consumer device can host "attention + routing" for even one layer range.

## License notes (L3) — paraphrased, not legal advice
- MIT-style grant: use, modify, distribute, sublicense, sell; keep the copyright + permission notice.
- Separate agreement needed if a "Model as a Service" operator (third parties control inputs via API-like access)
  has affiliate-wide revenue above USD 20M over 12 months and uses it commercially.
- Products with >100M MAU or >USD 20M monthly revenue must display "Kimi K3" prominently.
- These conditions don't apply to internal use, or to access via Moonshot's official products/partners.
- Reading for this project: a free, non-commercial volunteer network is well below the thresholds; the notice
  requirement applies. A P2P network plausibly *is* "Model as a Service" by definition, so revisit if any
  revenue or sponsorship appears. Real legal review still required before launch (plan v1, risk #5).

## Draft-model availability (L4)
Same `tiktoken.model` hash (`b6c497a7…`) confirmed for:
- `moonshotai/Kimi-Linear-48B-A3B-Instruct` (48B total / 3B active)
- `moonshotai/Moonlight-16B-A3B-Instruct` @ `4e735b07…` (16B total / 64 experts, top-6, 27 layers)
- `moonshotai/Kimi-VL-A3B-Instruct` @ `398eede0…`

So same-tokenizer drafts for K3 exist, but all are sparse 16–48B-total models (≈3B active), not the 1–3B
dense drafts the plan assumed. Memory footprint on a consumer device is ~9 GB (Moonlight, 4-bit, est.) or
~28 GB (Kimi Linear, 4-bit MLX, measured file size). Remaining check: special-token ID alignment for chat format.
