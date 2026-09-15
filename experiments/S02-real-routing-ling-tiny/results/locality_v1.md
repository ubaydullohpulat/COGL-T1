# Routing structure & locality — Ling-3.0-tiny (v1)

24 prompts, 6136 decode tokens, 965 prefill tokens, 23 MoE layers, E=128, k=8. Source: traces/routing_v1.npz.

## Usage skew (decode, all prompts pooled)

- Normalized entropy: mean 0.928 (1.0 = uniform), min layer 0.911
- Gini: mean 0.458 (0 = uniform)
- Traffic share of top 10% experts: 29.0% (uniform 10%); top 25%: 56.9% (uniform 25%)
- Within one sequence, its own top 25% experts carry 80.0% of its traffic

## Temporal locality (decode)

- Adjacent-token overlap |S_t ∩ S_t+1|/k: 43.8% (random 6.2%)
- Selected expert already used within last 1 tokens: 43.8% (random 6.2%)
- Selected expert already used within last 4 tokens: 65.0% (random 22.8%)
- Selected expert already used within last 16 tokens: 86.2% (random 64.4%)

## Domain clustering

- Cosine similarity of per-prompt expert usage: within domain 0.698, across domains 0.382

## Hybrid cache: P(all 8 selected experts are local) per token-layer

| fraction of experts held locally | random | global frequency (LOO) | same-domain frequency (LOO) | sequence's own (oracle) |
|---|---|---|---|---|
| 10% | 0.000 | 0.000 | 0.002 | 0.021 |
| 25% | 0.000 | 0.047 | 0.101 | 0.282 |
| 50% | 0.003 | 0.338 | 0.493 | 0.808 |
| 75% | 0.090 | 0.641 | 0.823 | 0.990 |

Mean missed experts per token-layer (of 8):

| fraction | random | global | domain | oracle |
|---|---|---|---|---|
| 10% | 7.17 | 5.70 | 4.97 | 3.81 |
| 25% | 5.97 | 3.71 | 2.74 | 1.60 |
| 50% | 4.01 | 1.75 | 0.91 | 0.26 |
| 75% | 1.99 | 0.72 | 0.23 | 0.01 |
