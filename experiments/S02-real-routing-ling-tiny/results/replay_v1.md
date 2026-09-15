# SwarmLab v1 replay — Ling-3.0-tiny, real routing + measured compute, 4 stage peers

Seed 20260916; 6136 real decode tokens replayed; single-machine decode 163.0 tok/s (MEASURED); component calibration ×0.54; remote call overhead 2 ms (ASSUMED). Links from S01 (ASSUMED except researcher).

## Decode tok/s (mean)

| setup | memory per stage peer | cold-pool peer | lan (1 ms) | metro (20 ms) | national (60 ms) | researcher (137 ms) | intercontinental (250 ms) |
|---|---|---|---|---|---|---|---|
| A_chain (all experts local) | 1.11 GB | 0.00 GB | 59.62 | 13.34 | 5.15 | 2.48 | 1.32 |
| B_star (experts spread) | 0.78 GB | 0.00 GB | 12.40 | 1.27 | 0.44 | 0.21 | 0.11 |
| C_hybrid, keep 90% hot experts | 1.01 GB | 0.10 GB | 33.07 | 5.22 | 1.93 | 0.91 | 0.48 |
| C_hybrid, keep 75% hot experts | 0.87 GB | 0.24 GB | 23.00 | 3.15 | 1.15 | 0.55 | 0.28 |
| C_hybrid, keep 50% hot experts | 0.62 GB | 0.49 GB | 15.13 | 1.88 | 0.67 | 0.32 | 0.16 |
| C_hybrid, keep 25% hot experts | 0.38 GB | 0.73 GB | 11.40 | 1.31 | 0.46 | 0.22 | 0.11 |
| C_hybrid, keep 50%, domain-aware cache | 0.62 GB | 0.49 GB | 18.39 | 2.45 | 0.89 | 0.42 | 0.22 |

## p95 seconds per token at `researcher` (137 ms)

| setup | mean s | p95 s |
|---|---|---|
| A_chain (all experts local) | 0.403 | 0.580 |
| B_star (experts spread) | 4.728 | 5.780 |
| C_hybrid, keep 90% hot experts | 1.099 | 3.105 |
| C_hybrid, keep 75% hot experts | 1.835 | 4.486 |
| C_hybrid, keep 50% hot experts | 3.142 | 5.264 |
| C_hybrid, keep 25% hot experts | 4.551 | 5.821 |
| C_hybrid, keep 50%, domain-aware cache | 2.388 | 4.399 |
