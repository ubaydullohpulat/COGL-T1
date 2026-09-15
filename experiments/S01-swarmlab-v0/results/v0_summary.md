# SwarmLab v0 results — Ling-3.0-flash

Seed 20260916, 2000 sampled tokens per cell, python 3.11.13, numpy 1.24.3, runtime 32s.
Routing SYNTHETIC (uniform random = no locality). Compute ASSUMED (client m4max, peers consumer_gpu). Links ASSUMED except `researcher` (partly MEASURED).

Single machine, no network (ASSUMED compute): **109.2 tok/s**

## Decode throughput (tok/s, mean) — rows: topology × remote peers; columns: link profile (median RTT)

| topology | peers | lan (1 ms) | metro (20 ms) | national (60 ms) | researcher (137 ms) | continental (150 ms) | intercontinental (250 ms) |
|---|---|---|---|---|---|---|---|
| A_chain | 2 | 58.88 | 18.38 | 7.96 | 3.89 | 3.51 | 2.09 |
| A_chain | 3 | 48.71 | 14.28 | 6.07 | 3.00 | 2.65 | 1.60 |
| A_chain | 4 | 41.54 | 11.56 | 4.82 | 2.40 | 2.11 | 1.27 |
| A_chain | 8 | 26.14 | 6.56 | 2.77 | 1.33 | 1.17 | 0.71 |
| A_relay | 2 | 57.05 | 14.79 | 6.22 | 3.00 | 2.68 | 1.59 |
| A_relay | 3 | 46.24 | 10.32 | 4.20 | 2.03 | 1.75 | 1.09 |
| A_relay | 4 | 38.90 | 7.96 | 3.14 | 1.52 | 1.34 | 0.81 |
| A_relay | 8 | 23.78 | 4.14 | 1.61 | 0.77 | 0.68 | 0.40 |
| B_star | 2 | 5.83 | 0.76 | 0.28 | 0.13 | 0.12 | 0.07 |
| B_star | 3 | 5.82 | 0.70 | 0.26 | 0.13 | 0.11 | 0.06 |
| B_star | 4 | 5.81 | 0.66 | 0.24 | 0.12 | 0.10 | 0.06 |
| B_star | 8 | 5.80 | 0.59 | 0.21 | 0.11 | 0.09 | 0.05 |

## p95 token latency (s) at `researcher` profile

| topology | peers | mean s | p95 s | compute share |
|---|---|---|---|---|
| A_chain | 2 | 0.257 | 0.431 | 6.0% |
| A_chain | 3 | 0.333 | 0.505 | 5.5% |
| A_chain | 4 | 0.417 | 0.600 | 5.1% |
| A_chain | 8 | 0.750 | 0.979 | 4.4% |
| A_relay | 2 | 0.333 | 0.512 | 4.6% |
| A_relay | 3 | 0.493 | 0.691 | 3.7% |
| A_relay | 4 | 0.657 | 0.884 | 3.3% |
| A_relay | 8 | 1.302 | 1.626 | 2.6% |
| B_star | 2 | 7.424 | 8.315 | 0.1% |
| B_star | 3 | 7.973 | 8.935 | 0.1% |
| B_star | 4 | 8.384 | 9.440 | 0.1% |
| B_star | 8 | 9.245 | 10.515 | 0.1% |

## Memory per holder (GB, 4.5-bit weights)

| topology | peers | client GB | each peer GB |
|---|---|---|---|
| A_chain | 2 | 0.5 | 35.0 |
| A_chain | 3 | 0.5 | 23.3 |
| A_chain | 4 | 0.5 | 17.5 |
| A_chain | 8 | 0.5 | 8.8 |
| A_relay | 2 | 0.5 | 35.0 |
| A_relay | 3 | 0.5 | 23.3 |
| A_relay | 4 | 0.5 | 17.5 |
| A_relay | 8 | 0.5 | 8.8 |
| B_star | 2 | 24.7 | 22.6 |
| B_star | 3 | 19.1 | 17.0 |
| B_star | 4 | 15.7 | 13.6 |
| B_star | 8 | 9.6 | 7.5 |

## Straggler: 4 peers, `researcher` links

| topology | peers | tok/s | p95 s |
|---|---|---|---|
| A_chain | 4x consumer_gpu | 2.38 | 0.616 |
| A_chain | 3x consumer_gpu + 1 slow_laptop | 2.32 | 0.622 |
| A_relay | 4x consumer_gpu | 1.53 | 0.880 |
| A_relay | 3x consumer_gpu + 1 slow_laptop | 1.50 | 0.899 |
| B_star | 4x consumer_gpu | 0.12 | 9.460 |
| B_star | 3x consumer_gpu + 1 slow_laptop | 0.12 | 9.548 |

## Prefill, 4 peers — time to first token (s, mean)

| profile | topology | prompt | bf16 wire | int8 wire |
|---|---|---|---|---|
| metro | A_chain | 1000 | 8.5 | 4.9 |
| metro | A_chain | 4000 | 30.9 | 17.2 |
| metro | A_relay | 1000 | 13.2 | 7.5 |
| metro | A_relay | 4000 | 47.9 | 26.0 |
| metro | B_star | 1000 | 245.3 | 131.4 |
| metro | B_star | 4000 | 930.2 | 475.2 |
| researcher | A_chain | 1000 | 8.9 | 5.4 |
| researcher | A_chain | 4000 | 31.2 | 17.5 |
| researcher | A_relay | 1000 | 13.7 | 8.3 |
| researcher | A_relay | 4000 | 48.4 | 26.5 |
| researcher | B_star | 1000 | 252.2 | 138.5 |
| researcher | B_star | 4000 | 937.1 | 482.0 |
| intercontinental | A_chain | 1000 | 12.9 | 7.8 |
| intercontinental | A_chain | 4000 | 45.5 | 25.0 |
| intercontinental | A_relay | 1000 | 20.3 | 12.1 |
| intercontinental | A_relay | 4000 | 71.3 | 38.5 |
| intercontinental | B_star | 1000 | 382.6 | 211.6 |
| intercontinental | B_star | 4000 | 1408.7 | 726.0 |

## Link profiles

| name | RTT ms | jitter σ | loss | up/down Mbps | evidence |
|---|---|---|---|---|---|
| lan | 1 | 0.1 | 0.0 | 1000/1000 | ASSUMED: wired home LAN |
| metro | 20 | 0.2 | 0.001 | 30/100 | ASSUMED: same city, cable/fiber |
| national | 60 | 0.25 | 0.003 | 30/100 | ASSUMED: same country |
| researcher | 137 | 0.25 | 0.005 | 30/72 | MEASURED 2026-09-15: base RTT 137 ms to Apple server, 30/72 Mbps (not peer-to-peer) |
| continental | 150 | 0.3 | 0.005 | 20/100 | ASSUMED: cross-continent |
| intercontinental | 250 | 0.3 | 0.01 | 20/100 | ASSUMED: between continents |
