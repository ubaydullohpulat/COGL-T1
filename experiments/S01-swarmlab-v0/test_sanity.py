"""Sanity checks against hand-computable cases. Run: python3 test_sanity.py"""
from dataclasses import replace
from math import comb
import numpy as np

from swarmlab import arch, sim
from swarmlab.compute import COMPUTE
from swarmlab.network import PROFILES, LinkProfile, serial_s, transit_s

CFG = "../P00-proxy-selection/configs/inclusionAI__Ling-3.0-flash.json"
LING = arch.load(CFG, params_total=127_486_405_600, name="Ling-3.0-flash")
rng = np.random.default_rng(0)
zero = LinkProfile("zero", 0, 0, 0, 0, 0, 1e9, 1e9, "test")


def close(a, b, tol):
    assert abs(a - b) <= tol * max(abs(b), 1e-12), (a, b)


# 1. architecture: routed params = 41 layers (40 MoE + 1 MTP) * 512 * 3*2560*768
assert LING.n_moe == 40 and LING.n_mtp == 1
assert LING.params_routed == 41 * 512 * 3 * 2560 * 768
assert 3e9 < LING.params_non_routed < 5e9, LING.params_non_routed

# 2. serialization: 1 MB over 8 Mbps = 1 s
close(float(serial_s(1e6, 8)), 1.0, 1e-9)

# 3. no jitter/loss: transit = rtt/2 exactly
p = LinkProfile("fixed", 100, 0, 0, 0, 0, 1e9, 1e9, "test")
close(float(transit_s(p, 1000, rng, 10).mean()), 0.05, 1e-9)

# 4. zero network: A_chain decode latency == pure compute, identical across N (same total work) up to overheads
m = COMPUTE["m4max"]
r2 = sim.decode_latency(sim.Scenario(LING, "A_chain", 2, zero, [m] * 2, m), 200, rng)
r8 = sim.decode_latency(sim.Scenario(LING, "A_chain", 8, zero, [m] * 8, m), 200, rng)
close(r8["mean_s"] - r2["mean_s"], 6 * m.call_overhead_s, 0.01)

# 5. fixed 100 ms RTT, A_relay with N peers: 2N one-way hops of 50 ms dominate
r = sim.decode_latency(sim.Scenario(LING, "A_relay", 4, p, [m] * 4, m), 200, rng)
assert 0.40 < r["mean_s"] < 0.45, r

# 6. B_star fixed RTT: every MoE layer waits one RTT whenever any remote expert is hit
r = sim.decode_latency(sim.Scenario(LING, "B_star", 4, p, [m] * 4, m, client_hosts_share=True), 500, rng)
per = 512 // 5
p_all_local = comb(per, 8) / comb(512, 8)  # all 8 experts on the client: ~0
assert p_all_local < 1e-4
close(r["mean_s"], 40 * 0.1, 0.1)

print("all sanity checks passed")
