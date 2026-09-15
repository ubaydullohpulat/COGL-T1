"""Home-internet link model: RTT with heavy-tailed jitter, packet loss -> retransmit stalls, uplink serialization."""
from dataclasses import dataclass
import numpy as np

MTU_PAYLOAD = 1400  # bytes per packet
RTO_MIN_S = 0.2     # TCP minimum retransmission timeout (typical Linux/macOS default)


@dataclass(frozen=True)
class LinkProfile:
    name: str
    rtt_ms: float          # median peer-to-peer round-trip time
    jitter_sigma: float    # lognormal sigma on RTT (0.2 ~ +/-20% typical spread)
    spike_p: float         # probability a message hits a latency spike (bufferbloat, Wi-Fi retries)
    spike_ms: float        # scale of a spike (Pareto, alpha=2)
    loss: float            # per-packet loss probability
    up_mbps: float         # peer uplink
    down_mbps: float       # peer downlink
    evidence: str          # ASSUMED / MEASURED, with note


def transit_s(p: LinkProfile, nbytes, rng: np.random.Generator, size=None) -> np.ndarray:
    """One-way propagation + latency spikes + loss stalls for a message of `nbytes` (excludes serialization)."""
    rtt = p.rtt_ms / 1000 * rng.lognormal(0.0, p.jitter_sigma, size)
    spikes = (rng.random(size) < p.spike_p) * (p.spike_ms / 1000) * (rng.pareto(2.0, size) + 1)
    packets = np.maximum(1, np.ceil(np.asarray(nbytes, dtype=float) / MTU_PAYLOAD))
    p_any_loss = 1 - (1 - p.loss) ** packets
    stalls = (rng.random(size) < p_any_loss) * np.maximum(RTO_MIN_S, rtt)
    return rtt / 2 + spikes + stalls


def serial_s(nbytes, mbps: float):
    """Time to push `nbytes` through a link of `mbps`."""
    return np.asarray(nbytes, dtype=float) * 8 / (mbps * 1e6)


PROFILES = {
    "lan": LinkProfile("lan", 1, 0.1, 0.0, 0, 0.0, 1000, 1000, "ASSUMED: wired home LAN"),
    "metro": LinkProfile("metro", 20, 0.2, 0.01, 30, 0.001, 30, 100, "ASSUMED: same city, cable/fiber"),
    "national": LinkProfile("national", 60, 0.25, 0.02, 50, 0.003, 30, 100, "ASSUMED: same country"),
    "researcher": LinkProfile("researcher", 137, 0.25, 0.02, 80, 0.005, 30, 72,
                              "MEASURED 2026-09-15: base RTT 137 ms to Apple server, 30/72 Mbps (not peer-to-peer)"),
    "continental": LinkProfile("continental", 150, 0.3, 0.03, 100, 0.005, 20, 100, "ASSUMED: cross-continent"),
    "intercontinental": LinkProfile("intercontinental", 250, 0.3, 0.03, 150, 0.01, 20, 100, "ASSUMED: between continents"),
}
