"""Topologies and Monte Carlo latency for decode (per token) and prefill (time to first token).

Topologies
  A_chain : pipeline; layers split contiguously over N peers; activations forwarded peer -> peer
            (client -> p1 -> ... -> pN -> client). Each peer holds attention + experts of its layers.
  A_relay : same placement, but the client relays every hop (Petals-style): client <-> each peer.
  B_star  : client (the user's own device) holds everything except routed experts: embeddings, attention,
            shared experts, router, head. Routed experts of every layer are split evenly over the holders.
            Per MoE layer the client fans out to the peers owning the selected experts, in parallel.
            If client_hosts_share, the client is itself one holder (every user also contributes).

Routing in v0 is SYNTHETIC: uniform random top-k per token per layer. That is the worst case for B_star
(no locality); real traces replace it in v1.
"""
from dataclasses import dataclass
import numpy as np

from .arch import ArchSpec
from .compute import ComputeProfile
from .network import LinkProfile, transit_s, serial_s

HEADER_BYTES = 64


@dataclass
class Scenario:
    arch: ArchSpec
    topology: str               # A_chain | A_relay | B_star
    n_peers: int                # remote peers (excludes the client)
    link: LinkProfile
    peer_compute: list          # ComputeProfile per remote peer (len == n_peers)
    client_compute: ComputeProfile
    client_hosts_share: bool = True
    bytes_per_value: float = 2.0  # activations on the wire: 2 = bf16, 1 = int8


def _per_layer_params(a: ArchSpec):
    embed_head = 2 * a.vocab * a.hidden
    non_routed_per_layer = (a.params_non_routed - embed_head) / (a.n_layers + a.n_mtp)
    return non_routed_per_layer, a.vocab * a.hidden  # (attention+shared+norms per layer, lm_head)


def memory_per_holder_gb(s: Scenario) -> dict:
    a = s.arch
    bits = s.client_compute.bits_per_weight
    gb = lambda n: n * bits / 8 / 1e9
    routed_main = a.n_moe * a.n_experts * a.params_per_expert
    if s.topology.startswith("A"):
        return {"client_gb": gb(2 * a.vocab * a.hidden), "peer_gb": gb(a.params_total - routed_main / a.n_moe * a.n_mtp) / s.n_peers}
    holders = s.n_peers + (1 if s.client_hosts_share else 0)
    client = gb(a.params_non_routed) + (gb(routed_main) / holders if s.client_hosts_share else 0)
    return {"client_gb": client, "peer_gb": gb(routed_main) / holders}


def decode_latency(s: Scenario, n_tokens: int, rng: np.random.Generator) -> dict:
    a, N, L = s.arch, s.n_peers, s.link
    nr_layer, head = _per_layer_params(a)
    msg = a.boundary_bytes(1, s.bytes_per_value) + HEADER_BYTES
    client_local = s.client_compute.time_for_params(head)  # lm_head on client
    comp = np.zeros(n_tokens) + client_local
    net = np.zeros(n_tokens)

    if s.topology in ("A_chain", "A_relay"):
        layers_per_peer = np.array_split(np.arange(a.n_layers), N)
        for i, layers in enumerate(layers_per_peer):
            pc = s.peer_compute[i]
            n_moe_here = sum(1 for l in layers if l >= a.n_dense)
            params = len(layers) * nr_layer + n_moe_here * a.top_k * a.params_per_expert
            comp += pc.time_for_params(params) + pc.call_overhead_s
        hops = N + 1 if s.topology == "A_chain" else 2 * N
        for _ in range(hops):
            net += transit_s(L, msg, rng, n_tokens) + serial_s(msg, L.up_mbps)
    elif s.topology == "B_star":
        holders = N + (1 if s.client_hosts_share else 0)
        comp += a.n_layers * s.client_compute.time_for_params(nr_layer)
        comp += a.n_dense * 0  # dense MLP included in nr_layer average
        layer_time = np.zeros(n_tokens)
        for _ in range(a.n_moe):
            # choose top_k distinct experts per token (uniform, synthetic)
            experts = np.argsort(rng.random((n_tokens, a.n_experts)), axis=1)[:, : a.top_k]
            owner = experts * holders // a.n_experts               # holder id per selected expert
            this_layer = np.zeros(n_tokens)
            for h in range(holders):
                k_h = (owner == h).sum(axis=1)
                hit = k_h > 0
                if s.client_hosts_share and h == 0:
                    local = s.client_compute.time_for_params(k_h * a.params_per_expert)
                    comp += local
                    continue
                pc = s.peer_compute[h - (1 if s.client_hosts_share else 0)]
                rtt = (transit_s(L, msg, rng, n_tokens) + transit_s(L, msg, rng, n_tokens)
                       + serial_s(msg, L.up_mbps) * 2)
                t = hit * (rtt + pc.time_for_params(k_h * a.params_per_expert) + pc.call_overhead_s)
                this_layer = np.maximum(this_layer, t)           # parallel fan-out: wait for slowest
            layer_time += this_layer
        net += layer_time  # includes remote expert compute (small); split reported below
    else:
        raise ValueError(s.topology)

    total = comp + net
    return {
        "mean_s": float(total.mean()),
        "p50_s": float(np.percentile(total, 50)),
        "p95_s": float(np.percentile(total, 95)),
        "tok_per_s": float(1 / total.mean()),
        "compute_share": float(comp.mean() / total.mean()),
    }


def prefill_latency(s: Scenario, n_prompt: int, rng: np.random.Generator, n_samples: int = 200) -> dict:
    """Time to process a prompt of n_prompt tokens (to first token). Expected-value routing, sampled network."""
    a, N, L = s.arch, s.n_peers, s.link
    nr_layer, head = _per_layer_params(a)
    act = a.boundary_bytes(n_prompt, s.bytes_per_value) + HEADER_BYTES
    total = np.zeros(n_samples)

    if s.topology in ("A_chain", "A_relay"):
        layers_per_peer = np.array_split(np.arange(a.n_layers), N)
        for i, layers in enumerate(layers_per_peer):
            n_moe_here = sum(1 for l in layers if l >= a.n_dense)
            params = len(layers) * nr_layer + n_moe_here * a.top_k * a.params_per_expert
            total += s.peer_compute[i].prefill_time(params, n_prompt)
        hops = N + 1 if s.topology == "A_chain" else 2 * N
        for _ in range(hops):
            total += transit_s(L, act, rng, n_samples) + serial_s(act, L.up_mbps)
    else:
        holders = N + (1 if s.client_hosts_share else 0)
        share = 1 / holders
        # P(token touches a given holder) with k distinct experts over evenly split experts (hypergeometric)
        from math import comb
        per = a.n_experts // holders
        p_touch = 1 - comb(a.n_experts - per, a.top_k) / comb(a.n_experts, a.top_k)
        tokens_per_peer = n_prompt * p_touch
        payload = a.boundary_bytes(tokens_per_peer, s.bytes_per_value) + HEADER_BYTES
        total += a.n_layers * s.client_compute.prefill_time(nr_layer, n_prompt)
        for _ in range(a.n_moe):
            # client uplink carries all outgoing payloads; each remote peer answers on its own uplink
            up = serial_s(payload * N, L.up_mbps)
            layer = np.zeros(n_samples)
            for i in range(N):
                pc = s.peer_compute[i]
                t = (transit_s(L, payload, rng, n_samples) + transit_s(L, payload, rng, n_samples)
                     + serial_s(payload, L.up_mbps)
                     + pc.prefill_time(a.top_k * share * a.params_per_expert, n_prompt))
                layer = np.maximum(layer, t)
            local = s.client_compute.prefill_time(a.top_k * share * a.params_per_expert, n_prompt) if s.client_hosts_share else 0
            total += up + np.maximum(layer, local)
    return {"ttft_mean_s": float(total.mean()), "ttft_p95_s": float(np.percentile(total, 95))}
