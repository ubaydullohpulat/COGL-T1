"""Compute-time model for one peer. v0: memory-bandwidth-bound decode + fixed per-call overhead (all ASSUMED)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeProfile:
    name: str
    mem_bandwidth_gbps: float   # effective GB/s of weights touched per second in decode
    call_overhead_s: float      # per RPC: deserialize, dispatch kernels, serialize
    bits_per_weight: float      # weight precision on this peer
    prefill_flops: float        # effective FLOPs/s for batched prefill
    evidence: str

    def prefill_time(self, n_params_active: float, n_tokens: float) -> float:
        return 2 * n_params_active * n_tokens / self.prefill_flops

    def time_for_params(self, n_params_touched: float) -> float:
        return n_params_touched * self.bits_per_weight / 8 / 1e9 / self.mem_bandwidth_gbps


COMPUTE = {
    "m4max": ComputeProfile("m4max", 400, 0.002, 4.5, 8e12, "ASSUMED: ~546 GB/s spec, ~75% effective; 2 ms call overhead; 8e12 FLOPs/s prefill"),
    "consumer_gpu": ComputeProfile("consumer_gpu", 300, 0.003, 4.5, 1.5e13, "ASSUMED: RTX 4070-class"),
    "slow_laptop": ComputeProfile("slow_laptop", 60, 0.010, 4.5, 5e11, "ASSUMED: CPU / low-end iGPU"),
}
