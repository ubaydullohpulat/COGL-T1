"""Architecture spec derived from a Hugging Face config.json (Ling-3.0 / Bailing-V3 naming)."""
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class ArchSpec:
    name: str
    n_layers: int            # main decoder layers (excludes MTP)
    n_dense: int             # first_k_dense_replace
    hidden: int
    n_experts: int
    top_k: int
    n_shared: int
    expert_ffn: int
    vocab: int
    params_total: int        # CITED from HF safetensors metadata (includes MTP layer)
    n_mtp: int

    @property
    def n_moe(self) -> int:
        return self.n_layers - self.n_dense

    @property
    def params_per_expert(self) -> int:  # gated FFN: gate, up, down
        return 3 * self.hidden * self.expert_ffn

    @property
    def params_routed(self) -> int:      # routed experts incl. MTP layer experts
        return (self.n_moe + self.n_mtp) * self.n_experts * self.params_per_expert

    @property
    def params_non_routed(self) -> int:  # attention, shared experts, dense MLP, embeddings, head, norms
        return self.params_total - self.params_routed

    def boundary_bytes(self, n_tokens: int, bytes_per_value: float = 2.0) -> float:
        """Activation payload for one hidden-state tensor crossing the wire."""
        return n_tokens * self.hidden * bytes_per_value


def load(config_path: str, params_total: int, name: str) -> ArchSpec:
    c = json.load(open(config_path))
    c = c.get("text_config", c)
    return ArchSpec(
        name=name,
        n_layers=c["num_hidden_layers"],
        n_dense=c.get("first_k_dense_replace", 0),
        hidden=c["hidden_size"],
        n_experts=c["num_experts"],
        top_k=c["num_experts_per_tok"],
        n_shared=c.get("num_shared_experts", 0),
        expert_ffn=c["moe_intermediate_size"],
        vocab=c["vocab_size"],
        params_total=params_total,
        n_mtp=c.get("num_nextn_predict_layers", 0),
    )
