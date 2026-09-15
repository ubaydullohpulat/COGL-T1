"""Fetch config.json + metadata for candidate proxy models at their current HF revision.

Writes configs/<org>__<name>.json (raw config) and candidates_meta.json (sha, params, license).
Usage: python3 fetch_configs.py
"""
import json, pathlib, urllib.request

CANDIDATES = [
    "moonshotai/Kimi-K3",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "moonshotai/Moonlight-16B-A3B-Instruct",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3.8-Flash-Next",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "zai-org/GLM-4.7-Flash",
    "zai-org/GLM-4.5-Air",
    "inclusionAI/Ling-3.0-tiny",
    "inclusionAI/Ling-3.0-flash",
    "inclusionAI/Ling-mini-2.0",
    "wdlctc/open-attnres-0.6b-block",
]

HERE = pathlib.Path(__file__).parent


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


meta = {}
for repo in CANDIDATES:
    try:
        info = json.loads(get(f"https://huggingface.co/api/models/{repo}?expand[]=sha&expand[]=safetensors&expand[]=cardData&expand[]=lastModified&expand[]=gated"))
        sha = info["sha"]
        cfg = get(f"https://huggingface.co/{repo}/resolve/{sha}/config.json")
        (HERE / "configs" / (repo.replace("/", "__") + ".json")).write_bytes(cfg)
        st = info.get("safetensors") or {}
        card = info.get("cardData") or {}
        meta[repo] = {
            "sha": sha,
            "lastModified": info.get("lastModified"),
            "params_total": st.get("total"),
            "params_by_dtype": st.get("parameters"),
            "license": card.get("license"),
            "license_name": card.get("license_name"),
            "gated": info.get("gated"),
        }
        print("ok  ", repo, sha[:7], st.get("total"))
    except Exception as e:  # keep going; record the failure
        meta[repo] = {"error": repr(e)}
        print("FAIL", repo, e)

(HERE / "candidates_meta.json").write_text(json.dumps(meta, indent=2))
