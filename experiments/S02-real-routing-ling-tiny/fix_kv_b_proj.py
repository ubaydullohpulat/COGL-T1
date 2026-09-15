"""Make rapid-mlx/Ling-3.0-tiny-MLX-4bit@328a497 loadable by mlx-lm@872ae88. The community build was made with a
different MLX port:
  1. attention.kv_b_proj is 4-bit quantized -> mlx-lm refuses ("Quantized Bailing V3 kv_b_proj weights are not
     supported"). Dequantize those 6 tensors to bfloat16 (same values the community quant would compute).
  2. Key names differ: mlp.experts.* -> mlp.switch_mlp.*, attention.{q,k,v}_conv1d.conv.weight -> {q,k,v}_conv1d.weight.
     Shapes are already MLX layout (conv [out, K, 1]; experts stacked [E, ...]).
The original directory stays untouched.
Run: ../../.venv/bin/python fix_kv_b_proj.py
"""
import json, os, shutil
import mlx.core as mx

SRC = "../../models/Ling-3.0-tiny-MLX-4bit"
DST = "../../models/Ling-3.0-tiny-MLX-4bit-kvbfix"
cfg = json.load(open(f"{SRC}/config.json"))
q = cfg["quantization"]
w = mx.load(f"{SRC}/model.safetensors")
keys = sorted(k[: -len(".weight")] for k in w if k.endswith("kv_b_proj.weight") and k[: -len(".weight")] + ".scales" in w)
for p in keys:
    w[p + ".weight"] = mx.dequantize(w[p + ".weight"], w.pop(p + ".scales"), w.pop(p + ".biases"),
                                     group_size=q["group_size"], bits=q["bits"]).astype(mx.bfloat16)
renamed = {}
for k in list(w):
    nk = k.replace(".mlp.experts.", ".mlp.switch_mlp.").replace("_conv1d.conv.weight", "_conv1d.weight")
    if nk != k:
        renamed[k] = nk
        w[nk] = w.pop(k)
print("renamed", len(renamed), "tensors")
os.makedirs(DST, exist_ok=True)
mx.save_safetensors(f"{DST}/model.safetensors", w, metadata={"format": "mlx"})
for f in os.listdir(SRC):
    if f not in ("model.safetensors", "model.safetensors.index.json") and os.path.isfile(f"{SRC}/{f}"):
        shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
print("dequantized", len(keys), "kv_b_proj tensors, e.g.", keys[:2])
