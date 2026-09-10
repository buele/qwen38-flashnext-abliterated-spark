#!/usr/bin/env python3
"""model-mtp 合并专家键拆包 → 逐专家键(vLLM ModelOpt 兼容)"""
import json, struct, os
import numpy as np

D = "/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
P = f"{D}/model-mtp.safetensors"

def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n)), n + 8

def f32_to_bf16(u32):
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)

with open(P, "rb") as f:
    h, base = read_header(f)
    tensors = {}
    for k, v in h.items():
        if k == "__metadata__": continue
        f.seek(base + v["data_offsets"][0])
        tensors[k] = (f.read(v["data_offsets"][1] - v["data_offsets"][0]), v["dtype"], v["shape"])

newh = {}
payload = []
off = 0
N_EXP = 512
for k in sorted(tensors):
    raw, dt, shape = tensors[k]
    if k.endswith("mlp.experts.down_proj"):
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)  # bf16 [512,2560,640]
        for i in range(N_EXP):
            d = arr[i].tobytes()
            nk = f"{k}.{i}.down_proj.weight"
            newh[nk] = {"dtype": "BF16", "shape": [shape[1], shape[2]], "data_offsets": [off, off+len(d)]}
            payload.append(d); off += len(d)
    elif k.endswith("mlp.experts.gate_up_proj"):
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)  # bf16 [512,1280,2560]
        half = shape[1] // 2
        for i in range(N_EXP):
            g = arr[i][:half].tobytes()
            nk_g = f"{k}.{i}.gate_proj.weight"
            newh[nk_g] = {"dtype": "BF16", "shape": [half, shape[2]], "data_offsets": [off, off+len(g)]}
            payload.append(g); off += len(g)
            u = arr[i][half:].tobytes()
            nk_u = f"{k}.{i}.up_proj.weight"
            newh[nk_u] = {"dtype": "BF16", "shape": [shape[1]-half, shape[2]], "data_offsets": [off, off+len(u)]}
            payload.append(u); off += len(u)
    else:
        newh[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off+len(raw)]}
        payload.append(raw); off += len(raw)

out = f"{D}/model-mtp.safetensors.new"
hj = json.dumps(newh).encode()
with open(out, "wb") as fo:
    fo.write(struct.pack("<Q", len(hj))); fo.write(hj)
    for d in payload: fo.write(d)
os.replace(out, P)
print(f"拆包完成: {len(newh)} 键(原 {len(tensors)})")

# 更新 index
IDX = f"{D}/model.safetensors.index.json"
idx = json.load(open(IDX))
wm = idx["weight_map"]
for old in ["model.language_model.mtp.layers.0.mlp.experts.down_proj", "model.language_model.mtp.layers.0.mlp.experts.gate_up_proj"]:
    if old in wm: del wm[old]
for i in range(N_EXP):
    wm[f"model.language_model.mtp.layers.0.mlp.experts.{i}.down_proj.weight"] = "model-mtp.safetensors"
    wm[f"model.language_model.mtp.layers.0.mlp.experts.{i}.gate_proj.weight"] = "model-mtp.safetensors"
    wm[f"model.language_model.mtp.layers.0.mlp.experts.{i}.up_proj.weight"] = "model-mtp.safetensors"
json.dump(idx, open(IDX, "w"))
print("index 更新: +1536 逐专家键")
