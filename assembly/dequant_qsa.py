#!/usr/bin/env python3
"""QSA 层 F8 q_proj → BF16 dequant(裸 FP8,直接转 float)"""
import json, struct, os, shutil
import numpy as np

D = "/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"

lut = np.arange(256, dtype=np.uint16)
sign = np.where(lut & 0x80, -1.0, 1.0)
exp = (lut >> 3) & 0xF
man = lut & 7
val = np.where(exp == 0, (man/8.0)*0.015625, (1+man/8.0)*np.exp2(exp.astype(np.float32)-7))
LUT = (sign * val).astype(np.float32)

def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n)), n + 8

idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
# 找全部 F8 的 self_attn q_proj
f8_keys = []
for k, f in idx.items():
    if not (k.endswith("self_attn.q_proj.weight") or k.endswith("self_attn.k_proj.weight") or k.endswith("self_attn.v_proj.weight")):
        continue
    if ".mtp." in k:
        continue
    fh = open(f"{D}/{f}", "rb")
    h, _ = read_header(fh)
    if h.get(k, {}).get("dtype") == "F8_E4M3":
        f8_keys.append((k, f))
print("F8 qkv keys:", len(f8_keys))

# 按文件分组重写
from collections import defaultdict
by_file = defaultdict(list)
for k, f in f8_keys:
    by_file[f].append(k)

for f, keys in by_file.items():
    p = f"{D}/{f}"
    with open(p, "rb") as fh:
        h, base = read_header(fh)
        # 读出所有张量,重写 F8 → BF16
        tensors = {}
        for k, v in h.items():
            if k == "__metadata__": continue
            fh.seek(base + v["data_offsets"][0])
            tensors[k] = (fh.read(v["data_offsets"][1] - v["data_offsets"][0]), v["dtype"], v["shape"])
    newh = {}
    payload = []
    off = 0
    for k in sorted(tensors):
        raw, dt, shape = tensors[k]
        if k in keys:
            w = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            f32 = LUT[w]
            u32 = f32.view(np.uint32)
            bf = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
            data = bf.tobytes()
            newh[k] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off+len(data)]}
        else:
            data = raw
            newh[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off+len(data)]}
        payload.append(data)
        off += len(data)
    hj = json.dumps(newh).encode()
    with open(p, "wb") as fo:
        fo.write(struct.pack("<Q", len(hj)))
        fo.write(hj)
        for d in payload:
            fo.write(d)
    print(f"{f}: {len(keys)} F8→BF16", flush=True)
print("QSA DEQUANT DONE")
