#!/usr/bin/env python3
"""orca g0 300层 FP8→BF16 dequant v2(修正模块名前缀匹配)"""
import json, struct, os, re, shutil
import numpy as np

SRC = "/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
TMP = f"{SRC}/.dequant-tmp"
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP, exist_ok=True)
CFG = json.load(open(f"{SRC}/config.json"))
g0 = CFG["quantization_config"]["config_groups"]["group_0"]["targets"]
# 模块名前缀匹配:target "X" 匹配键 "X.weight" / "X.weight_scale"
pats = []
for t in g0:
    if t.startswith("re:"):
        pats.append(re.compile(t[3:]))
    else:
        pats.append(re.compile("^" + re.escape(t) + r"(\.|$)"))
def is_g0(k):
    return any(p.search(k) for p in pats)

lut = np.arange(256, dtype=np.uint16)
sign = np.where(lut & 0x80, -1.0, 1.0)
exp = (lut >> 3) & 0xF; man = lut & 7
val = np.where(exp == 0, (man/8.0)*0.015625, (1+man/8.0)*np.exp2(exp.astype(np.float32)-7))
LUT = (sign * val).astype(np.float32)

def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n)), n + 8

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

nconv = 0
for pi in range(1, 18):
    p = f"{SRC}/model-{pi:05d}-of-00017.safetensors"
    outp = f"{TMP}/model-{pi:05d}-of-00017.safetensors"
    if os.path.exists(outp + ".ok"):
        continue
    if not os.path.exists(p):
        print(f"shard {pi}: source missing, skip", flush=True)
        continue
    with open(p, "rb") as f:
        h, base = read_header(f)
        pairs = []
        for k in h:
            if k == "__metadata__": continue
            if k.endswith(".weight") and h[k].get("dtype") == "F8_E4M3" and is_g0(k):
                if k + "_scale" in h:
                    pairs.append((k, k + "_scale"))
        wset = {wk for wk, _ in pairs}
        sset = {sk for _, sk in pairs}
        newh = {}; off = 0
        for k, v in h.items():
            if k == "__metadata__":
                newh[k] = v; continue
            if k in sset: continue
            dt = v["dtype"]; shape = v["shape"]
            nb = v["data_offsets"][1] - v["data_offsets"][0]
            if k in wset:
                dt = "BF16"; nb = nb * 2
            newh[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off+nb]}
            off += nb
        hj = json.dumps(newh).encode()
        with open(outp, "wb") as fo:
            fo.write(struct.pack("<Q", len(hj))); fo.write(hj)
            for k, v in h.items():
                if k == "__metadata__": continue
                if k in sset: continue
                f.seek(base + v["data_offsets"][0])
                raw = f.read(v["data_offsets"][1] - v["data_offsets"][0])
                if k in wset:
                    w = np.frombuffer(raw, dtype=np.uint8).reshape(v["shape"])
                    sk = k + "_scale"
                    f.seek(base + h[sk]["data_offsets"][0])
                    sc = np.frombuffer(f.read(h[sk]["data_offsets"][1]-h[sk]["data_offsets"][0]), dtype=np.uint16).reshape(h[sk]["shape"])
                    prod = (LUT[w] * bf16_to_f32(sc)).astype(np.float32)
                    b = prod.view(np.uint32)
                    bf = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)
                    fo.write(bf.tobytes()); nconv += 1
                else:
                    fo.write(raw)
    open(outp + ".ok", "w").write("ok")
    print(f"shard {pi}/17 done (cum {nconv} conv)", flush=True)

for pi in range(1, 18):
    src = f"{TMP}/model-{pi:05d}-of-00017.safetensors"
    dst = f"{SRC}/model-{pi:05d}-of-00017.safetensors"
    if os.path.exists(src):
        os.replace(src, dst)
        if os.path.exists(src + ".ok"): os.remove(src + ".ok")
print("replaced originals", flush=True)

CFG["quantization_config"]["config_groups"].pop("group_0", None)
json.dump(CFG, open(f"{SRC}/config.json", "w"), indent=1)
print("config: group_0 removed", flush=True)

idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = idx["weight_map"]
drop = [k for k in wm if k.endswith("_scale") and is_g0(k[:-len("_scale")])]
for k in drop: del wm[k]
json.dump(idx, open(f"{SRC}/model.safetensors.index.json", "w"))
print(f"index: dropped {len(drop)} scale keys", flush=True)
shutil.rmtree(TMP, ignore_errors=True)
print("DEQUANT DONE, converted:", nconv, flush=True)
