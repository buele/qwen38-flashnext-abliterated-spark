#!/usr/bin/env python3
"""orca 专家键 compressed-tensors → ModelOpt 改名(weight_packed→weight, weight_global_scale→weight_scale_2)"""
import json, struct, os, shutil

D = "/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
TMP = f"{D}/.rename-tmp"
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP, exist_ok=True)

def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n)), n + 8

def rename_key(k):
    if not (".mlp.experts." in k or ".experts." in k):
        return None
    if k.endswith(".weight_packed"):
        return k[:-len("weight_packed")] + "weight"
    if k.endswith(".weight_global_scale"):
        return k[:-len("weight_global_scale")] + "weight_scale_2"
    return None  # weight_scale 等保持原名(不处理)

nconv = 0
for pi in range(1, 18):
    p = f"{D}/model-{pi:05d}-of-00017.safetensors"
    if not os.path.exists(p):
        continue
    with open(p, "rb") as f:
        h, base = read_header(f)
        # 读全部张量字节(每片 ~5GB,流式逐张量拷)
        newh = {}
        renames = {}
        for k, v in h.items():
            if k == "__metadata__":
                newh[k] = v
                continue
            nk = rename_key(k)
            if nk:
                renames[k] = (nk, v)
                if k.endswith("weight_packed"):
                    newh[nk] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": v["data_offsets"]}
                else:
                    newh[nk] = {"dtype": "F32", "shape": [], "data_offsets": v["data_offsets"]}
            else:
                newh[k] = v
        # data_offsets 不变(数值原位,只改 header)——直接写新 header + 原数据块
        hj = json.dumps(newh).encode()
        outp = f"{TMP}/model-{pi:05d}-of-00017.safetensors"
        with open(outp, "wb") as fo:
            fo.write(struct.pack("<Q", len(hj)))
            fo.write(hj)
            f.seek(0)
            f.read(8 + base - 8)  # skip old header... 简化:直接拷剩余
        # 其实最稳:重读原文件,跳过 header,把 data blob 整块拷
        with open(p, "rb") as f2, open(outp, "wb") as fo:
            fo.write(struct.pack("<Q", len(hj)))
            fo.write(hj)
            f2.seek(base)
            while True:
                chunk = f2.read(1 << 30)
                if not chunk:
                    break
                fo.write(chunk)
    nconv += len(renames)
    print(f"shard {pi}: renamed {len(renames)} keys", flush=True)

# 替换原文件
for pi in range(1, 18):
    src = f"{TMP}/model-{pi:05d}-of-00017.safetensors"
    dst = f"{D}/model-{pi:05d}-of-00017.safetensors"
    if os.path.exists(src):
        os.replace(src, dst)
# 更新 index
IDX = f"{D}/model.safetensors.index.json"
idx = json.load(open(IDX))
wm = idx["weight_map"]
nidx = {}
for k, v in wm.items():
    if k.endswith(".weight_packed"):
        nidx[k[:-len("weight_packed")] + "weight"] = v
    elif k.endswith(".weight_global_scale"):
        nidx[k[:-len("weight_global_scale")] + "weight_scale_2"] = v
    else:
        nidx[k] = v
idx["weight_map"] = nidx
json.dump(idx, open(IDX, "w"))
print(f"DONE: {nconv} keys renamed, index updated ({len(nidx)} keys)")
