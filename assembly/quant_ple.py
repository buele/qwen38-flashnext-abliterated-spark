#!/usr/bin/env python3
"""PLE FP8→NVFP4 v3: 每 shard 一个进程,显式并行"""
import json, struct, os, time, re, sys
from multiprocessing import Process
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

SRC = "/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE"
DST = "/home/mwyzs/models/ple-nvfp4-orca"
TMP = f"{DST}/.tmp"
KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"

os.makedirs(DST, exist_ok=True); os.makedirs(TMP, exist_ok=True)
E2M1 = np.array([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], dtype=np.float32)
E4M3_GRIDS = np.array(sorted([(1+m/8)*2**(e-7) for e in range(1,15) for m in range(8)] + [(m/8)*0.015625 for m in range(1,8)]))

def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n)), n + 8

_E4M3_LUT = None
def _e4m3_lut():
    global _E4M3_LUT
    if _E4M3_LUT is None:
        b = np.arange(256, dtype=np.uint16)
        sign = np.where(b & 0x80, -1.0, 1.0)
        exp = (b >> 3) & 0xF
        man = b & 7
        val = np.where(exp == 0, (man/8.0)*0.015625, (1+man/8.0)*np.exp2(exp.astype(np.float32)-7))
        _E4M3_LUT = (sign * val).astype(np.float32)
    return _E4M3_LUT

def e4m3_to_float(arr):
    return _e4m3_lut()[arr]

def scan_shard(p, key):
    with open(f"{SRC}/{p}", "rb") as f:
        h, base = read_header(f)
        v = h[key]
        n, d = v["shape"]
        f.seek(base + v["data_offsets"][0])
        raw = np.frombuffer(f.read(n*d), dtype=np.uint8).reshape(n, d)
        return float(np.abs(e4m3_to_float(raw)).max())

def encode_one(args):
    sid, glob, p, key = args
    outp = f"{TMP}/{sid:03d}"
    if os.path.exists(outp + ".packed.npy"): return sid
    with open(f"{SRC}/{p}", "rb") as f:
        h, base = read_header(f)
        v = h[key]
        n, d = v["shape"]
        f.seek(base + v["data_offsets"][0])
        raw = np.frombuffer(f.read(n*d), dtype=np.uint8).reshape(n, d)
        block = e4m3_to_float(raw) * np.float32(glob)
    r = n
    blocks = block.reshape(r, 10, 16)
    amax = np.abs(blocks).max(axis=2)
    sc = E4M3_GRIDS[np.searchsorted(E4M3_GRIDS, np.maximum(amax/(6.0*glob), 1e-30))]
    q = blocks / (sc*glob)[:, :, None]
    aq = np.abs(q)
    mag = np.digitize(aq, np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32))  # 0..7
    sgn = (q < 0).astype(np.uint8)
    codes = (mag + (sgn << 3)).astype(np.uint8)
    flat = codes.reshape(r, 160)
    packed = ((flat[:, 1::2] & 0xF).astype(np.uint8) << 4) | (flat[:, 0::2] & 0xF)
    sv = sc
    e_unb = np.floor(np.log2(np.maximum(sv, 1e-38))).astype(np.int32)
    be = np.clip(e_unb + 7, 1, 14)  # 偏置指数(bias 7),normal exp∈[1,14]
    man = np.clip(np.round(8.0 * (sv / np.exp2(e_unb.astype(np.float32)) - 1.0)), 0, 7).astype(np.int32)
    sub = sv < 0.015625  # E4M3 normal 最小 2^-6;低于此必须 subnormal
    mans = np.clip(np.round(sv / 0.015625 * 8).astype(np.int32), 1, 7)
    efin = np.where(sub, 0, be); mfin = np.where(sub, mans, man)
    sb = ((efin << 3) | mfin).astype(np.uint8)
    np.save(outp + ".packed.npy", packed)
    np.save(outp + ".scales.npy", sb)
    return sid

# ---- main ----
mode = sys.argv[1] if len(sys.argv) > 1 else "all"

# 文件→shard 映射
file_tabs = {}
for p in sorted(os.listdir(SRC)):
    if not p.startswith("model-plefp8-"): continue
    with open(f"{SRC}/{p}", "rb") as f:
        h, _ = read_header(f)
        for k in h:
            m = re.search(r"shard_(\d+)\.weight$", k)
            if m: file_tabs[(int(m.group(1)), p)] = k

# global fp8 scale
with open(f"{SRC}/model-plefp8-00009.safetensors", "rb") as f:
    h, base = read_header(f)
    gk = [k for k in h if k.endswith("weight_scale")][0]
    f.seek(base + h[gk]["data_offsets"][0])
    gfp8 = float(np.frombuffer(f.read(2), dtype=np.float16)[0])  # dtype BF16 实为 FP16

if mode == "scan":
    amax_all = 0.0
    for (sid, p), k in sorted(file_tabs.items()):
        a = scan_shard(p, k)
        amax_all = max(amax_all, a)
        if sid % 16 == 0: print(f"  shard {sid}: amax so far {amax_all*gfp8:.4g}", flush=True)
    glob = (amax_all * gfp8) / 448.0
    print(f"AMAX={amax_all*gfp8!r} GLOB={glob!r}")
    with open(f"{DST}/.glob.json", "w") as f:
        json.dump({"amax": amax_all*gfp8, "glob": glob}, f)
    sys.exit(0)

# load or scan glob
if os.path.exists(f"{DST}/.glob.json"):
    g = json.load(open(f"{DST}/.glob.json"))
    glob = g["glob"]
    print(f"glob from cache: {glob:.4g}")
else:
    print("no .glob.json — run scan first", flush=True); sys.exit(1)

if mode == "test":
    # 单 shard 验证
    sid = 0
    (p, k) = [(p, k) for (s, p), k in file_tabs.items() if s == 0][0]
    t0 = time.time()
    encode_one((sid, glob, p, k))
    pk = np.load(f"{TMP}/000.packed.npy"); sb = np.load(f"{TMP}/000.scales.npy")
    print(f"test shard0: packed{pk.shape} scales{sb.shape} {time.time()-t0:.1f}s")
    # 抽样反量化对比源
    with open(f"{SRC}/{p}", "rb") as f:
        h, base = read_header(f)
        v = h[k]; n, d = v["shape"]
        f.seek(base + v["data_offsets"][0])
        raw = np.frombuffer(f.read(n*d), dtype=np.uint8).reshape(n, d)
    src = e4m3_to_float(raw) * np.float32(gfp8)
    rows = np.random.default_rng(0).integers(0, n, 200)
    codes = np.stack(((pk[rows] & 0xF), (pk[rows] >> 4)), axis=-1).reshape(len(rows), 160)
    sc8 = e4m3_to_float(sb[rows]) * np.float32(glob)
    deq = E2M1[codes.astype(np.int64)] * np.repeat(sc8, 16, axis=1)
    err = np.abs(deq - src[rows])
    rel = err.max() / np.abs(src[rows]).max()
    cos = (deq*src[rows]).sum() / (np.linalg.norm(deq)*np.linalg.norm(src[rows]))
    print(f"质量: max_rel_err={rel:.4%} cos={cos:.6f}")
    sys.exit(0)

# all: 128 shard,分批 32 进程
tasks = [(sid, glob, p, k) for (sid, p), k in sorted(file_tabs.items())]
BATCH = 12
t0 = time.time()
for bi in range(0, len(tasks), BATCH):
    batch = tasks[bi:bi+BATCH]
    procs = []
    for t in batch:
        pr = Process(target=encode_one, args=(t,))
        pr.start(); procs.append(pr)
    for pr in procs: pr.join()
    print(f"batch {bi//BATCH+1}/{(len(tasks)+BATCH-1)//BATCH} done ({time.time()-t0:.0f}s)", flush=True)

# 聚合写 8 大分片
ROWS = 2500012; GROUP = 16
for gi in range(8):
    okf = f"{DST}/ple-nvfp4-{gi+1:05d}-of-00008.safetensors.ok"
    if os.path.exists(okf): continue
    pks = [np.load(f"{TMP}/{s:03d}.packed.npy") for s in range(gi*GROUP, (gi+1)*GROUP)]
    sbs = [np.load(f"{TMP}/{s:03d}.scales.npy") for s in range(gi*GROUP, (gi+1)*GROUP)]
    out_p, out_s = np.concatenate(pks), np.concatenate(sbs)
    hf = {f"{KEY}.nvfp4_global": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}
    off = 4
    hf[f"{KEY}.nvfp4_shard_{gi}.scales"] = {"dtype": "F8_E4M3", "shape": list(out_s.shape), "data_offsets": [off, off+out_s.nbytes]}; off += out_s.nbytes
    hf[f"{KEY}.nvfp4_shard_{gi}.packed"] = {"dtype": "U8", "shape": list(out_p.shape), "data_offsets": [off, off+out_p.nbytes]}; off += out_p.nbytes
    hj = json.dumps(hf).encode()
    with open(f"{DST}/ple-nvfp4-{gi+1:05d}-of-00008.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        f.write(struct.pack("<f", glob))
        f.write(out_s.tobytes()); f.write(out_p.tobytes())
    open(okf, "w").write("ok")
    del pks, sbs, out_p, out_s
    print(f"big {gi+1}/8 written ({time.time()-t0:.0f}s)", flush=True)
print("QUANT DONE", time.time()-t0, flush=True)
