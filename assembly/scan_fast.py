
import json, struct, os, re
import numpy as np
SRC = "/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE"
DST = "/home/mwyzs/models/ple-nvfp4-orca"

def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n)), n + 8

glob_max_byte = 0
for p in sorted(os.listdir(SRC)):
    if not p.startswith("model-plefp8-"): continue
    with open(f"{SRC}/{p}", "rb") as f:
        h, base = read_header(f)
        for k, v in h.items():
            m = re.search(r"shard_(\d+)\.weight$", k)
            if not m: continue
            n, d = v["shape"]
            f.seek(base + v["data_offsets"][0])
            mb = 0
            remaining = n*d
            while remaining > 0:
                chunk = np.frombuffer(f.read(min(remaining, 1<<30)), dtype=np.uint8)
                mb = max(mb, int(chunk.max()))
                remaining -= len(chunk)
            glob_max_byte = max(glob_max_byte, mb)
print("max byte:", glob_max_byte, hex(glob_max_byte))

def e4m3_abs(b):
    e = (b >> 3) & 0xF; man = b & 7
    if e == 0: return (man/8.0)*0.015625
    if e == 15: return 448.0
    return (1+man/8.0)*2**(e-7)
amax_raw = e4m3_abs(glob_max_byte)

with open(f"{SRC}/model-plefp8-00009.safetensors", "rb") as f:
    h, base = read_header(f)
    gk = [k for k in h if k.endswith("weight_scale")][0]
    f.seek(base + h[gk]["data_offsets"][0])
    graw = np.frombuffer(f.read(2), dtype=np.uint16)[0]
    gfp8 = np.frombuffer(struct.pack("<H", graw) + b"\x00\x00", dtype=np.float32)[0]
amax_real = amax_raw * gfp8
glob_nvfp4 = amax_real / 448.0
print(f"amax(real)={amax_real:.6g} glob_nvfp4={glob_nvfp4:.6g}")
json.dump({"amax": float(amax_real), "glob": float(glob_nvfp4), "gfp8": float(gfp8)}, open(f"{DST}/.glob.json", "w"))
print("SCAN-FAST-DONE")
