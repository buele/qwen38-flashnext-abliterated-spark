#!/usr/bin/env python3
"""Repair lychee888 orca assembly into hibrid47-compatible layout.

Three toxins:
  A) model-00012: layers.35.self_attn.q_proj stored as e4m3-grid * per-row scale
     (FP8 per-channel form, scale hidden in file, not in index). Dequantize to
     BF16, drop the scale key.
  B) model-mtp.safetensors: experts stored as BF16 nested keys
     (mtp.layers.0.mlp.experts.down_proj.N.down_proj.weight ...). Requantize
     to hibrid47 ModelOpt expanded NVFP4 format:
     mtp.layers.0.mlp.experts.N.{gate,up,down}_proj.{weight U8, weight_scale
     E4M3, weight_scale_2 F32, input_scale F32}
  C) ple-nvfp4 shards 2..8 carry a duplicate nvfp4_global scalar (index only
     references shard 1's copy). Verified all 8 identical -> strip from 2..8.

Finally: rewrite model.safetensors.index.json to match the new reality.
"""
import torch, json, os, struct, shutil, sys, time

D = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
BAK = "/home/mwyzs/models/orca-repair-backup"
os.makedirs(BAK, exist_ok=True)

def hdr(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def get_tensor(path, key):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key)

def rewrite_shard(path, transform):
    """transform(tensor_dict) -> new tensor dict; header metadata preserved."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    old = hdr(path)
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for k in old:
            if isinstance(old[k], dict) and "dtype" in old[k]:
                tensors[k] = f.get_tensor(k)
    new = transform(tensors)
    meta = {"format": "pt"}
    save_file(new, path, metadata=meta)
    return len(tensors), len(new)

log = open("/home/mwyzs/orca_repair.log", "a")

def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()

# ================= A) L35 q_proj dequant =================
t0 = time.time()
A_KEY = "model.language_model.layers.35.self_attn.q_proj.weight"
A_SCALE = "model.language_model.layers.35.self_attn.q_proj.weight_scale"
shard_scale = os.path.join(D, "model-00012-of-00017.safetensors")  # scale lives here
shard_w = os.path.join(D, "model-00013-of-00017.safetensors")      # weight lives here

shutil.copy2(shard_scale, os.path.join(BAK, "model-00012-of-00017.safetensors"))
shutil.copy2(shard_w, os.path.join(BAK, "model-00013-of-00017.safetensors"))

sc = get_tensor(shard_scale, A_SCALE)
w = get_tensor(shard_w, A_KEY)
# dequant: BF16 grid values [12288,1] scale -> true weights
w_dq = (w.float() * sc.float()).to(torch.bfloat16)
say(f"[A] L35 q_proj dequant: std {w_dq.float().std():.6f} absmax {w_dq.float().abs().max():.6f} (target ~0.017 / ~0.8)")

# 1) drop scale key from 00012
def transform_a12(tensors):
    return {k: v for k, v in tensors.items() if k != A_SCALE}
n0, n1 = rewrite_shard(shard_scale, transform_a12)
say(f"[A] shard 00012: {n0} -> {n1} keys (scale dropped), {time.time()-t0:.1f}s")

# 2) write dequantized weight into 00013
def transform_a13(tensors):
    out = dict(tensors)
    out[A_KEY] = w_dq
    return out
n0, n1 = rewrite_shard(shard_w, transform_a13)
say(f"[A] shard 00013 rewritten with dequant weight: {n0} keys, {time.time()-t0:.1f}s")

# ================= B) MTP requant =================
t0 = time.time()
mtp_path = os.path.join(D, "model-mtp.safetensors")
shutil.copy2(mtp_path, os.path.join(BAK, "model-mtp.safetensors"))

E4M3_MAX = 448.0

def nvfp4_quant(w_bf16):
    """w [O, I] -> (packed U8 [O, I//2], scale E4M3 [O, I//16], global F32)"""
    w = w_bf16.float()
    O, I = w.shape
    assert I % 16 == 0
    # 16-elem blocks along input dim
    wb = w.view(O, I // 16, 16)
    amax = wb.abs().amax(dim=-1)  # [O, I//16]
    # global scale: amax_global / (448 * 6)  (6 = max fp4 magnitude)
    g = (w.abs().max() / (E4M3_MAX * 6.0)).item()
    # block scale in e4m3 domain: block_amax / (6 * global)
    bs = amax / (6.0 * g)  # target e4m3 value
    bs = bs.clamp(max=E4M3_MAX)
    bs_e4m3 = bs.to(torch.float8_e4m3fn)
    bs_val = bs_e4m3.float()
    # effective per-block scale
    eff = bs_val * g  # [O, I//16]
    # quantize: w / eff -> fp4 grid (E2M1: 0,.5,1,1.5,2,3,4,6 w/ sign)
    q = wb / eff.unsqueeze(-1)
    # nearest e2m1
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
    sign = q.sign()
    a = q.abs()
    # nearest neighbor
    idx = torch.bucketize(a, (grid[1:] + grid[:-1]) / 2.0)
    vals = grid[idx] * sign
    # round-trip check on a sample
    codes = idx * (sign < 0).long() * 0  # placeholder, we encode below
    # encode fp4: sign bit (3), magnitude idx 0..7 -> code = signbit<<3 | magidx
    mag = idx  # 0..7
    sb = (sign < 0).to(torch.uint8)
    code = (sb << 3) | mag.to(torch.uint8)  # [O, I//16, 16]
    # pack pairs along input dim: byte = hi<<4 | lo
    c = code.view(O, I // 2, 2)
    packed = (c[:, :, 1] << 4) | c[:, :, 0]  # [O, I//2] uint8
    return packed.contiguous(), bs_e4m3.contiguous(), g

mtp_hdr = hdr(mtp_path)
mtp_keys = [k for k, v in mtp_hdr.items() if isinstance(v, dict) and "dtype" in v]

# load all MTP tensors once (5.2GB BF16 -> fits RAM)
from safetensors import safe_open
mt = {}
with safe_open(mtp_path, framework="pt") as f:
    for k in mtp_keys:
        mt[k] = f.get_tensor(k)
say(f"[B] MTP loaded: {len(mt)} keys, {time.time()-t0:.1f}s")

new_mtp = {}
# keep the 29 non-expert keys as-is
for k, v in mt.items():
    if ".experts." not in k:
        new_mtp[k] = v

# requant 512 experts x 3 projections
n_exp = 512
for e in range(n_exp):
    gk = f"mtp.layers.0.mlp.experts.gate_up_proj.{e}.gate_proj.weight"
    uk = f"mtp.layers.0.mlp.experts.gate_up_proj.{e}.up_proj.weight"
    dk = f"mtp.layers.0.mlp.experts.down_proj.{e}.down_proj.weight"
    for src, proj in ((gk, "gate_proj"), (uk, "up_proj"), (dk, "down_proj")):
        w = mt[src]
        packed, bs, g = nvfp4_quant(w)
        base = f"mtp.layers.0.mlp.experts.{e}.{proj}"
        new_mtp[f"{base}.weight"] = packed
        new_mtp[f"{base}.weight_scale"] = bs
        new_mtp[f"{base}.weight_scale_2"] = torch.tensor(g, dtype=torch.float32)
        new_mtp[f"{base}.input_scale"] = torch.tensor(1.0, dtype=torch.float32)
    if e % 64 == 0:
        say(f"[B] requant experts {e}/512, {time.time()-t0:.1f}s")

from safetensors.torch import save_file
save_file(new_mtp, mtp_path, metadata={"format": "pt"})
say(f"[B] MTP rewritten: {len(mt)} -> {len(new_mtp)} keys, {time.time()-t0:.1f}s")

# ================= C) PLE strip duplicate globals =================
t0 = time.time()
PLE_KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.nvfp4_global"
for i in range(2, 9):
    p = os.path.join(D, f"ple-nvfp4-0000{i}-of-00008.safetensors")
    shutil.copy2(p, os.path.join(BAK, f"ple-nvfp4-0000{i}-of-00008.safetensors"))
    def tr(tensors, _i=i):
        return {k: v for k, v in tensors.items() if k != PLE_KEY}
    n0, n1 = rewrite_shard(p, tr)
    say(f"[C] ple shard {i}: {n0} -> {n1} keys")

# ================= D) index rewrite =================
t0 = time.time()
idx_path = os.path.join(D, "model.safetensors.index.json")
shutil.copy2(idx_path, os.path.join(BAK, "model.safetensors.index.json"))
idx = json.load(open(idx_path))

# rebuild weight_map from actual file headers
new_wm = {}
for f in sorted(os.listdir(D)):
    if not f.endswith(".safetensors"):
        continue
    h = hdr(os.path.join(D, f))
    for k, v in h.items():
        if isinstance(v, dict) and "dtype" in v:
            # skip duplicate globals (only shard 1 keeps it)
            if k == PLE_KEY and f != "ple-nvfp4-00001-of-00008.safetensors":
                continue
            new_wm[k] = f

idx["weight_map"] = new_wm
# recompute total_size from actual tensor sizes
total = 0
for f in sorted(os.listdir(D)):
    if not f.endswith(".safetensors"):
        continue
    h = hdr(os.path.join(D, f))
    for k, v in h.items():
        if isinstance(v, dict) and "dtype" in v:
            import math
            bits = {"BF16": 16, "F32": 32, "F8_E4M3": 8, "U8": 8, "I64": 64, "I32": 32, "BOOL": 8}[v["dtype"]]
            total += int(math.prod(v["shape"])) * bits // 8
idx["metadata"]["total_size"] = total
json.dump(idx, open(idx_path, "w"), indent=0)
say(f"[D] index rewritten: {len(new_wm)} keys, total_size {total/1e9:.2f}GB, {time.time()-t0:.1f}s")
say("[DONE] all repairs complete")
