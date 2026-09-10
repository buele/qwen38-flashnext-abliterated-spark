#!/usr/bin/env python3
"""Fix3: invert main-expert weight_scale_2 (lychee stored divisor 2688/amax;
vLLM ModelOpt expects multiplicative amax/2688). In-place 4-byte patches.
MTP file untouched (fix2 already wrote correct multiplicative values)."""
import json, struct, os, shutil
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open

ORCA="/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
BAK="/models/orca-fix3-bak"
os.makedirs(BAK, exist_ok=True)
say=print

def hdr_full(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q", f.read(8))[0]
        raw=json.loads(f.read(n))
    return n, raw

# ---- collect all main-expert scale_2 keys with offsets ----
patch_plan={}   # file -> list of (key, abs_offset, old_value)
originals={}    # key -> old float
for fn in sorted(os.listdir(ORCA)):
    if not fn.endswith(".safetensors") or fn.startswith("ple-") or "mtp" in fn:
        continue
    p=os.path.join(ORCA,fn)
    hlen, raw=hdr_full(p)
    tensors={k:v for k,v in raw.items() if isinstance(v,dict) and "dtype" in v}
    entries=[]
    for k,v in tensors.items():
        if (k.startswith("model.language_model.layers.")
                and ".mlp.experts." in k and k.endswith(".weight_scale_2")
                and v["dtype"]=="F32"):
            begin=v["data_offsets"][0]
            abs_off=8+hlen+begin
            entries.append((k,abs_off))
    if entries:
        patch_plan[fn]=entries
        say(f"{fn}: {len(entries)} scale_2 keys")

total=sum(len(v) for v in patch_plan.values())
say(f"TOTAL scale_2 keys to patch: {total} (expect 73728 = 48L x 512E x 3P)")

# ---- backup originals + in-place patch ----
for fn, entries in patch_plan.items():
    p=os.path.join(ORCA,fn)
    with open(p,"r+b") as f:
        for key,off in entries:
            f.seek(off); b=f.read(4)
            old=struct.unpack("<f", b)[0]
            if old <= 1.0:
                say(f"  WARN {fn} {key}: value {old:.6g} already <=1, skipping")
                originals[key]=old
                continue
            originals[key]=old
            new=1.0/old
            f.seek(off); f.write(struct.pack("<f", new))
    say(f"patched {fn}")

json.dump(originals, open(os.path.join(BAK,"scale2_originals.json"),"w"))
say(f"backup: {len(originals)} original values -> {BAK}/scale2_originals.json")

# ---- verification ----
say("== VERIFY 1: all main scale_2 now multiplicative ==")
bad=0
for fn, entries in patch_plan.items():
    p=os.path.join(ORCA,fn)
    hlen,raw=hdr_full(p)
    tensors={k:v for k,v in raw.items() if isinstance(v,dict) and "dtype" in v}
    with open(p,"rb") as f:
        for key,off in entries:
            f.seek(off); v=struct.unpack("<f", f.read(4))[0]
            if v<=0 or v>1.0:
                bad+=1; say(f"  BAD {key}: {v:.6g}")
say(f"bad values: {bad} (expect 0)")

say("== VERIFY 2: dequant sample experts across layers ==")
GRID=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
def deq(w_u8, ws, g):
    O,I2=w_u8.shape
    lo=(w_u8&0xF).long(); hi=(w_u8>>4).long()
    codes=torch.empty(O,I2*2,dtype=torch.long)
    codes[:,0::2]=lo; codes[:,1::2]=hi
    sign=(codes>>3)&1; mag=codes&7
    val=GRID[mag]*torch.where(sign==1,-1.,1.)
    eff=(ws.float()*g.float().reshape(-1)).repeat_interleave(16,dim=1)
    return val*eff

def get(key):
    for fn in sorted(os.listdir(ORCA)):
        if not fn.endswith(".safetensors"): continue
        p=os.path.join(ORCA,fn)
        try:
            with safe_open(p, framework="pt") as fh:
                return fh.get_tensor(key)
        except Exception:
            continue
    return None

import random
random.seed(1)
for L in [0,7,19,31,35,38,47]:
    for e in [0, 255, 511]:
        base=f"model.language_model.layers.{L}.mlp.experts.{e}.gate_proj"
        w=get(base+".weight"); s=get(base+".weight_scale"); g=get(base+".weight_scale_2")
        if w is None or s is None or g is None:
            say(f"  L{L}e{e}: MISSING"); continue
        d=deq(w,s,g)
        ok = (not torch.isnan(d).any()) and d.std()<0.5 and d.abs().max()<10
        say(f"  L{L}e{e} gate: std={d.std():.5f} absmax={d.abs().max():.4f} nan={torch.isnan(d).any().item()} {'OK' if ok else '** BAD **'}")

say("== VERIFY 3: full-attn self_attn projections sane (no 448-grid) ==")
for L in [3,7,11,15,19,23,27,31,35,39,43,47]:
    for proj in ["q_proj","k_proj","v_proj","o_proj"]:
        t=get(f"model.language_model.layers.{L}.self_attn.{proj}.weight")
        if t is None: say(f"  L{L} {proj}: MISSING"); continue
        f_=t.float()
        flag="" if (f_.abs().max()<50 and not torch.isnan(f_).any()) else " ** BAD **"
        if flag or L in (3,35,47):
            say(f"  L{L} {proj}: dtype={t.dtype} absmax={f_.abs().max():.3f} std={f_.std():.5f}{flag}")

say("== VERIFY 4: MTP untouched (still multiplicative) ==")
for k in ["mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_2",
          "mtp.layers.0.mlp.experts.511.down_proj.weight_scale_2"]:
    t=get(k)
    if t is not None: say(f"  {k}: {t.item():.6g} (must be small ~1e-5..1e-3)")

say("== VERIFY 5: PLE cache value sanity (3000 random rows) ==")
CACHE="/cache/ple-mmap/orca47-language_model_model_layers_1_ple_ple_embedding-320001536x160-bfloat16.bin"
import numpy as np
sz=os.path.getsize(CACHE)
rows=sz//320
nanrows=0; mx=0.0; sd_accum=[]
with open(CACHE,"rb") as f:
    for _ in range(3000):
        r=random.randrange(rows)
        f.seek(r*320); buf=f.read(320)
        a=np.frombuffer(buf,dtype=np.float32).astype(np.float32)
        # BF16 stored in upper? file is bfloat16 raw -> reinterpret as uint16 then convert
        u=np.frombuffer(buf,dtype=np.uint16)
        # bf16 -> f32: value = uint16 << 16
        f32=(u.astype(np.uint32)<<16).view(np.float32)
        if np.isnan(f32).any() or np.isinf(f32).any(): nanrows+=1
        mx=max(mx,float(np.abs(f32).max())); sd_accum.append(float(f32.std()))
say(f"  rows={rows} sampled=3000 nan/inf rows={nanrows} max|v|={mx:.4g} mean-std={sum(sd_accum)/len(sd_accum):.4g}")

say("FIX3 DONE")
