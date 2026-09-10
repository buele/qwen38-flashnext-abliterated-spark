import json, struct, os, collections
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open
say=print

ORCA="/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47="/models/Qwen3.8-Flash-Next-hibrid47"

def hdr(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q", f.read(8))[0]
        if n > 500_000_000: raise ValueError(f"insane header {n}")
        raw=json.loads(f.read(n))
    return {k:v for k,v in raw.items() if isinstance(v,dict) and "dtype" in v}

# 1) FULL census of orca 00012 & 00013 (the two repaired/problem shards)
for fn in ["model-00012-of-00017.safetensors","model-00013-of-00017.safetensors"]:
    h=hdr(os.path.join(ORCA,fn))
    say(f"== ORCA {fn}: total={len(h)} ==")
    # categorize
    cat=collections.Counter()
    for k,v in h.items():
        if ".mlp.experts." in k:
            L=int(k.split(".layers.")[1].split(".")[0])
            suf=k.rsplit(".",1)[1]
            cat[(f"L{L}","expert",suf,v["dtype"])]+=1
        elif ".layers." in k:
            L=k.split(".layers.")[1].split(".")[0]
            cat[(f"L{L}","attn/other",k.split(".self_attn.")[-1][:20] if ".self_attn." in k else k.split(".layers.")[-1][:20],"")]+=1
        else:
            cat[("other","","","")]+=1
    for key,n in sorted(cat.items()):
        say(f"   {key}: {n}")

# 2) sample actual expert keys in 00012 for L32
h12=hdr(os.path.join(ORCA,"model-00012-of-00017.safetensors"))
e32=sorted(k for k in h12 if ".layers.32.mlp.experts." in k)
say(f"== L32 expert keys in 00012: {len(e32)} ==")
for k in e32[:10]: say(f"   {k} {h12[k][dtype]} {h12[k][shape]}")
e31=sorted(k for k in h12 if ".layers.31.mlp.experts." in k)
say(f"== L31 expert keys in 00012: {len(e31)} ==")
for k in e31[:6]: say(f"   {k} {h12[k][dtype]} {h12[k][shape]}")

# 3) what dtype/scale_2 do the L32 experts carry (if any)
with safe_open(os.path.join(ORCA,"model-00012-of-00017.safetensors"), framework="pt") as fh:
    for k in e32[:3]:
        t=fh.get_tensor(k)
        say(f"   RAW {k}: {t.dtype} {tuple(t.shape)} min={t.float().min():.5g} max={t.float().max():.5g}")
    # check scale_2 value if exists
    s2keys=[k for k in e32 if k.endswith("weight_scale_2")]
    for k in s2keys[:3]:
        t=fh.get_tensor(k); say(f"   scale_2 {k}: {t.item():.6g}")
    pkkeys=[k for k in e32 if k.endswith("weight_packed")]
    say(f"   L32 weight_packed keys: {len(pkkeys)}")
    glkeys=[k for k in e32 if k.endswith("weight_global_scale")]
    say(f"   L32 weight_global_scale keys: {len(glkeys)}")
    for k in glkeys[:3]:
        t=fh.get_tensor(k); say(f"   global {k}: {t.item():.6g}")

# 4) L38 remainder in 00013
h13=hdr(os.path.join(ORCA,"model-00013-of-00017.safetensors"))
e38=sorted(k for k in h13 if ".layers.38.mlp.experts." in k)
say(f"== L38 expert keys in 00013: {len(e38)} ==")
for k in e38[:6]: say(f"   {k} {h13[k][dtype]} {h13[k][shape]}")

# 5) orca vs h47 ple shard1 md5
import hashlib
def md5(p):
    hsh=hashlib.md5()
    with open(p,"rb") as f:
        while True:
            b=f.read(1<<24)
            if not b: break
            hsh.update(b)
    return hsh.hexdigest()
o=md5(os.path.join(ORCA,"ple-nvfp4-00001-of-00008.safetensors"))
hh=md5(os.path.join(H47,"ple-nvfp4-00001-of-00008.safetensors"))
say(f"== PLE shard1 md5: orca={o[:16]} h47={hh[:16]} identical={o==hh} ==")

# 6) count ALL expert weight_scale_2 keys across orca (main only) for the inversion fix
total=0; byfile=collections.Counter()
for f in sorted(os.listdir(ORCA)):
    if not f.endswith(".safetensors"): continue
    h=hdr(os.path.join(ORCA,f))
    for k in h:
        if ".mlp.experts." in k and not k.startswith("mtp.") and k.endswith("weight_scale_2"):
            total+=1; byfile[f]+=1
say(f"== orca main-expert weight_scale_2 keys: total={total} ==")
for f,n in sorted(byfile.items()): say(f"   {f}: {n}")
say("DIAG4 DONE")
