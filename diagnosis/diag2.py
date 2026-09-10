import json, struct, os, collections
import torch
torch.set_grad_enabled(False)

ORCA="/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47="/models/Qwen3.8-Flash-Next-hibrid47"
FP8="/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE"
CACHE="/cache/ple-mmap/orca47-language_model_model_layers_1_ple_ple_embedding-320001536x160-bfloat16.bin"

def hdr(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q", f.read(8))[0]
        raw=json.loads(f.read(n))
    return {k:v for k,v in raw.items() if isinstance(v,dict) and "dtype" in v}

def say(*a): print(*a,flush=True)

def build_keyinfo(d):
    ki={}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            for k,v in hdr(os.path.join(d,f)).items():
                ki[k]=(f,v["dtype"],tuple(v["shape"]))
    return ki

from safetensors import safe_open
def get_tensor(d, ki, key):
    f=ki[key][0]
    with safe_open(os.path.join(d,f), framework="pt") as fh:
        return fh.get_tensor(key)

say("== P0: ORCA index dtype census by pattern ==")
oki=build_keyinfo(ORCA)
def pat(k):
    if k.startswith("mtp."): return "mtp"
    if ".mlp.experts." in k: return "main_expert"
    if "linear_attn" in k: return "linear_attn"
    if "indexer" in k: return "indexer"
    if ".ple." in k or k.endswith(".ple") or ".ngram" in k: return "ple"
    if "lm_head" in k: return "lm_head"
    if "embed" in k.lower(): return "embed"
    return "other"
cen=collections.Counter((pat(k),oki[k][1]) for k in oki)
for (p,dt),n in sorted(cen.items()): say(f"  {p:14} {dt:10} {n}")

say("== P0b: linear_attn sample keys ==")
la=sorted(k for k in oki if "linear_attn" in k)
say(f"  count={len(la)}")
for k in la[:8]: say(f"   {k}  {oki[k]}")
la_scales=[k for k in oki if "linear_attn" in k and ("scale" in k)]
say(f"  linear_attn scale keys: {len(la_scales)}")
for k in sorted(la_scales)[:6]: say(f"   {k}  {oki[k]}")

say("== P0c: H47 quantized_layers non-mtp entries ==")
hql=json.load(open(H47+"/config.json"))["quantization_config"]["quantized_layers"]
others=sorted(k for k in hql if "mtp" not in k)
say(f"  h47 ql total={len(hql)} non-mtp={len(others)}")
for k in others[:8]: say(f"   {k} = {hql[k]}")

say("== P0d: ORCA quantized_layers sample ==")
oql=json.load(open(ORCA+"/config.json"))["quantization_config"]["quantized_layers"]
say(f"  total={len(oql)}")
for k in sorted(oql)[:8]: say(f"   {k} = {oql[k]}")

say("== P1: main-expert input_scale counts ==")
hki=build_keyinfo(H47)
h_is=[k for k in hki if ".mlp.experts." in k and not k.startswith("mtp.") and k.endswith("input_scale")]
o_is=[k for k in oki if ".mlp.experts." in k and not k.startswith("mtp.") and k.endswith("input_scale")]
say(f"  h47 main-expert input_scale={len(h_is)}  orca={len(o_is)}")

say("== P2: NaN / numeric checks ==")
def nanstat(t,name):
    f=t.float()
    say(f"  {name}: dtype={t.dtype} shape={tuple(t.shape)} nan={torch.isnan(f).any().item()} inf={torch.isinf(f).any().item()} std={f.std():.5g} absmax={f.abs().max():.5g}")

q35=get_tensor(ORCA,oki,"model.language_model.layers.35.self_attn.q_proj.weight")
nanstat(q35,"orca L35 q_proj (our dequant)")

for k in sorted(k for k in oki if pat(k) in ("lm_head","embed")):
    nanstat(get_tensor(ORCA,oki,k),f"orca {pat(k)} {k.split('.')[-2]+'.'+k.split('.')[-1]}")

for k in sorted(k for k in oki if "linear_attn" in k and k.endswith(".weight") and ("layers.0." in k or "layers.10." in k))[:6]:
    nanstat(get_tensor(ORCA,oki,k),f"orca {k.replace('model.language_model.','')}")

for k in sorted(k for k in oki if "indexer" in k and k.endswith(".weight") and "layers.0." in k)[:3]:
    nanstat(get_tensor(ORCA,oki,k),f"orca {k.replace('model.language_model.','')}")

GRID=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
def dequant_nvfp4(w_u8, ws, g):
    O,I2=w_u8.shape
    lo=(w_u8&0xF).long(); hi=(w_u8>>4).long()
    codes=torch.empty(O,I2*2,dtype=torch.long)
    codes[:,0::2]=lo; codes[:,1::2]=hi
    sign=(codes>>3)&1; mag=codes&7
    val=GRID[mag]*torch.where(sign==1,-1.,1.)
    eff=(ws.float()*g.float().reshape(-1)).repeat_interleave(16,dim=1)
    return val*eff

def corr(a,b):
    a=a.reshape(-1).float(); b=b.reshape(-1).float(); a=a-a.mean(); b=b-b.mean()
    return ((a*b).sum()/(a.norm()*b.norm())).item()

for L in (10,20):
    for e in (0,7):
        base=f"model.language_model.layers.{L}.mlp.experts.{e}"
        try:
            w=get_tensor(ORCA,oki,base+".gate_proj.weight")
            s=get_tensor(ORCA,oki,base+".gate_proj.weight_scale")
            g=get_tensor(ORCA,oki,base+".gate_proj.weight_scale_2")
            dq=dequant_nvfp4(w,s,g)
            nanstat(dq,f"orca L{L} e{e} gate dequant")
        except KeyError as ex:
            say(f"  L{L} e{e} gate: key missing {ex}")

w=get_tensor(ORCA,oki,"mtp.layers.0.mlp.experts.0.gate_proj.weight")
s=get_tensor(ORCA,oki,"mtp.layers.0.mlp.experts.0.gate_proj.weight_scale")
g=get_tensor(ORCA,oki,"mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_2")
nanstat(dequant_nvfp4(w,s,g),"orca mtp e0 gate dequant")

say("== P3: L35 numeric vs FP8PLE (same source model) ==")
fki=build_keyinfo(FP8)
L="model.language_model.layers.35.self_attn."
for p in ("q_proj","k_proj","v_proj","o_proj"):
    a=get_tensor(ORCA,oki,L+p+".weight")
    try:
        b=get_tensor(FP8,fki,L+p+".weight"); bs=get_tensor(FP8,fki,L+p+".weight_scale")
        bd=(b.float()*bs.float()).to(torch.bfloat16) if bs is not None else b.to(torch.bfloat16)
        say(f"  L35 {p}: fp8ple dtype={b.dtype} scale={tuple(bs.shape) if bs is not None else None} corr={corr(a,bd):.4f}")
    except KeyError:
        say(f"  L35 {p}: fp8ple keys missing")

say("== P3b: fp8ple main-expert key samples ==")
fk=sorted(k for k in fki if ".mlp.experts." in k and not k.startswith("mtp."))[:6]
say(f"  {fk}")
if fk:
    k0=fk[0]
    t=get_tensor(FP8,fki,k0)
    say(f"  fp8ple {k0}: dtype={t.dtype} shape={tuple(t.shape)}")

say("== P4: PLE shard structure + cache validation ==")
p1=hdr(os.path.join(ORCA,"ple-nvfp4-00001-of-00008.safetensors"))
for k,v in sorted(p1.items()): say(f"  shard1 {k}: {v['dtype']} {v['shape']}")
wk=[k for k in p1 if k.endswith(".weight")][0]
sk=[k for k in p1 if k.endswith(".weight_scale")][0]
gk=[k for k in p1 if k.endswith("nvfp4_global")][0]
ROWS=p1[wk]["shape"][0]
say(f"  rows/shard={ROWS} total={ROWS*8}")
with safe_open(os.path.join(ORCA,"ple-nvfp4-00001-of-00008.safetensors"), framework="pt") as fh:
    g=fh.get_tensor(gk)
say(f"  global scale = {g.item():.8g}")

def dq_row(W,S,r):
    w=W[r].unsqueeze(0); s=S[r].unsqueeze(0)
    return dequant_nvfp4(w,s,g).squeeze(0)

import random
random.seed(0)
test_rows=[0,1,999,ROWS-1, ROWS, ROWS+5, 2*ROWS+7, 5*ROWS+3, 8*ROWS-1]
cache_size=os.path.getsize(CACHE)
say(f"  cache size={cache_size/2**30:.3f}GiB rows={cache_size//320} (expect {320001536})")
for r in test_rows:
    shard=r//ROWS; lr=r%ROWS
    sf=os.path.join(ORCA,f"ple-nvfp4-0000{shard+1}-of-00008.safetensors")
    with safe_open(sf, framework="pt") as fh:
        W=fh.get_tensor(wk); S=fh.get_tensor(sk)
    d=dq_row(W,S,lr).float()
    with open(CACHE,"rb") as cf:
        cf.seek(r*320); buf=cf.read(320)
    c=torch.frombuffer(bytearray(buf),dtype=torch.bfloat16).float()
    say(f"  row {r} (shard{shard+1}): corr={corr(d,c):.4f} maxdiff={(d-c).abs().max():.5f} cache_nan={torch.isnan(c).any().item()}")

say("== P4b: cache NaN scan (3000 random rows) ==")
nan_rows=0; checked=0
with open(CACHE,"rb") as cf:
    for _ in range(3000):
        r=random.randrange(320001536)
        cf.seek(r*320); buf=cf.read(320)
        c=torch.frombuffer(bytearray(buf),dtype=torch.bfloat16).float()
        checked+=1
        if torch.isnan(c).any() or torch.isinf(c).any(): nan_rows+=1
say(f"  checked={checked} nan/inf rows={nan_rows}")

say("DIAG2 DONE")
