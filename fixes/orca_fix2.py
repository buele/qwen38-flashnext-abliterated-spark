import json, struct, os, shutil, collections, torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_grad_enabled(False)
BASE="/models"
ORCA=BASE+"/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47=BASE+"/Qwen3.8-Flash-Next-hibrid47"
FP8=BASE+"/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE"
BAK=BASE+"/orca-fix2-bak"
os.makedirs(BAK, exist_ok=True)

def hdr(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q", f.read(8))[0]
        raw=json.loads(f.read(n))
    return {k:v for k,v in raw.items() if isinstance(v,dict) and "dtype" in v}

def say(*a): print(*a, flush=True)

# ---------- 0) backups (into MOUNTED dir, survive container) ----------
for fn in ["config.json","model-mtp.safetensors","model.safetensors.index.json"]:
    src=os.path.join(ORCA,fn); dst=os.path.join(BAK,fn)
    if not os.path.exists(dst):
        shutil.copy2(src,dst); say(f"[bak] saved {fn}")
    else:
        say(f"[bak] already exists {fn}")

# ---------- 1) h47 mtp reference ----------
H18=os.path.join(H47,"model-00018-of-00021.safetensors")
h18=hdr(H18)
mtp18={k:v for k,v in h18.items() if k.startswith("mtp.")}
is_keys=sorted(k for k in mtp18 if k.endswith(".input_scale"))
say(f"[h47-00018] mtp keys={len(mtp18)} input_scale={len(is_keys)}")
H19=os.path.join(H47,"model-00019-of-00021.safetensors")
mtp19={k:v for k,v in hdr(H19).items() if k.startswith("mtp.")}
say(f"[h47-00019] mtp keys={sorted(mtp19)}")
assert len(is_keys)==1536, f"unexpected input_scale count {len(is_keys)}"

is_vals={}
with safe_open(H18, framework="pt") as f:
    for k in is_keys: is_vals[k]=f.get_tensor(k)
vv=torch.cat([v.float().reshape(-1) for v in is_vals.values()])
say(f"[h47 input_scale] n={vv.numel()} min={vv.min():.6g} max={vv.max():.6g} mean={vv.mean():.6g} distinct={len(vv.unique())}")
say(f"[h47 input_scale] sample shapes: {[(k.split('.experts.')[-1], tuple(is_vals[k].shape), str(is_vals[k].dtype)) for k in is_keys[:3]]}")

# ---------- 2) orca current mtp ----------
MTP=os.path.join(ORCA,"model-mtp.safetensors")
oh=hdr(MTP)
say(f"[orca mtp] current total={len(oh)}")

# ---------- 3) numeric cross-checks ----------
GRID=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
def dequant(w_u8, ws, g):
    O,I2=w_u8.shape
    lo=(w_u8&0xF).long(); hi=(w_u8>>4).long()
    codes=torch.empty(O,I2*2,dtype=torch.long)
    codes[:,0::2]=lo; codes[:,1::2]=hi
    sign=(codes>>3)&1; mag=codes&7
    val=GRID[mag]*torch.where(sign==1,-1.,1.)
    eff=(ws.float()*g.float().reshape(-1)).repeat_interleave(16,dim=1)
    return val*eff

def corr(a,b):
    a=a.reshape(-1).float(); b=b.reshape(-1).float()
    a=a-a.mean(); b=b-b.mean()
    return ((a*b).sum()/(a.norm()*b.norm())).item()

with safe_open(MTP, framework="pt") as f:
    o_dw=f.get_tensor("mtp.layers.0.mlp.experts.0.down_proj.weight")
    o_ds=f.get_tensor("mtp.layers.0.mlp.experts.0.down_proj.weight_scale")
    o_dg=f.get_tensor("mtp.layers.0.mlp.experts.0.down_proj.weight_scale_2")
    o_gw=f.get_tensor("mtp.layers.0.mlp.experts.0.gate_proj.weight")
    o_gs=f.get_tensor("mtp.layers.0.mlp.experts.0.gate_proj.weight_scale")
    o_gg=f.get_tensor("mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_2")
o_down=dequant(o_dw,o_ds,o_dg); o_gate=dequant(o_gw,o_gs,o_gg)
say(f"[orca dequant e0] down {tuple(o_down.shape)} std={o_down.std():.6f} absmax={o_down.abs().max():.4f} | gate {tuple(o_gate.shape)} std={o_gate.std():.6f}")

with safe_open(H18, framework="pt") as f:
    h_dw=f.get_tensor("mtp.layers.0.mlp.experts.0.down_proj.weight")
    h_ds=f.get_tensor("mtp.layers.0.mlp.experts.0.down_proj.weight_scale")
    h_dg=f.get_tensor("mtp.layers.0.mlp.experts.0.down_proj.weight_scale_2")
h_down=dequant(h_dw,h_ds,h_dg)
say(f"[h47 dequant e0] down {tuple(h_down.shape)} std={h_down.std():.6f}")
say(f"[CHECK] corr(orca e0 down, h47 e0 down)   = {corr(o_down,h_down):.4f}")

FP8M=os.path.join(FP8,"model-mtp.safetensors")
with safe_open(FP8M, framework="pt") as f:
    fused_gu=f.get_tensor("mtp.layers.0.mlp.experts.gate_up_proj")
    fused_dn=f.get_tensor("mtp.layers.0.mlp.experts.down_proj")
f0g=fused_gu[0,:640].float(); f0d=fused_dn[0].float()
del fused_gu, fused_dn
say(f"[CHECK] corr(orca e0 down, fp8ple e0 down) = {corr(o_down,f0d):.4f}")
say(f"[CHECK] corr(orca e0 gate, fp8ple e0 gate) = {corr(o_gate,f0g):.4f}")

# ---------- 4) patch config: add h47's 6 mtp quantized_layers ----------
cfgp=os.path.join(ORCA,"config.json")
c=json.load(open(cfgp))
h=json.load(open(os.path.join(H47,"config.json")))
hql=h["quantization_config"]["quantized_layers"]
mtpql={k:v for k,v in hql.items() if "mtp" in k}
assert len(mtpql)==6, f"h47 mtp ql = {len(mtpql)}"
ql=c["quantization_config"]["quantized_layers"]
before=len(ql)
for k,v in mtpql.items(): ql[k]=v
json.dump(c, open(cfgp,"w"), indent=2, ensure_ascii=False)
say(f"[config] quantized_layers {before} -> {len(ql)} (+{len(mtpql)} mtp): {sorted(mtpql)}")

# ---------- 5) rewrite orca mtp file: + input_scale (and any other h47-mtp keys orca lacks) ----------
h47_mtp_all = set(mtp18) | set(mtp19)
orca_mtp_all = set(oh)
missing = sorted(h47_mtp_all - orca_mtp_all)
extra   = sorted(orca_mtp_all - h47_mtp_all)
say(f"[keyset] h47-mtp={len(h47_mtp_all)} orca-mtp={len(orca_mtp_all)} missing_in_orca={len(missing)} extra_in_orca={len(extra)}")
if extra: say(f"[keyset] EXTRA (kept): {extra[:10]}")

cur={}
with safe_open(MTP, framework="pt") as f:
    for k in oh: cur[k]=f.get_tensor(k)

added=0
for k in missing:
    src = H18 if k in mtp18 else H19
    with safe_open(os.path.join(H47,os.path.basename(src)), framework="pt") as f:
        cur[k]=f.get_tensor(k)
    added+=1
say(f"[mtp file] adding {added} keys from h47 ({len(missing)-len([k for k in missing if k.endswith('.input_scale')])} non-input_scale)")

save_file(cur, MTP, metadata={"format":"pt"})
del cur
nh=hdr(MTP)
say(f"[mtp file] {len(oh)} -> {len(nh)} keys")
assert len(nh)==len(oh)+added

# ---------- 6) patch index ----------
idxp=os.path.join(ORCA,"model.safetensors.index.json")
idx=json.load(open(idxp))
wm=idx["weight_map"]
for k in missing: wm[k]="model-mtp.safetensors"
json.dump(idx, open(idxp,"w"), indent=0)
mtpn=sum(1 for k in wm if k.startswith("mtp."))
say(f"[index] total={len(wm)} mtp={mtpn}")

# ---------- 7) final verification ----------
nh2=hdr(MTP)
exp=[k for k in nh2 if ".mlp.experts." in k]
say(f"[VERIFY] mtp file: total={len(nh2)} expert={len(exp)} nonexpert={len(nh2)-len(exp)}")
say(f"[VERIFY] expert suffixes: {dict(collections.Counter(k.rsplit('.',1)[1] for k in exp))}")
say(f"[VERIFY] expert dtypes:   {dict(collections.Counter(nh2[k]['dtype'] for k in exp))}")
c2=json.load(open(cfgp))
ql2=c2["quantization_config"]["quantized_layers"]
say(f"[VERIFY] config ql={len(ql2)} mtp entries={sorted(k for k in ql2 if 'mtp' in k)}")
idx2=json.load(open(idxp))
wm2=idx2["weight_map"]
m2=[k for k in wm2 if k.startswith("mtp.")]
fileset=set(nh2)
idxset=set(m2)
say(f"[VERIFY] index-mtp={len(m2)} file-mtp={sum(1 for k in fileset if k.startswith('mtp.'))} idx-only={len(idxset-fileset)} file-only={len(fileset-idxset)}")
assert not (idxset-fileset) and not (fileset-idxset), "index/file mtp mismatch!"
say("REPAIR-2 DONE — mtp layout now matches h47 (NVFP4 ModelOpt + input_scale), config declares mtp quantized_layers")
