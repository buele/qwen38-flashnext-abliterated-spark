import json, struct, os, collections
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open
say=print

ORCA="/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
FP8="/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE"
H47="/models/Qwen3.8-Flash-Next-hibrid47"

def hdr(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q", f.read(8))[0]
        if n > 500_000_000: raise ValueError(f"insane header {n}")
        raw=json.loads(f.read(n))
    return {k:v for k,v in raw.items() if isinstance(v,dict) and "dtype" in v}

def locate(d, key):
    for f in sorted(os.listdir(d)):
        if not f.endswith(".safetensors"): continue
        try: h=hdr(os.path.join(d,f))
        except Exception: continue
        if key in h: return os.path.join(d,f)
    return None

def get(d, key):
    p=locate(d,key)
    if p is None: return None
    with safe_open(p, framework="pt") as fh:
        return fh.get_tensor(key)

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

def st(t,name):
    f=t.float()
    say(f"  {name}: dtype={t.dtype} shape={tuple(t.shape)} min={f.min():.5g} max={f.max():.5g} mean={f.mean():.5g} std={f.std():.5g}")

def corr(a,b):
    a=a.reshape(-1).float(); b=b.reshape(-1).float(); a=a-a.mean(); b=b-b.mean()
    return ((a*b).sum()/(a.norm()*b.norm())).item()

say("===== L10 e0 gate_proj: ORCA vs FP8PLE vs H47 =====")
K="model.language_model.layers.10.mlp.experts.0.gate_proj"
for tag,d,wskey,gkey in [("ORCA",ORCA,".weight_scale",".weight_scale_2"),
                          ("FP8PLE",FP8,".weight_scale",".weight_global_scale"),
                          ("H47",H47,".weight_scale",".weight_scale_2")]:
    w=get(d,K+".weight"); s=get(d,wskey if tag!="ORCA" else ".weight_scale"); 
    # fix: use suffix per tag
    s=get(d,K+( ".weight_scale" if tag!="FP8PLE" else ".weight_scale"))
    g=get(d,K+(gkey))
    if w is None or s is None or g is None:
        say(f"[{tag}] MISSING keys (w={w is None} s={s is None} g={g is None})"); continue
    say(f"[{tag}] raw tensors:")
    st(w,"  packed")
    st(s,"  ws")
    st(g,"  global")
    d1=deq(w,s,g); say(f"[{tag}] dequant e2m1*ws*g: std={d1.std():.5g} absmax={d1.abs().max():.5g}")
    d2=deq(w,s,torch.ones_like(g)); say(f"[{tag}] dequant e2m1*ws   : std={d2.std():.5g} absmax={d2.abs().max():.5g}")

# byte compare orca vs fp8ple packed
ow=get(ORCA,K+".weight"); fw=get(FP8,K+".weight_packed")
if ow is not None and fw is not None:
    say(f"packed identical: {torch.equal(ow,fw)}  (orca {tuple(ow.shape)} vs fp8ple {tuple(fw.shape)})")
os_=get(ORCA,K+".weight_scale"); fs=get(FP8,K+".weight_scale")
if os_ is not None and fs is not None:
    say(f"ws identical: {torch.equal(os_,fs)}")
og=get(ORCA,K+".weight_scale_2"); fg=get(FP8,K+".weight_global_scale")
if og is not None and fg is not None:
    say(f"orca scale_2={og.item():.6g}  fp8ple global={fg.item():.6g}")

# h47 scale_2 value
hg=get(H47,K+".weight_scale_2")
if hg is not None: say(f"h47 scale_2={hg.item():.6g}")
hs=get(H47,K+".weight_scale")
if hs is not None: st(hs,"h47 ws")

# corr of dequants (fp8ple with its g; orca with its g)
fw_=get(FP8,K+".weight_packed"); fs_=get(FP8,K+".weight_scale"); fg_=get(FP8,K+".weight_global_scale")
if fw_ is not None:
    fd=deq(fw_,fs_,fg_)
    od=deq(ow,os_,og)
    say(f"corr(orca_deq_with_own_g, fp8ple_deq_with_own_g) = {corr(od,fd):.4f}")
    fd2=deq(fw_,fs_,torch.ones_like(fg_))
    say(f"corr(orca_deq, fp8ple_deq_no_g) = {corr(od,fd2):.4f}")
    od2=deq(ow,os_,torch.ones_like(og))
    say(f"corr(orca_deq_no_g, fp8ple_deq_no_g) = {corr(od2,fd2):.4f}")
    say(f"corr(orca_deq_no_g, fp8ple_deq_with_g) = {corr(od2,fd):.4f}")

# h47 expert dequant for same layer (weights differ - abliterated vs hibrid) 
hw=get(H47,K+".weight"); hws=get(H47,K+".weight_scale"); hg2=get(H47,K+".weight_scale_2")
if hw is not None:
    hd=deq(hw,hws,hg2)
    say(f"h47 dequant std={hd.std():.5g} (reference: this is what vLLM digests fine)")

say("===== scale_2 / global distribution across orca & fp8ple =====")
for tag,d,gkey in [("ORCA",ORCA,"weight_scale_2"),("FP8PLE",FP8,"weight_global_scale")]:
    vals=[]
    for L in (0,10,20,31,38,47):
        v=get(d,f"model.language_model.layers.{L}.mlp.experts.0.gate_proj.{gkey}")
        if v is not None: vals.append((L,v.item()))
    say(f"[{tag}] L->global: {[(L,f'{x:.4g}') for L,x in vals]}")

say("===== FP8PLE L35 (missing in orca) e0 gate sanity =====")
K35="model.language_model.layers.35.mlp.experts.0.gate_proj"
w=get(FP8,K35+".weight_packed"); s=get(FP8,K35+".weight_scale"); g=get(FP8,K35+".weight_global_scale")
if w is not None:
    d1=deq(w,s,g); d2=deq(w,s,torch.ones_like(g))
    say(f"fp8ple L35e0 gate: deq_with_g std={d1.std():.5g}  deq_no_g std={d2.std():.5g}")
    st(s,"  L35 ws")
    say(f"  L35 global={g.item():.6g}")

say("DIAG3 DONE")
