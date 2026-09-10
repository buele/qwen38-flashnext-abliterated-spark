import os, struct, json, numpy as np, torch
torch.set_grad_enabled(False)
from safetensors import safe_open
say=print

ORCA="/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
CACHE="/cache/ple-mmap/orca47-language_model_model_layers_1_ple_ple_embedding-320001536x160-bfloat16.bin"

def hdr(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q", f.read(8))[0]
        raw=json.loads(f.read(n))
    return {k:v for k,v in raw.items() if isinstance(v,dict) and "dtype" in v}

# 1) raw byte histogram of cache: is it all zeros?
sz=os.path.getsize(CACHE)
say(f"cache size={sz/2**30:.3f}GiB")
with open(CACHE,"rb") as f:
    f.seek(0); head=f.read(4096)
    nz_head=sum(1 for b in head if b!=0)
    say(f"first 4KB nonzero bytes: {nz_head}")
    # sample 20 random 64KB blocks
    import random
    random.seed(2)
    nz_total=0; tot=0
    for _ in range(20):
        off=random.randrange(0,sz-65536)
        f.seek(off); blk=f.read(65536)
        nz_total+=sum(1 for b in blk if b!=0); tot+=len(blk)
    say(f"random 20x64KB: nonzero {nz_total}/{tot} = {nz_total/tot*100:.4f}%")

# 2) PLE shard1 weight tensor stats (U8 packed) - is source all zeros?
h=hdr(os.path.join(ORCA,"ple-nvfp4-00001-of-00008.safetensors"))
wk=[k for k in h if k.endswith(".weight")][0]
sk=[k for k in h if k.endswith(".weight_scale")][0]
gk=[k for k in h if k.endswith("nvfp4_global")][0]
say(f"shard1 keys: {sorted(h)}")
with safe_open(os.path.join(ORCA,"ple-nvfp4-00001-of-00008.safetensors"), framework="pt") as fh:
    W=fh.get_tensor(wk); S=fh.get_tensor(sk); G=fh.get_tensor(gk)
say(f"shard1 W: {W.dtype} {tuple(W.shape)} nonzero={int((W!=0).sum())}/{W.numel()} ({(W!=0).float().mean()*100:.2f}%)")
say(f"shard1 S: {S.dtype} {tuple(S.shape)} min={S.float().min():.4g} max={S.float().max():.4g}")
say(f"shard1 G: {G.item():.8g}")
# dequant a few rows of shard1
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
d0=deq(W[:8],S[:8],G)
say(f"shard1 first 8 rows dequant: std={d0.std():.5g} absmax={d0.abs().max():.5g} nonzero={(d0!=0).float().mean()*100:.2f}%")

# 3) check cache tail (last rows from shard 8, the ones we stripped duplicate global from)
with open(CACHE,"rb") as f:
    f.seek(sz-3200); tail=f.read(3200)
u=np.frombuffer(tail,dtype=np.uint16)
f32=(u.astype(np.uint32)<<16).view(np.float32)
say(f"cache tail 10 rows: nonzero={np.count_nonzero(f32)}/{len(f32)} absmax={np.abs(f32).max():.5g}")

# 4) cache rows corresponding to shard1 start
with open(CACHE,"rb") as f:
    f.seek(0); b=f.read(320*8)
u=np.frombuffer(b,dtype=np.uint16)
f32=(u.astype(np.uint32)<<16).view(np.float32)
say(f"cache first 8 rows: nonzero={np.count_nonzero(f32)}/{len(f32)} absmax={np.abs(f32).max():.5g}")

# 5) what does the ENGINE log say about ple mmap? (grep from host later)
say("DIAG5 DONE")
