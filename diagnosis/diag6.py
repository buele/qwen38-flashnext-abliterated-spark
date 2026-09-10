import os, struct, json, collections
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open
say=print

ORCA="/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47="/models/Qwen3.8-Flash-Next-hibrid47"

def hdr(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q", f.read(8))[0]
        raw=json.loads(f.read(n))
    return {k:v for k,v in raw.items() if isinstance(v,dict) and "dtype" in v}

# 1) exact keys in orca ple shard1 and h47 ple shard1
for tag,d in [("ORCA",ORCA),("H47",H47)]:
    h=hdr(os.path.join(d,"ple-nvfp4-00001-of-00008.safetensors"))
    say(f"== {tag} ple shard1: {len(h)} keys ==")
    for k in sorted(h):
        v=h[k]
        say("   %s  %s %s" % (k, v["dtype"], v["shape"]))

# 2) dtype census per file for all 8 orca ple shards
say("== ORCA ple shards key census ==")
for i in range(1,9):
    h=hdr(os.path.join(ORCA,"ple-nvfp4-0000%d-of-00008.safetensors" % i))
    cen=collections.Counter((k.rsplit(".",1)[1], v["dtype"]) for k,v in h.items())
    say("   shard%d: %s  total=%d" % (i, dict(cen), len(h)))

# 3) sample values from shard1
h=hdr(os.path.join(ORCA,"ple-nvfp4-00001-of-00008.safetensors"))
k0=sorted(h)[0]
with safe_open(os.path.join(ORCA,"ple-nvfp4-00001-of-00008.safetensors"), framework="pt") as fh:
    t=fh.get_tensor(k0)
say("first key %s: %s %s" % (k0, t.dtype, tuple(t.shape)))
if t.numel()<20:
    say("   values: %s" % t.float().tolist())
else:
    say("   nonzero: %d/%d  absmax=%.5g" % (int((t!=0).sum()), t.numel(), t.float().abs().max()))

# 4) cache files
for f in sorted(os.listdir("/cache/ple-mmap")):
    p=os.path.join("/cache/ple-mmap",f)
    say("cache file: %s %.3fGiB" % (f, os.path.getsize(p)/2**30))
