import json, glob
from collections import Counter
from safetensors import safe_open

D = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H = "/models/Qwen3.8-Flash-Next-hibrid47"

idx = json.load(open(D + "/model.safetensors.index.json"))["weight_map"]
hidx = json.load(open(H + "/model.safetensors.index.json"))["weight_map"]

# key families for layer 10 (non-expert), orca vs h47
def fam(k):
    return k.replace("model.language_model.", "").replace("model.", "")

o_l10 = sorted(fam(k) for k in idx if ".layers.10." in k and "experts" not in k)
h_l10 = sorted(fam(k) for k in hidx if ".layers.10." in k and "experts" not in k)
print("== orca L10 non-expert keys ==")
for k in o_l10:
    print("  ", k)
print("== h47 L10 non-expert keys (diff only) ==")
so, sh = set(o_l10), set(h_l10)
for k in sorted(sh - so):
    print("  H47-ONLY:", k)
for k in sorted(so - sh):
    print("  ORCA-ONLY:", k)

# dtype census of orca L10 non-expert keys (need real dtypes from files)
need = {}
for k in idx:
    if ".layers.10." in k and "experts" not in k:
        need[k] = idx[k]
byfile = {}
for k, f in need.items():
    byfile.setdefault(f, []).append(k)
print("== orca L10 non-expert dtypes ==")
for f, ks in sorted(byfile.items()):
    with safe_open(D + "/" + f, framework="pt") as fh:
        for k in ks:
            t = fh.get_slice(k)
            print("  %-70s %-9s %s" % (fam(k), t.get_dtype(), list(t.get_shape())))

# h47: what do ITS linear_attn keys look like (quantized? dtypes?)
print("== h47 L10 linear_attn/conv keys dtypes ==")
hl = {k: v for k, v in hidx.items() if ".layers.10." in k and "experts" not in k and ("linear_attn" in k or "conv" in k or "gdn" in k or "in_proj" in k or "out_proj" in k)}
byfile2 = {}
for k, f in hl.items():
    byfile2.setdefault(f, []).append(k)
for f, ks in sorted(byfile2.items()):
    with safe_open(H + "/" + f, framework="pt") as fh:
        for k in ks:
            try:
                t = fh.get_slice(k)
                print("  %-70s %-9s %s" % (fam(k), t.get_dtype(), list(t.get_shape())))
            except Exception as e:
                print("  %-70s ERR %s" % (fam(k), str(e)[:60]))
