import re, json, glob, struct, os, subprocess, random
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open

def P(*a):
    print(*a, flush=True)

ORCA = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47 = "/models/Qwen3.8-Flash-Next-hibrid47"
BAK = "/models/orca-fix4-bak"
os.makedirs(BAK, exist_ok=True)

P("### E1: h47 main-expert input_scale extraction (tiny memory) ###")
h_idx = json.load(open(H47 + "/model.safetensors.index.json"))["weight_map"]
o_idx = json.load(open(ORCA + "/model.safetensors.index.json"))["weight_map"]

orca_w = [k for k in o_idx if ".mlp.experts." in k and k.endswith(".weight") and not k.startswith("mtp.")]
need = {k[:-len(".weight")] + ".input_scale" for k in orca_w}
present = {k for k in need if k in h_idx}
P("orca main expert weights: %d | input_scale needed: %d | present in h47: %d" % (len(orca_w), len(need), len(present)))
missing = need - present
for k in sorted(missing)[:5]:
    P("  MISSING e.g.:", k)

vals = {}
byfile = {}
for k in present:
    byfile.setdefault(h_idx[k], []).append(k)
for f, ks in sorted(byfile.items()):
    with safe_open(H47 + "/" + f, framework="pt") as fh:
        for k in ks:
            v = fh.get_slice(k)
            if v.get_shape() == []:
                vals[k] = float(fh.get_tensor(k))
            else:
                vals[k] = [float(x) for x in fh.get_tensor(k).flatten()]
json.dump(vals, open(BAK + "/h47_input_scales.json", "w"))
P("extracted %d keys; scalar sample: %s" % (len(vals), {k: vals[k] for k in sorted(vals)[:3]}))

h_main_is = [k for k in h_idx if ".mlp.experts." in k and k.endswith(".input_scale") and not k.startswith("mtp.")]
P("h47 main input_scale total: %d (expect 73728)" % len(h_main_is))

P("### E2: sampled embed/lm_head/router/PLE compare (small mem) ###")
random.seed(0)

def get_rows(D, idx, k, nrow=2048):
    with safe_open(D + "/" + idx[k], framework="pt") as f:
        sl = f.get_slice(k)
        shp = list(sl.get_shape())
        if len(shp) != 2:
            t = f.get_tensor(k)
            return t.float().flatten()[:: max(1, t.numel() // 300000)].clone()
        idxs = sorted(random.sample(range(shp[0]), min(nrow, shp[0])))
        rows = []
        for i in idxs:
            rows.append(sl[i])
        return torch.stack(rows).float().flatten()

PAIRS = ["model.language_model.embed_tokens.weight", "lm_head.weight",
         "model.language_model.layers.0.mlp.gate.weight",
         "model.language_model.layers.10.mlp.gate.weight",
         "model.language_model.layers.47.mlp.gate.weight",
         "model.language_model.layers.10.mlp.shared_expert.gate_proj.weight",
         "model.language_model.layers.35.self_attn.q_proj.weight",
         "model.language_model.layers.10.linear_attn.in_proj_qkv.weight"]
for k in PAIRS:
    ok = k if k in o_idx else None
    hk = k if k in h_idx else None
    if not ok or not hk:
        P("  %-58s SKIP o=%s h=%s" % (k[:58], ok is not None, hk is not None))
        continue
    a = get_rows(ORCA, o_idx, ok)
    b = get_rows(H47, h_idx, hk)
    if a.numel() != b.numel():
        b = get_rows(H47, h_idx, hk, nrow=a.numel() // max(1, list(o_idx.values()).count(1)))
    n = min(a.numel(), b.numel())
    a2, b2 = a[:n], b[:n]
    va, vb = a2 - a2.mean(), b2 - b2.mean()
    d = (va.norm() * vb.norm()).item()
    corr = (va @ vb).item() / d if d > 0 else float("nan")
    P("  %-58s n=%d corr=%.6f md=%.3e" % (k[:58], n, corr, (a2 - b2).abs().mean().item()))
    del a, b, a2, b2, va, vb

P("### E3: PLE raw compare (48 rows) ###")
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])

def ple_raw(D, nrows=48):
    p = sorted(glob.glob(D + "/*ple*.safetensors"))[0]
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    out = {}
    with open(p, "rb") as f:
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            if k.endswith(".packed"):
                out["packed"] = f.read(nrows * v["shape"][1])
            elif k.endswith(".scales"):
                out["scales"] = f.read(nrows * v["shape"][1])
            elif k.endswith("nvfp4_global"):
                out["global"] = struct.unpack("<f", f.read(4))[0]
    return out

def ple_deq(o, nrows=48):
    b = torch.frombuffer(bytearray(o["packed"]), dtype=torch.uint8).view(nrows, -1)
    sc = torch.frombuffer(bytearray(o["scales"]), dtype=torch.uint8).view(nrows, -1).view(torch.float8_e4m3fn).float()
    lo = FP4[b & 0xF].float()
    hi = FP4[b >> 4].float()
    vals = torch.stack([lo, hi], dim=-1).reshape(nrows, -1)
    per = vals.shape[1] // sc.shape[1]
    return vals * sc.repeat_interleave(per, dim=1) * o["global"]

oo = ple_raw(ORCA)
oh = ple_raw(H47)
P("orca global=%.8f h47 global=%.8f" % (oo.get("global", -1), oh.get("global", -1)))
ao, ah = ple_deq(oo), ple_deq(oh)
va, vb = ao.flatten() - ao.mean(), ah.flatten() - ah.mean()
P("PLE corr=%.6f maxdiff=%.3g orca_std=%.5f h47_std=%.5f" % (
    ((va @ vb) / (va.norm() * vb.norm())).item(), (ao - ah).abs().max().item(), ao.std(), ah.std()))

P("### E4: orca MTP vs h47 MTP input_scale identical? ###")
om = {k: v for k, v in o_idx.items() if k.startswith("mtp.") and k.endswith(".input_scale")}
hm = {k: v for k, v in h_idx.items() if k.startswith("mtp.") and k.endswith(".input_scale")}
P("orca mtp input_scale: %d | h47: %d" % (len(om), len(hm)))
same = diff = 0
for k in sorted(set(om) & set(hm)):
    with safe_open(ORCA + "/" + om[k], framework="pt") as f1, safe_open(H47 + "/" + hm[k], framework="pt") as f2:
        a = float(f1.get_tensor(k))
        b = float(f2.get_tensor(k))
    if a == b:
        same += 1
    else:
        diff += 1
        if diff <= 3:
            P("  DIFF", k, a, b)
P("mtp input_scale same=%d diff=%d" % (same, diff))
P("SCRIPT E DONE")
