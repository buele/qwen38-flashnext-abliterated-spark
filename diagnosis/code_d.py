import re, json, glob, struct, os, subprocess
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open

def P(*a):
    print(*a, flush=True)

ORCA = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47 = "/models/Qwen3.8-Flash-Next-hibrid47"
BAK = "/models/orca-fix4-bak"
os.makedirs(BAK, exist_ok=True)

P("### D1: convert_to_nvfp4_moe_kernel_format source ###")
r = subprocess.run(["grep", "-rln", "def convert_to_nvfp4_moe_kernel_format", "/usr/local/lib/python3.12/dist-packages/vllm/"], capture_output=True, text=True)
for path in r.stdout.split():
    srcf = open(path).read()
    m = re.search(r"def convert_to_nvfp4_moe_kernel_format.*?(?=\ndef |\Z)", srcf, re.S)
    if m:
        P("FILE:", path)
        P(m.group(0)[:9500])
        break
else:
    P("NOT FOUND")

P("### D2: select_nvfp4_moe_backend source ###")
r = subprocess.run(["grep", "-rln", "def select_nvfp4_moe_backend", "/usr/local/lib/python3.12/dist-packages/vllm/"], capture_output=True, text=True)
for path in r.stdout.split():
    srcf = open(path).read()
    m = re.search(r"def select_nvfp4_moe_backend.*?(?=\ndef |\Z)", srcf, re.S)
    if m:
        P("FILE:", path)
        P(m.group(0)[:6000])
        break
else:
    P("NOT FOUND")

P("### D4: embed / lm_head / router / shared corr (orca vs h47) ###")
o_idx = json.load(open(ORCA + "/model.safetensors.index.json"))["weight_map"]
h_idx = json.load(open(H47 + "/model.safetensors.index.json"))["weight_map"]

def get(D, idx, k):
    with safe_open(D + "/" + idx[k], framework="pt") as f:
        return f.get_tensor(k)

PAIRS = ["model.language_model.embed_tokens.weight", "lm_head.weight",
         "model.language_model.layers.0.mlp.gate.weight",
         "model.language_model.layers.10.mlp.gate.weight",
         "model.language_model.layers.47.mlp.gate.weight",
         "model.language_model.layers.10.mlp.shared_expert.gate_proj.weight",
         "model.language_model.layers.10.mlp.shared_expert_gate.weight",
         "model.language_model.layers.10.linear_attn.in_proj_qkv.weight",
         "model.language_model.layers.10.linear_attn.out_proj.weight",
         "model.language_model.layers.35.self_attn.q_proj.weight"]
for ok in PAIRS:
    ok2 = ok if ok in o_idx else None
    hk2 = ok if ok in h_idx else None
    if not ok2 or not hk2:
        P("  %-62s SKIP (o:%s h:%s)" % (ok[:62], ok2 is not None, hk2 is not None))
        continue
    a = get(ORCA, o_idx, ok2).float().flatten()
    b = get(H47, h_idx, hk2).float().flatten()
    if a.shape != b.shape:
        P("  %-62s SHAPES %s vs %s" % (ok[:62], list(a.shape), list(b.shape)))
        continue
    va, vb = a - a.mean(), b - b.mean()
    corr = ((va @ vb) / (va.norm() * vb.norm())).item()
    P("  %-62s corr=%.6f md=%.3e" % (ok[:62], corr, (a - b).abs().mean().item()))
    del a, b, va, vb

P("### D5: PLE raw-offset dequant compare (48 rows, shard 1) ###")
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])

def ple_raw(D, nrows=48):
    p = sorted(glob.glob(D + "/*ple*.safetensors"))[0]
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    out = {"file": os.path.basename(p)}
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
P("orca ple file:", oo["file"], "global=%.8f | h47 global=%.8f" % (oo.get("global", -1), oh.get("global", -1)))
ao = ple_deq(oo)
ah = ple_deq(oh)
va, vb = ao.flatten() - ao.mean(), ah.flatten() - ah.mean()
corr = ((va @ vb) / (va.norm() * vb.norm())).item()
P("PLE dequant corr=%.6f maxabsdiff=%.6g orca_std=%.5f h47_std=%.5f" % (
    corr, (ao - ah).abs().max().item(), ao.std(), ah.std()))

P("### D6: h47 input_scale extraction (main experts) ###")
orca_experts = [k for k in o_idx if ".mlp.experts." in k and k.endswith(".weight") and not k.startswith("mtp.")]
P("orca main expert .weight keys:", len(orca_experts))
need = {k[:-7] + ".input_scale" for k in orca_experts}
P("needed input_scale keys:", len(need))
present = need & set(h_idx)
P("present in h47:", len(present), "| missing:", len(need) - len(present))
if len(need) - len(present):
    for k in sorted(need - set(h_idx))[:5]:
        P("  MISSING e.g.:", k)

vals = {}
byfile = {}
for k in present:
    byfile.setdefault(h_idx[k], []).append(k)
for f, ks in sorted(byfile.items()):
    with safe_open(H47 + "/" + f, framework="pt") as fh:
        for k in ks:
            vals[k] = float(fh.get_tensor(k))
json.dump(vals, open(BAK + "/h47_input_scales.json", "w"))
P("extracted:", len(vals), "min=%.6g max=%.6g mean=%.6g" % (min(vals.values()), max(vals.values()), sum(vals.values()) / len(vals)))
ex = sorted(vals)[0]
with safe_open(H47 + "/" + h_idx[ex], framework="pt") as fh:
    sl = fh.get_slice(ex)
    P("exemplar:", ex, "dtype:", sl.get_dtype(), "shape:", list(sl.get_shape()))

h_main_is = [k for k in h_idx if ".mlp.experts." in k and k.endswith(".input_scale") and not k.startswith("mtp.")]
P("h47 main input_scale total:", len(h_main_is), "(expect 73728 = 48L x 512E x 3P)")
P("SCRIPT D DONE")
