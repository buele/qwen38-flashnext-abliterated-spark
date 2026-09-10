import glob, json
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open

def P(*a):
    print(*a, flush=True)

ORCA = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"

P("== L0 bad-key breakdown: which projs, which dtypes, sample values ==")
for p in sorted(glob.glob(ORCA + "/model-*.safetensors")):
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if ".layers.0.mlp.experts." not in k:
                continue
            t = f.get_tensor(k)
            bad = None
            if t.dtype == torch.float8_e4m3fn:
                b = t.view(torch.uint8)
                nb = ((b == 0x7F) | (b == 0xFF)).sum().item()
                if nb:
                    bad = "E4M3 nan_bytes=%d" % nb
            elif t.dtype in (torch.float32, torch.bfloat16):
                x = t.float()
                if torch.isnan(x).any() or torch.isinf(x).any():
                    bad = "NAN/INF"
                elif t.numel() == 1 and float(x.abs().max()) > 1e-2:
                    bad = "SCALE_TOO_BIG val=%.4g" % float(x)
            if bad:
                P("  %s | %s | %s" % (p.split("/")[-1], k, bad))

P("")
P("== L0 e0/e1 full dump of every key + val ==")
want = set()
for p in sorted(glob.glob(ORCA + "/model-*.safetensors")):
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if ".layers.0.mlp.experts.0." in k or ".layers.0.mlp.experts.1." in k:
                want.add((p, k))
for p, k in sorted(want):
    with safe_open(p, framework="pt") as f:
        t = f.get_tensor(k)
        if t.numel() == 1:
            P("  %-72s %-9s scalar=%.8g" % (k.replace("model.", ""), t.dtype, float(t)))
        else:
            x = t.float()
            P("  %-72s %-9s shape=%s amax=%.4g nan=%d" % (k.replace("model.", ""), t.dtype, list(t.shape), x.abs().max(), torch.isnan(x).sum()))

P("")
P("== same for h47 L0 e0/e1 ==")
H47 = "/models/Qwen3.8-Flash-Next-hibrid47"
want = set()
for p in sorted(glob.glob(H47 + "/model-*.safetensors")):
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if ".layers.0.mlp.experts.0." in k or ".layers.0.mlp.experts.1." in k:
                want.add((p, k))
for p, k in sorted(want)[:30]:
    with safe_open(p, framework="pt") as f:
        t = f.get_tensor(k)
        if t.numel() == 1:
            P("  %-72s %-9s scalar=%.8g" % (k.replace("model.", ""), t.dtype, float(t)))
        else:
            x = t.float()
            P("  %-72s %-9s shape=%s amax=%.4g" % (k.replace("model.", ""), t.dtype, list(t.shape), x.abs().max()))
