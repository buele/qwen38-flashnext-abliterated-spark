import json, glob, re, gc
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open

def P(*a):
    print(*a, flush=True)

ORCA = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47 = "/models/Qwen3.8-Flash-Next-hibrid47"

P("########## SCRIPT B: structural + config + PLE diff ##########")

o_idx = json.load(open(ORCA + "/model.safetensors.index.json"))["weight_map"]
h_idx = json.load(open(H47 + "/model.safetensors.index.json"))["weight_map"]

def norm(k):
    k = k.replace("model.language_model.", "").replace("model.", "")
    k = re.sub(r"layers\.\d+\.", "layers.N.", k)
    k = re.sub(r"experts\.\d+\.", "experts.N.", k)
    return k

o_t, h_t = {}, {}
for k in o_idx:
    o_t[norm(k)] = o_t.get(norm(k), 0) + 1
for k in h_idx:
    h_t[norm(k)] = h_t.get(norm(k), 0) + 1

P("== templates ONLY in ORCA (count | template) ==")
for t in sorted(set(o_t) - set(h_t)):
    P("  %5d | %s" % (o_t[t], t))
P("== templates ONLY in H47 ==")
for t in sorted(set(h_t) - set(o_t)):
    P("  %5d | %s" % (h_t[t], t))

P("\n== config quantized_layers: entry diff for expert layer + mtp ==")
oc = json.load(open(ORCA + "/config.json"))
hc = json.load(open(H47 + "/config.json"))
oq = oc["quantization_config"]["quantized_layers"]
hq = hc["quantization_config"]["quantized_layers"]
P("orca entries:", len(oq), "h47 entries:", len(hq))
P("orca type:", type(oq).__name__, "h47 type:", type(hq).__name__)

def show(d, pat, label):
    if isinstance(d, dict):
        hits = {k: v for k, v in d.items() if pat in k}
        for k, v in sorted(hits.items())[:4]:
            P("  %s %s -> %s" % (label, k, v))
    else:
        for e in d:
            s = json.dumps(e)
            if pat in s:
                P("  %s %s" % (label, s))
                break

show(oq, "layers.10.mlp.experts", "ORCA")
show(hq, "layers.10.mlp.experts", "H47 ")
show(oq, "mtp.layers.0", "ORCA")
show(hq, "mtp.layers.0", "H47 ")
if isinstance(oq, dict):
    P("  ORCA expert-layer entry keys:", sorted(oq.get("model.layers.10.mlp.experts", oq.get("layers.10.mlp.experts", {}))) if isinstance(oq.get("model.layers.10.mlp.experts", oq.get("layers.10.mlp.experts")), dict) else "?")
    P("  sample orca keys:", list(oq.keys())[:3])
    P("  sample h47 keys:", list(hq.keys())[:3])

P("\n== h47 PLE shards FULL census ==")
for p in sorted(glob.glob(H47 + "/*ple*.safetensors")):
    with safe_open(p, framework="pt") as f:
        ks = sorted(f.keys())
        P("  %s: %d keys" % (p.split("/")[-1], len(ks)))
        for k in ks:
            sl = f.get_slice(k)
            P("      %s  %s %s" % (k.replace("model.language_model.", "")[:80], sl.get_dtype(), list(sl.get_shape())))

P("\n== orca PLE shards FULL census ==")
for p in sorted(glob.glob(ORCA + "/*ple*.safetensors")):
    with safe_open(p, framework="pt") as f:
        ks = sorted(f.keys())
        P("  %s: %d keys" % (p.split("/")[-1], len(ks)))
        for k in ks:
            sl = f.get_slice(k)
            P("      %s  %s %s" % (k.replace("model.language_model.", "")[:80], sl.get_dtype(), list(sl.get_shape())))

P("\n== C-section fixed: structural value compare (flatten) ==")
o_open, h_open = {}, {}
def get(D, idx, k, cache):
    f = idx[k]
    if f not in cache:
        cache[f] = safe_open(D + "/" + f, framework="pt")
    return cache[f].get_tensor(k)

PAIRS = [
    ("model.language_model.embed_tokens.weight", None),
    ("lm_head.weight", None),
    ("model.language_model.layers.35.self_attn.q_proj.weight", None),
    ("model.language_model.layers.35.self_attn.k_proj.weight", None),
    ("model.language_model.layers.35.self_attn.v_proj.weight", None),
    ("model.language_model.layers.35.self_attn.o_proj.weight", None),
    ("model.language_model.layers.10.linear_attn.in_proj_qkv.weight", None),
    ("model.language_model.layers.10.linear_attn.out_proj.weight", None),
    ("model.language_model.layers.10.mlp.gate.weight", None),
    ("model.language_model.layers.10.mlp.shared_expert.gate_proj.weight", None),
    ("model.layers.35.self_attn.q_proj.weight", None),
]
for ok, _ in PAIRS:
    ok2 = ok if ok in o_idx else None
    hk2 = None
    for cand in (ok, ok.replace("model.language_model.", "model."), ok.replace("model.language_model.", "model.")):
        if cand in h_idx:
            hk2 = cand
            break
    if ok2 is None:
        for cand in (ok.replace("model.language_model.", "model."), ok.replace("model.", "model.language_model.")):
            if cand in o_idx:
                ok2 = cand
                break
    if ok2 is None or hk2 is None:
        P("  %-64s SKIP (missing in %s)" % (ok[:64], "orca" if ok2 is None else "h47"))
        continue
    try:
        a = get(ORCA, o_idx, ok2, o_open).float().flatten()
        b = get(H47, h_idx, hk2, h_open).float().flatten()
    except Exception as e:
        P("  %-64s ERR %s" % (ok[:64], str(e)[:70]))
        continue
    if a.shape != b.shape or a.numel() < 2:
        P("  %-64s shape mismatch %s vs %s" % (ok[:64], a.shape, b.shape))
        continue
    va, vb = a - a.mean(), b - b.mean()
    d = (va.norm() * vb.norm()).item()
    corr = (va @ vb).item() / d if d > 0 else float("nan")
    md = (a - b).abs().mean().item()
    P("  %-64s corr=%.4f md=%.3e amax_o=%.4g amax_h=%.4g std_o=%.4g std_h=%.4g" % (
        ok[:64], corr, md, a.abs().max(), b.abs().max(), a.std(), b.std()))
    del a, b, va, vb

P("\n== L32-37 full-attn: all key names (orca) ==")
for L in range(32, 38):
    ks = sorted(k for k in o_idx if ".layers.%d.self_attn." % L in k)
    P("  L%d: %s" % (L, [k.split("layers.%d." % L)[1][:30] for k in ks]))
    for k in ks:
        sl = safe_open(ORCA + "/" + o_idx[k], framework="pt").get_slice(k) if False else None
    # dtype quick
    byfile = {}
    for k in ks:
        byfile.setdefault(o_idx[k], []).append(k)
    for f, kk in byfile.items():
        with safe_open(ORCA + "/" + f, framework="pt") as fh:
            for k in kk:
                sl = fh.get_slice(k)
                P("      %-50s %-8s %s" % (k.split("layers.%d." % L)[1], sl.get_dtype(), list(sl.get_shape())))

P("SCRIPT B DONE")
