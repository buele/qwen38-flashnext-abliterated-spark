import glob, json, sys, gc
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open

def P(*a):
    print(*a, flush=True)

ORCA = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47 = "/models/Qwen3.8-Flash-Next-hibrid47"

FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])

def sample_t(f, k, cap=8_000_000):
    """Load at most `cap` elements from tensor k (chunked on dim0)."""
    sl = f.get_slice(k)
    shp = list(sl.get_shape())
    n = shp[0] if shp else 1
    tot = 1
    for s in shp:
        tot *= s
    if tot <= cap:
        return f.get_tensor(k)
    rows = max(1, cap // max(1, tot // n))
    got = []
    for i in range(0, n, max(1, int(n / 6))):
        got.append(sl[i:i + rows])
        if sum(g.numel() for g in got) >= cap:
            break
    return torch.cat(got, 0) if got else f.get_tensor(k)

def grid_frac(x):
    nz = x[x != 0]
    if nz.numel() < 32:
        return 0.0, 0.0
    nz = nz[:200000]
    q = nz.to(torch.float8_e4m3fn).float()
    return (q == nz).float().mean().item(), nz.abs().max().item()

P("== A. anomaly sweep (non-expert, sampled) ==")
anoms = 0
for p in sorted(glob.glob(ORCA + "/*.safetensors")):
    nm = p.split("/")[-1]
    try:
        with safe_open(p, framework="pt") as f:
            for k in f.keys():
                if ".mlp.experts." in k:
                    continue
                sl = f.get_slice(k)
                dt = sl.get_dtype()
                if dt not in ("BF16", "F32", "F16"):
                    continue
                t = sample_t(f, k)
                x = t.float()
                nan = torch.isnan(x).sum().item()
                inf = torch.isinf(x).sum().item()
                amx = x.abs().max().item() if x.numel() else 0.0
                std = x.std().item() if x.numel() > 1 else 0.0
                fl = []
                if nan or inf:
                    fl.append("NAN=%d INF=%d" % (nan, inf))
                if amx > 1e3:
                    fl.append("HUGE %.2e" % amx)
                if x.numel() >= 64:
                    fr, gm = grid_frac(x)
                    if fr > 0.98:
                        fl.append("E4M3GRID fr=%.3f gmax=%.1f" % (fr, gm))
                if std > 60:
                    fl.append("STD %.0f" % std)
                if fl:
                    anoms += 1
                    P("  %s | %s | %s | %s" % (nm, k[:90], dt, " ".join(fl)))
                del t, x
        gc.collect()
    except Exception as e:
        P("  FILE ERR", nm, str(e)[:100])
P("anomaly count:", anoms)

P("== B. expert scale sanity layers {0,10,20,35,47} ==")
for L in [0, 10, 20, 35, 47]:
    bad = checked = 0
    for p in sorted(glob.glob(ORCA + "/model-*.safetensors")):
        with safe_open(p, framework="pt") as f:
            for k in f.keys():
                if (".layers.%d.mlp.experts." % L) not in k:
                    continue
                t = f.get_tensor(k)
                checked += 1
                if t.dtype == torch.float8_e4m3fn:
                    b = t.view(torch.uint8)
                    if (b == 0x7F).any() or (b == 0xFF).any():
                        bad += 1
                elif t.dtype in (torch.float32, torch.bfloat16):
                    x = t.float()
                    if torch.isnan(x).any() or torch.isinf(x).any() or (t.numel() == 1 and float(x.abs().max()) > 1e-2):
                        bad += 1
    P("  layer %2d: checked %d, bad %d" % (L, checked, bad))

P("== C. cross-model structural compare (sampled, corr only) ==")
o_idx = json.load(open(ORCA + "/model.safetensors.index.json"))["weight_map"]
h_idx = json.load(open(H47 + "/model.safetensors.index.json"))["weight_map"]
o_open, h_open = {}, {}

def get(D, idx, k, cache):
    f = idx[k]
    if f not in cache:
        cache[f] = safe_open(D + "/" + f, framework="pt")
    return cache[f].get_tensor(k)

INTEREST = ["A_log", "dt_bias", "conv1d", "norm.weight", "embed_tokens", "lm_head",
            "mlp.gate.weight", "shared_expert", "hyper_connection", "in_proj_a", "in_proj_b",
            "final_norm", "rotary_emb"]
cnt = 0
for k in sorted(o_idx):
    if ".mlp.experts." in k:
        continue
    if not any(s in k for s in INTEREST):
        continue
    hk = k if k in h_idx else None
    if hk is None:
        continue
    try:
        a = get(ORCA, o_idx, k, o_open).float()
        b = get(H47, h_idx, hk, h_open).float()
    except Exception as e:
        P("  ERR", k[:70], str(e)[:60])
        continue
    if a.shape != b.shape or a.numel() < 2:
        continue
    va, vb = a - a.mean(), b - b.mean()
    d = (va.norm() * vb.norm()).item()
    corr = (va @ vb).item() / d if d > 0 else float("nan")
    md = (a - b).abs().mean().item()
    scale = max(1e-9, b.abs().mean().item())
    if corr < 0.99 or md > 0.05 * scale:
        P("  %-58s corr=%.4f md=%.3e relmd=%.2f%s" % (k.replace("model.language_model.", "")[:58], corr, md, md / scale, " <-- MISMATCH"))
    cnt += 1
    del a, b
P("  compared:", cnt)

P("== D. PLE table compare (locate via index) ==")
ple_o = {k: f for k, f in o_idx.items() if "ngram_embedding" in k}
ple_h = {k: f for k, f in h_idx.items() if "ngram_embedding" in k}
P("  orca ple keys:", len(ple_o), "| h47:", len(ple_h))

def deq(idx, D, n=48):
    ks = [k for k in idx if k.endswith(".packed")]
    ks.sort()
    k0 = ks[0]
    with safe_open(D + "/" + idx[k0], framework="pt") as f:
        pk = f.get_slice(k0)
        packed = pk[0:n].clone()
        sk = [k for k in idx if k.endswith(".scales") and idx[k] == idx[k0]][0]
        scales = f.get_slice(sk)[0:n].clone()
        gk = [k for k in idx if k.endswith("nvfp4_global") and idx[k] == idx[k0]]
        g = f.get_tensor(gk[0]).item() if gk else 1.0
    pb = packed.view(torch.uint8)
    lo = FP4[pb & 0xF].float()
    hi = FP4[pb >> 4].float()
    vals = torch.stack([lo, hi], -1).reshape(n, -1)
    sc = scales.float().repeat_interleave(16, 1)
    return vals * sc * g, g, scales.float()

do, go, so = deq(o_idx, ORCA)
dh, gh, sh = deq(h_idx, H47)
a, b = do.flatten(), dh.flatten()
va, vb = a - a.mean(), b - b.mean()
corr = (va @ vb).item() / (va.norm() * vb.norm()).item()
P("  global o=%.6f h=%.6f | dequant corr=%.5f o_amax=%.4f o_std=%.5f h_amax=%.4f h_std=%.5f" % (go, gh, corr, a.abs().max(), a.std(), b.abs().max(), b.std()))

P("== E. PLE nan-bytes all shards ==")
seen = set()
for k, f in sorted(ple_o.items()):
    if f in seen:
        continue
    seen.add(f)
    with safe_open(ORCA + "/" + f, framework="pt") as fo:
        for kk in fo.keys():
            t = fo.get_tensor(kk)
            if t.dtype == torch.float8_e4m3fn:
                bb = t.view(torch.uint8)
                nb = ((bb == 0x7F) | (bb == 0xFF)).sum().item()
                P("  %s | %s | E4M3 nan_bytes=%d/%d" % (f, kk.split("ngram_embedding.")[1], nb, bb.numel()))
            elif t.dtype in (torch.float32, torch.bfloat16) and t.numel() == 1:
                P("  %s | %s | scalar=%.8f" % (f, kk.split("ngram_embedding.")[1], float(t)))

P("== F. mtp structural vs h47 ==")
for L in [0, 48]:
    for base, parts in [("linear_attn", ["A_log", "dt_bias", "norm.weight"]), ("mlp", ["gate.weight"])]:
        for part in parts:
            nk = "mtp.layers.%d.%s.%s" % (L, base, part)
            if nk not in o_idx or nk not in h_idx:
                continue
            a = get(ORCA, o_idx, nk, o_open).float().flatten()
            b = get(H47, h_idx, nk, h_open).float().flatten()
            if a.shape == b.shape and a.numel() > 1:
                va, vb = a - a.mean(), b - b.mean()
                corr = (va @ vb).item() / (va.norm() * vb.norm()).item()
                P("  %-52s corr=%.4f" % (nk, corr))
for c in list(o_open.values()) + list(h_open.values()):
    try:
        c.__exit__(None, None, None)
    except Exception:
        pass
P("SCAN DONE")
