import glob, struct, json, random
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open

def P(*a):
    print(*a, flush=True)

ORCA = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47 = "/models/Qwen3.8-Flash-Next-hibrid47"

P("### F1: PLE packed/scales BYTE compare, orca vs h47, all 8 shards (sampled rows) ###")
def ple_files(D):
    return sorted(glob.glob(D + "/ple-nvfp4-*.safetensors"))

def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

random.seed(42)
for po, ph in zip(ple_files(ORCA), ple_files(H47)):
    ho, base_o = header(po)
    hh, base_h = header(ph)
    same_keys = set(k for k in ho if k != "__metadata__") == set(k for k in hh if k != "__metadata__")
    row_o = [v["shape"][0] for k, v in ho.items() if k.endswith(".packed")][0]
    row_h = [v["shape"][0] for k, v in hh.items() if k.endswith(".packed")][0]
    rows = sorted(random.sample(range(min(row_o, row_h)), 200))
    byts_o, byts_h = b"", b""
    with open(po, "rb") as f, open(ph, "rb") as g:
        for k, v in ho.items():
            if k == "__metadata__" or not k.endswith(".packed"):
                continue
            w = v["shape"][1]
            for r in rows:
                f.seek(base_o + v["data_offsets"][0] + r * w)
                byts_o += f.read(w)
        for k, v in hh.items():
            if k == "__metadata__" or not k.endswith(".packed"):
                continue
            w = v["shape"][1]
            for r in rows:
                g.seek(base_h + v["data_offsets"][0] + r * w)
                byts_h += g.read(w)
    identical = byts_o == byts_h
    P("  %s vs %s | keys_match=%s rows=%d | packed bytes IDENTICAL=%s (len %d vs %d)" % (
        po.split("/")[-1], ph.split("/")[-1], same_keys, len(rows), identical, len(byts_o), len(byts_h)))
    if not identical:
        bo = torch.frombuffer(bytearray(byts_o), dtype=torch.uint8)
        bh = torch.frombuffer(bytearray(byts_h), dtype=torch.uint8)
        P("      byte diff count: %d / %d" % ((bo != bh).sum().item(), bo.numel()))

P("")
P("### F2: scales bytes compare (same rows) ###")
for po, ph in zip(ple_files(ORCA), ple_files(H47)):
    ho, base_o = header(po)
    hh, base_h = header(ph)
    row_o = [v["shape"][0] for k, v in ho.items() if k.endswith(".packed")][0]
    rows = sorted(random.sample(range(row_o), 200))
    so, sh = b"", b""
    with open(po, "rb") as f:
        for k, v in ho.items():
            if k == "__metadata__" or not k.endswith(".scales"):
                continue
            w = v["shape"][1]
            for r in rows:
                f.seek(base_o + v["data_offsets"][0] + r * w)
                so += f.read(w)
    with open(ph, "rb") as f:
        for k, v in hh.items():
            if k == "__metadata__" or not k.endswith(".scales"):
                continue
            w = v["shape"][1]
            for r in rows:
                f.seek(base_h + v["data_offsets"][0] + r * w)
                sh += f.read(w)
    P("  %s scales IDENTICAL=%s" % (po.split("/")[-1], so == sh))

P("")
P("### F3: dequant with correct global, orca vs h47 (48 rows, fixed indexing) ###")
FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])

def ple_deq_file(p, nrows=48, global_override=None):
    h, base = header(p)
    packed = scales = None
    g = None
    with open(p, "rb") as f:
        for k, v in h.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            if k.endswith(".packed"):
                f.seek(base + s)
                packed = f.read(nrows * v["shape"][1])
            elif k.endswith(".scales"):
                f.seek(base + s)
                scales = f.read(nrows * v["shape"][1])
            elif k.endswith("nvfp4_global"):
                f.seek(base + s)
                g = struct.unpack("<f", f.read(4))[0]
    if global_override is not None:
        g = global_override
    b = torch.frombuffer(bytearray(packed), dtype=torch.uint8).view(nrows, -1)
    sc = torch.frombuffer(bytearray(scales), dtype=torch.uint8).view(nrows, -1).view(torch.float8_e4m3fn).float()
    lo = FP4[(b & 0xF).long()].float()
    hi = FP4[(b >> 4).long()].float()
    vals = torch.stack([lo, hi], dim=-1).reshape(nrows, -1)
    per = vals.shape[1] // sc.shape[1]
    return vals * sc.repeat_interleave(per, dim=1) * g, g

po = ple_files(ORCA)[0]
ph = ple_files(H47)[0]
ao, go = ple_deq_file(po)
ah, gh = ple_deq_file(ph)
P("  orca global=%.8g  h47 global=%.8g  ratio=%.1f" % (go, gh, go / gh))
va, vb = ao.flatten() - ao.mean(), ah.flatten() - ah.mean()
P("  dequant(orca_g) vs dequant(h47_g): corr=%.6f" % ((va @ vb) / (va.norm() * vb.norm())).item())
P("  orca amax=%.4g std=%.4g | h47 amax=%.4g std=%.4g" % (ao.abs().max(), ao.std(), ah.abs().max(), ah.std()))
ao2, _ = ple_deq_file(po, global_override=gh)
P("  orca bytes with h47 global: amax=%.4g std=%.4g (target ~ h47)" % (ao2.abs().max(), ao2.std()))
P("  byte-equal check (orca w/ h47_g vs h47): maxdiff=%.6g" % (ao2 - ah).abs().max().item())

P("")
P("### F4: embed/lm_head matched-row corr (same rows both models) ###")
o_idx = json.load(open(ORCA + "/model.safetensors.index.json"))["weight_map"]
h_idx = json.load(open(H47 + "/model.safetensors.index.json"))["weight_map"]
random.seed(7)

def rows(D, idx, k, rws):
    with safe_open(D + "/" + idx[k], framework="pt") as f:
        sl = f.get_slice(k)
        if len(list(sl.get_shape())) != 2:
            t = f.get_tensor(k)
            return t.float().flatten()[:: max(1, t.numel() // 400000)].clone()
        return torch.stack([sl[i] for i in rws]).float().flatten()

for k in ["model.language_model.embed_tokens.weight", "lm_head.weight"]:
    if k not in o_idx or k not in h_idx:
        P("  %s SKIP" % k)
        continue
    nrow = json.load(open(ORCA + "/config.json"))["vocab_size"] if False else 151936
    rws = sorted(random.sample(range(nrow), 3000))
    a = rows(ORCA, o_idx, k, rws)
    b = rows(H47, h_idx, k, rws)
    va, vb = a - a.mean(), b - b.mean()
    P("  %-50s corr=%.6f md=%.3e" % (k, ((va @ vb) / (va.norm() * vb.norm())).item(), (a - b).abs().mean().item()))
    del a, b

P("SCRIPT F DONE")
