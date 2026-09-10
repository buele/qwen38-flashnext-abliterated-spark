import glob, struct, json, os, random, shutil
import torch
torch.set_grad_enabled(False)

def P(*a):
    print(*a, flush=True)

ORCA = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47 = "/models/Qwen3.8-Flash-Next-hibrid47"
FP4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])

def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def read_rows(p, shard, rows, suffix):
    hdr, base = header(p)
    out = b""
    with open(p, "rb") as f:
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            kk = k.split("ngram_embedding.")[-1]
            if kk == "nvfp4_shard_%d.%s" % (shard, suffix):
                w = v["shape"][1]
                for r in rows:
                    f.seek(base + v["data_offsets"][0] + r * w)
                    out += f.read(w)
    return out

def dq(packed, scales, nrows):
    b = torch.frombuffer(bytearray(packed), dtype=torch.uint8).view(nrows, -1)
    sc = torch.frombuffer(bytearray(scales), dtype=torch.uint8).view(nrows, -1).view(torch.float8_e4m3fn).float()
    lo = FP4[(b & 0xF).long()].float()
    hi = FP4[(b >> 4).long()].float()
    vals = torch.stack([lo, hi], dim=-1).reshape(nrows, -1)
    return vals * sc.repeat_interleave(vals.shape[1] // sc.shape[1], dim=1)

random.seed(123)
P("== LSQ global: orca content (g=1) vs h47 dequant (actual global), 150 rows x 4 shards ==")
num = 0.0
den = 0.0
g_h47 = None
po = sorted(glob.glob(ORCA + "/ple-nvfp4-*.safetensors"))
ph = sorted(glob.glob(H47 + "/ple-nvfp4-*.safetensors"))
for shard in [0, 2, 4, 7]:
    rows = sorted(random.sample(range(40000192), 150))
    pk_o = read_rows(po[shard], shard, rows, "packed")
    sc_o = read_rows(po[shard], shard, rows, "scales")
    pk_h = read_rows(ph[shard], shard, rows, "packed")
    sc_h = read_rows(ph[shard], shard, rows, "scales")
    if not pk_o or not pk_h:
        P("  shard %d: no data (keys split across files differently), skip" % shard)
        continue
    ao = dq(pk_o, sc_o, len(rows))
    ah = dq(pk_h, sc_h, len(rows))
    # h47 actual global
    hdr, base = header(ph[0])
    for k, v in hdr.items():
        if k.endswith("nvfp4_global"):
            with open(ph[0], "rb") as f:
                f.seek(base + v["data_offsets"][0])
                g_h47 = struct.unpack("<f", f.read(4))[0]
    b_h = ah * g_h47
    a = ao.flatten()
    b = b_h.flatten()
    num += float(a @ b)
    den += float(a @ a)
    P("  shard %d: orca_g1 std=%.5f  h47 std=%.6f" % (shard, ao.std(), b_h.std()))
c = num / den if den > 0 else float("nan")
P("LSQ global = %.8e (h47 global ref = %.8e)" % (c, g_h47))

P("== sanity: apply c to orca content, compare stats to h47 ==")
rows = sorted(random.sample(range(40000192), 300))
pk_o = read_rows(po[0], 0, rows, "packed")
sc_o = read_rows(po[0], 0, rows, "scales")
pk_h = read_rows(ph[0], 0, rows, "packed")
sc_h = read_rows(ph[0], 0, rows, "scales")
ao = dq(pk_o, sc_o, len(rows))
ah = dq(pk_h, sc_h, len(rows))
a2 = ao * c
b2 = ah * g_h47
va, vb = a2.flatten() - a2.mean(), b2.flatten() - b2.mean()
P("  orca(c): std=%.6f amax=%.6f | h47: std=%.6f amax=%.6f | corr=%.6f" % (
    a2.std(), a2.abs().max(), b2.std(), b2.abs().max(),
    ((va @ vb) / (va.norm() * vb.norm())).item()))

P("== PATCH: write LSQ global into orca shard1 ==")
p1 = ORCA + "/ple-nvfp4-00001-of-00008.safetensors"
hdr, base = header(p1)
for k, v in hdr.items():
    if k.endswith("nvfp4_global"):
        off = base + v["data_offsets"][0]
        with open(p1, "r+b") as f:
            f.seek(off)
            old = struct.unpack("<f", f.read(4))[0]
            f.seek(off)
            f.write(struct.pack("<f", c))
        P("  global %.8e -> %.8e" % (old, c))
# verify
hdr, base = header(p1)
for k, v in hdr.items():
    if k.endswith("nvfp4_global"):
        with open(p1, "rb") as f:
            f.seek(base + v["data_offsets"][0])
            P("  verify readback: %.8e" % struct.unpack("<f", f.read(4))[0])
P("PATCH DONE")
