import glob, struct, json, os, shutil
import torch
torch.set_grad_enabled(False)
from safetensors import safe_open

def P(*a):
    print(*a, flush=True)

ORCA = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47 = "/models/Qwen3.8-Flash-Next-hibrid47"
BAK = "/models/orca-fix5-bak"
os.makedirs(BAK, exist_ok=True)

P("### FIX A: PLE nvfp4_global patch (in-place 4 bytes per shard) ###")
# engine: dequant = FP4[pb] * scales * global. orca global is 20000x too big.
# h47 global = 3.3242362e-05 is the proven-good value for this engine stack.
NEW_GLOBAL = 3.324236240587197e-05
bak_dir = BAK + "/ple_globals"
os.makedirs(bak_dir, exist_ok=True)
for p in sorted(glob.glob(ORCA + "/ple-nvfp4-*.safetensors")):
    nm = os.path.basename(p)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    gk = [k for k in hdr if k.endswith("nvfp4_global")]
    if not gk:
        P("  %s: no global key (expected for shards 2-8), skip" % nm)
        continue
    k = gk[0]
    s, e = hdr[k]["data_offsets"]
    off = base + s
    with open(p, "r+b") as f:
        f.seek(off)
        old = struct.unpack("<f", f.read(4))[0]
        f.seek(off)
        f.write(struct.pack("<f", NEW_GLOBAL))
    shutil.copy2(p, bak_dir + "/" + nm)
    P("  %s: global %.8g -> %.8g (backup saved)" % (nm, old, NEW_GLOBAL))

P("")
P("### FIX B: inject 73728 main-expert input_scale from h47 into orca shards ###")
vals = json.load(open(BAK + "/../orca-fix4-bak/h47_input_scales.json"))
P("loaded %d h47 input_scale values" % len(vals))
o_idx = json.load(open(ORCA + "/model.safetensors.index.json"))["weight_map"]
h_idx = json.load(open(H47 + "/model.safetensors.index.json"))["weight_map"]

need = {}
for k in vals:
    if k.startswith("model.language_model.") or k.startswith("language_model.model."):
        core = k.split("layers.", 1)[1]
        for pref in ("model.language_model.layers.", "model.layers."):
            need[pref + core] = vals[k]

targets = [k for k in need if k in h_idx and (k not in o_idx)]
P("candidate keys to inject (not yet in orca index): %d" % len(targets))

# group by orca file: use the sibling .weight's file for each expert proj
byfile = {}
for k in targets:
    wk = k[:-len(".input_scale")] + ".weight"
    f = o_idx.get(wk) or h_idx.get(wk)
    byfile.setdefault(f, []).append(k)
P("files to touch: %d" % len(byfile))

bak_files = BAK + "/shard_copies"
os.makedirs(bak_files, exist_ok=True)

from safetensors.torch import load_file, save_file

n_added = 0
for f, keys in sorted(byfile.items()):
    path = ORCA + "/" + f
    shutil.copy2(path, bak_files + "/" + f)
    ten = load_file(path)
    for k in keys:
        ten[k] = torch.tensor(need[k], dtype=torch.float32)
        n_added += 1
    save_file(ten, path, metadata={"format": "pt"})
    del ten
P("injected %d input_scale scalars across %d files" % (n_added, len(byfile)))

P("")
P("### FIX C: index rewrite ###")
all_keys = set()
for p in sorted(glob.glob(ORCA + "/model-*.safetensors")) + sorted(glob.glob(ORCA + "/model-mtp.safetensors")) + sorted(glob.glob(ORCA + "/ple-nvfp4-*.safetensors")):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for k in hdr:
        if k != "__metadata__":
            all_keys.add((k, os.path.basename(p)))
wm = {}
for k, f in all_keys:
    wm[k] = f
idxp = ORCA + "/model.safetensors.index.json"
shutil.copy2(idxp, BAK + "/model.safetensors.index.json")
json.dump({"metadata": {"total_size": json.load(open(idxp))["metadata"]["total_size"]}, "weight_map": wm}, open(idxp, "w"))
P("index rewritten: %d keys" % len(wm))

P("")
P("### VERIFY: files, values, index consistency ###")
ok = True
# globals
for p in sorted(glob.glob(ORCA + "/ple-nvfp4-*.safetensors")):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    for k in hdr:
        if k.endswith("nvfp4_global"):
            s, e = hdr[k]["data_offsets"]
            with open(p, "rb") as f:
                f.seek(base + s)
                P("  %s global=%.8g" % (os.path.basename(p), struct.unpack("<f", f.read(4))[0]))

# input_scale spot check
o_idx2 = json.load(open(idxp))["weight_map"]
P("index size now: %d (was 228777)" % len(o_idx2))
checks = ["model.language_model.layers.0.mlp.experts.0.down_proj.input_scale",
          "model.language_model.layers.47.mlp.experts.511.up_proj.input_scale",
          "model.layers.10.mlp.experts.123.gate_proj.input_scale"]
for k in checks:
    if k not in o_idx2:
        P("  MISSING in index:", k)
        ok = False
        continue
    with safe_open(ORCA + "/" + o_idx2[k], framework="pt") as f:
        v = float(f.get_tensor(k))
    ref = vals.get(k.replace("model.language_model.", "model.").replace("model.layers.", "model.language_model.layers."), vals.get(k))
    P("  %s = %.8g (ref %s)" % (k.split("layers.", 1)[1], v, "ok" if ref and abs(ref - v) < 1e-9 else "??"))

# file-vs-index diff
for p in sorted(glob.glob(ORCA + "/model-*.safetensors")) + sorted(glob.glob(ORCA + "/model-mtp.safetensors")) + sorted(glob.glob(ORCA + "/ple-nvfp4-*.safetensors")):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    fkeys = set(k for k in hdr if k != "__metadata__")
    nm = os.path.basename(p)
    ikeys = set(k for k, f in o_idx2.items() if f == nm)
    if fkeys != ikeys:
        P("  MISMATCH %s: file-only=%d index-only=%d" % (nm, len(fkeys - ikeys), len(ikeys - fkeys)))
        ok = False
P("ALL FILES MATCH INDEX" if ok else "INCONSISTENCY FOUND")
P("FIX5 DONE")
