import json, struct, os, glob, collections

BASE = "/models"
DIRS = {
  "ORCA": BASE + "/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4",
  "H47":  BASE + "/Qwen3.8-Flash-Next-hibrid47",
  "FP8":  BASE + "/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE",
  "PLEORCA": BASE + "/ple-nvfp4-orca",
}

def hdr(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = json.loads(f.read(n))
    return {k: v for k, v in raw.items() if isinstance(v, dict) and "dtype" in v}

def stats(tag, path):
    try:
        h = hdr(path)
    except Exception as e:
        print(tag, "HDR ERR", repr(e)); return
    keys = list(h.keys())
    exp = [k for k in keys if ".mlp.experts." in k]
    non = [k for k in keys if ".mlp.experts." not in k]
    print(f"### {tag} {os.path.basename(path)}: total={len(keys)} expert={len(exp)} nonexpert={len(non)}")
    if exp:
        print("  exp prefixes:", dict(collections.Counter(k.split('.mlp.experts')[0] for k in exp)))
        print("  exp suffixes:", dict(collections.Counter(k.rsplit('.',1)[1] for k in exp)))
        print("  exp dtypes  :", dict(collections.Counter(h[k]['dtype'] for k in exp)))
        for suf in sorted(set(k.rsplit('.',1)[1] for k in exp)):
            s = [k for k in exp if k.rsplit('.',1)[1] == suf]
            print(f"  exp {suf}: n={len(s)} shape={h[s[0]]['shape']}")
        print("  exp sample  :", sorted(exp)[:4])
    for k in sorted(non):
        print(f"  NON {k}: {h[k]['dtype']} {h[k]['shape']}")

def idx_and_cfg(tag, d):
    try:
        wm = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
        mtpk = [k for k in wm if k.startswith("mtp.")]
        print(f"[{tag}] index: total={len(wm)} mtp_entries={len(mtpk)}")
        if mtpk:
            print("  idx mtp suffixes:", dict(collections.Counter(k.rsplit('.',1)[1] for k in mtpk)))
            print("  idx mtp prefixes:", dict(collections.Counter(k.split('.mlp.experts')[0] for k in mtpk if '.mlp.experts.' in k)))
            print("  idx mtp files   :", dict(collections.Counter(wm[k] for k in mtpk)))
    except Exception as e:
        print(f"[{tag}] index ERR {e!r}")
    try:
        c = json.load(open(d + "/config.json"))
        qc = c.get("quantization_config", {})
        ql = qc.get("quantized_layers", {})
        mtpq = {k: v for k, v in ql.items() if "mtp" in k}
        print(f"[{tag}] quant_algo={qc.get('quant_algo')} group={qc.get('group_size')} ql={len(ql)} mtp_ql={json.dumps(mtpq)}")
        for k, v in c.items():
            if any(s in k.lower() for s in ("mtp","nextn","predict","draft","num_hidden_layers")):
                print(f"[{tag}] cfg {k} = {json.dumps(v)[:200]}")
    except Exception as e:
        print(f"[{tag}] cfg ERR {e!r}")

for tag, d in DIRS.items():
    print(f"==== {tag} dir ====")
    try:
        fs = sorted(os.listdir(d))
    except Exception as e:
        print("  ERR", repr(e)); continue
    for f in fs:
        p = os.path.join(d, f)
        if os.path.isfile(p):
            print(f"  {f}  {os.path.getsize(p)/2**30:.2f}G")
    for m in sorted(glob.glob(d + "/*mtp*")):
        if os.path.isfile(m):
            stats(tag, m)
    idx_and_cfg(tag, d)
    print()
print("PROBE DONE")
