import glob, json
from safetensors import safe_open

D = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
targets = [
    "layers.10.mlp.experts.0.gate_proj.weight_scale_2",
    "layers.47.mlp.experts.511.down_proj.weight_scale_2",
    "layers.35.self_attn.q_proj.weight_scale",
]
found = {}
for p in sorted(glob.glob(D + "/model-*.safetensors")):
    try:
        with safe_open(p, framework="pt") as f:
            ks = list(f.keys())
            for t in targets:
                if t in found:
                    continue
                for k in ks:
                    if k.endswith(t):
                        v = f.get_tensor(k)
                        if v.numel() == 1:
                            found[t] = (p.split("/")[-1], "scalar %.6e" % float(v))
                        else:
                            found[t] = (p.split("/")[-1], "shape %s first %.6e" % (list(v.shape), float(v.flatten()[0])))
    except Exception as e:
        print("skip", p.split("/")[-1], type(e).__name__, str(e)[:80])
for t in targets:
    print("%-58s %s" % (t, found.get(t, "NOT FOUND")))

idx = json.load(open(D + "/model.safetensors.index.json"))
print("index weight_map size:", len(idx["weight_map"]))

cfg = json.load(open(D + "/config.json"))
def find_ql(o, pref=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "quantized_layers":
                print("quantized_layers at %s count=%d" % (pref + "/" + k, len(v)))
            find_ql(v, pref + "/" + k)
    elif isinstance(o, list):
        pass
find_ql(cfg)
